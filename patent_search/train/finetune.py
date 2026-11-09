"""Fine-tune the bi-encoder on mined (anchor, positive, hard negative) triplets
(DESIGN.md Part 3A.5).

Loss: Multiple Negatives Ranking (InfoNCE): for each anchor, its positive must score above every
other positive in the batch (in-batch negatives) and above the mined hard negatives.
A plain PyTorch loop over SentenceTransformer's own forward pass, which reproduces .encode()
exactly; this keeps the dependency list short and every step visible.

Local, CPU by default, fixed seeds, deterministic data order. Writes the model plus a training card.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np


def load_triplets(path: Path, max_pairs: int | None, seed: int) -> list[tuple[str, str, str | None]]:
    recs = [json.loads(line) for line in open(path)]
    rng = random.Random(seed)
    rng.shuffle(recs)
    out = []
    per_doc: dict[str, int] = {}
    for r in recs:
        if per_doc.get(r["doc"], 0) >= 6:                 # cap per patent so long patents don't dominate
            continue
        per_doc[r["doc"]] = per_doc.get(r["doc"], 0) + 1
        out.append((r["anchor"], r["positive"], r["negatives"][0] if r["negatives"] else None))
        if max_pairs and len(out) >= max_pairs:
            break
    return out


def finetune(triplets_path: Path, out_dir: Path, base_model: str, base_revision: str, device: str = "cpu",
             epochs: int = 1, batch_size: int = 16, lr: float = 2e-5, warmup: float = 0.1, max_seq_length: int = 256,
             scale: float = 20.0, max_pairs: int | None = None, seed: int = 13, log=print) -> dict:
    import torch
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    data = load_triplets(triplets_path, max_pairs, seed)
    model = SentenceTransformer(base_model, revision=base_revision, device=device,
                                model_kwargs={"dtype": torch.float32})
    model.max_seq_length = max_seq_length
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(data) / batch_size)
    warm = max(1, int(steps * warmup))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))

    def embed(texts):
        feats = model.preprocess(texts)
        feats = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in feats.items()}
        return F.normalize(model(feats)["sentence_embedding"], dim=-1)

    losses, t0, step = [], time.time(), 0
    for ep in range(epochs):
        order = list(range(len(data)))
        random.Random(seed + ep).shuffle(order)
        for b in range(0, len(order), batch_size):
            batch = [data[i] for i in order[b:b + batch_size]]
            anchors = [a for a, _, _ in batch]
            cands = [p for _, p, _ in batch] + [n for _, _, n in batch if n]
            a = embed(anchors)
            c = embed(cands)
            logits = scale * a @ c.T                       # (B, B + n_hard)
            loss = F.cross_entropy(logits, torch.arange(len(batch), device=device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            losses.append(float(loss))
            if step % 10 == 0 or step == steps:
                el = time.time() - t0
                log(f"  step {step}/{steps}  loss {np.mean(losses[-10:]):.4f}  "
                    f"({el / step:.1f} s/step, eta {(steps - step) * el / step / 60:.1f} min)")
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir))
    data_hash = hashlib.sha256(Path(triplets_path).read_bytes()).hexdigest()[:16]
    card = {"base_model": base_model, "base_revision": base_revision, "triplets_file": str(triplets_path),
            "triplets_sha256_16": data_hash, "pairs_used": len(data),
            "with_hard_negative": sum(n is not None for _, _, n in data), "epochs": epochs, "batch_size": batch_size,
            "lr": lr, "warmup": warmup, "max_seq_length_train": max_seq_length, "scale": scale, "seed": seed,
            "device": device, "steps": steps, "loss_first10": float(np.mean(losses[:10])),
            "loss_last10": float(np.mean(losses[-10:])), "train_seconds": round(time.time() - t0, 1),
            "loss": "MultipleNegativesRanking (in-batch + 1 mined hard negative)"}
    (out_dir / "training_card.json").write_text(json.dumps(card, indent=2))
    return card
