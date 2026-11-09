"""verify-index: check that the index on disk is what the code + data + pinned model produce.

1. data files on disk have the SHA-256s recorded in the manifest
2. re-running ingest + adaptive chunking gives the same chunk digest (texts and ids)
3. vectors.npy has the SHA-256 recorded in the manifest
4. re-encoding a deterministic sample of chunks reproduces the stored vectors
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..config import Config
from ..embed.encoder import get_encoder
from ..manifest import _sha256_file, load_manifest
from .build import chunks_digest, prepare


def verify(data_dir: Path, cfg: Config, sample: int = 200, log=print) -> bool:
    m = load_manifest(cfg)
    from dataclasses import replace
    from ..config import ChunkingConfig
    cfg = replace(cfg, chunking=ChunkingConfig(**m["config"]["chunking"]),
                  model_name=m["embedding"]["model"], model_revision=m["embedding"]["revision"])
    ok = True

    def check(name, passed, detail=""):
        nonlocal ok
        ok &= passed
        log(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    files = {f["file"]: f["sha256"] for f in m["data_files"]}
    disk = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(data_dir).glob("*.json"))}
    check("data files match manifest", files == disk, f"{len(disk)} files")

    _, _, chunks, _ = prepare(data_dir, cfg)
    check("chunking reproduces chunk digest", chunks_digest(chunks) == m["chunks_digest"], f"{len(chunks)} chunks")

    check("vectors.npy sha256 matches manifest", _sha256_file(cfg.vectors_path) == m["vectors"]["sha256"])

    vecs = np.load(cfg.vectors_path, mmap_mode="r")
    step = max(1, len(chunks) // sample)
    idx = list(range(0, len(chunks), step))[:sample]
    enc = get_encoder(cfg)
    fresh = enc.encode([chunks[i].embed_text for i in idx], batch_size=cfg.embed_batch_size)
    diff = float(np.abs(fresh - vecs[idx]).max())
    cos = float(np.min(np.sum(fresh * vecs[idx], axis=1)))
    check("re-encoded sample matches stored vectors", diff < 1e-4,
          f"{len(idx)} chunks, max |diff| = {diff:.2e}, min cosine = {cos:.6f} "
          f"(device now {cfg.device}, built on {m['embedding']['device']})")
    return ok
