"""Evaluation and training sets from document structure (DESIGN.md Part 3A.1–3A.2).

No human labels exist, so labels are silver:
  T1  claim -> patent         query = an independent claim; relevant = its own patent, whose own
                              claim/element chunks are hidden (it must be found via abstract/description)
  T2  partial claim -> patent query = 2–3 elements of an independent claim; same relevance and masking
  T4  pair discrimination     (claim, own abstract) vs (claim, hard-negative abstracts)
  train                       triplets (anchor, positive, hard negative) from the train split only

Splits are by patent FAMILY (same normalized title filed < 8 weeks apart, or near-identical
abstracts), so near-duplicates never straddle train and test. Patents with data issues stay in;
every query carries the source patent's data_status so metrics can be split by it.
"""
from __future__ import annotations

import json
import random
import re
from datetime import date
from pathlib import Path

import numpy as np

from ..schemas import SearchRequest
from ..search.engine import Engine

SPLIT_SHARES = (("train", 0.6), ("dev", 0.2), ("test", 0.2))
QUERY_CAPS = {"dev": {"T1": 150, "T2": 100}, "test": {"T1": 300, "T2": 200}}
_W = re.compile(r"[a-z0-9]+")


def _shingles(text: str, n: int = 5) -> set:
    w = _W.findall((text or "").lower())
    return {" ".join(w[i:i + n]) for i in range(max(len(w) - n + 1, 1))}


def families(eng: Engine) -> dict[str, int]:
    docs = sorted(eng.patents)
    parent = {d: d for d in docs}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    by_title: dict[str, list[str]] = {}
    for d in docs:
        by_title.setdefault(eng.patents[d]["title_norm"], []).append(d)
    for group in by_title.values():
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                wa, wb = eng.patents[a]["filing_week"], eng.patents[b]["filing_week"]
                if wa and wb and abs((date.fromisoformat(wa) - date.fromisoformat(wb)).days) < 56:
                    union(a, b)
    sh = {d: _shingles(eng.patents[d]["abstract"]) for d in docs}
    for i, a in enumerate(docs):
        for b in docs[i + 1:]:
            sa, sb = sh[a], sh[b]
            if sa and sb and len(sa & sb) / len(sa | sb) >= 0.8:
                union(a, b)
    roots = sorted({find(d) for d in docs})
    rid = {r: i for i, r in enumerate(roots)}
    return {d: rid[find(d)] for d in docs}


def assign_splits(fam: dict[str, int], seed: int) -> dict[str, str]:
    fams: dict[int, list[str]] = {}
    for d, f in fam.items():
        fams.setdefault(f, []).append(d)
    order = sorted(fams)
    random.Random(seed).shuffle(order)
    n = len(fam)
    out, filled, si = {}, 0, 0
    bounds = np.cumsum([s for _, s in SPLIT_SHARES]) * n
    for f in order:
        while si < len(SPLIT_SHARES) - 1 and filled >= bounds[si]:
            si += 1
        for d in fams[f]:
            out[d] = SPLIT_SHARES[si][0]
        filled += len(fams[f])
    return out


def _independent_claims(eng: Engine, doc: str):
    return eng.con.execute(
        "SELECT claim_number, text, tier, elements FROM claims WHERE doc_number=? AND status='active' "
        "AND depends_on='[]' AND unresolved_refs='[]' ORDER BY claim_number", (doc,)).fetchall()


def build_queries(eng: Engine, split_of: dict[str, str], seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    out: dict[str, list[dict]] = {"dev": [], "test": []}
    for split in ("dev", "test"):
        docs = sorted(d for d, s in split_of.items() if s == split)
        t1, t2 = [], []
        for d in docs:
            claims = [c for c in _independent_claims(eng, d) if len(c["text"].split()) >= 15]
            for c in claims[:2]:                                   # at most 2 per patent: spread over patents
                t1.append((d, c))
            multi = [c for c in claims if len(json.loads(c["elements"])) >= 4]
            if multi:
                t2.append((d, multi[0]))
        rng.shuffle(t1)
        rng.shuffle(t2)
        for task, items in (("T1", t1), ("T2", t2)):
            for i, (d, c) in enumerate(items[:QUERY_CAPS[split][task]]):
                if task == "T1":
                    text = c["text"]
                else:
                    els = json.loads(c["elements"])
                    k = rng.choice([2, 3])
                    pick = sorted(rng.sample(range(len(els)), k))
                    text = "; ".join(els[j] for j in pick)
                out[split].append({"qid": f"{split}-{task}-{i:04d}", "task": task, "text": text, "source_doc": d,
                                   "claim_number": c["claim_number"], "tier": c["tier"],
                                   "data_status": eng.patents[d]["data_status"]})
    return out


def _abstract_row(eng: Engine, doc: str) -> int | None:
    for r in eng.doc_rows.get(doc, []):
        if eng.kind[r] == "abstract":
            return int(r)
    return None


def mine_hard_negatives(eng: Engine, anchor: str, doc: str, fam: dict[str, int], pos_vec: np.ndarray,
                        k: int = 3, allowed: set[str] | None = None) -> list[str]:
    """Other patents, same CPC subclass, ranked 5–50 by the current engine, not the same family,
    and not near-identical to the positive (likely false negatives). `allowed` restricts negatives to
    a set of patents: for training triplets it is the training split, so no dev/test patent's text is
    ever seen during training."""
    sub = eng.patents[doc]["cpc_subclass"]
    r = eng.search(SearchRequest(query=anchor, mode="claim", top_k=50, facets=False, rerank=False))
    out = []
    for x in r.results[4:]:
        d = x.doc_number
        if d == doc or fam[d] == fam[doc] or eng.patents[d]["cpc_subclass"] != sub:
            continue
        if allowed is not None and d not in allowed:
            continue
        row = _abstract_row(eng, d)
        if row is None or float(eng.vectors[row] @ pos_vec) > 0.95:
            continue
        out.append(d)
        if len(out) == k:
            break
    return out


def build_pairs(eng: Engine, split_of: dict[str, str], fam: dict[str, int], split: str, seed: int,
                max_anchors: int, log=print) -> list[dict]:
    """(anchor, positive, negatives) records. Positives: own abstract (claim->abstract), and for
    training also the own description passage closest to the anchor (claim->passage, element->passage)."""
    rng = random.Random(seed)
    docs = sorted(d for d, s in split_of.items() if s == split)
    anchors = []
    for d in docs:
        for c in _independent_claims(eng, d):
            if len(c["text"].split()) >= 15:
                anchors.append((d, c))
    rng.shuffle(anchors)
    anchors = anchors[:max_anchors]
    train_only = {d for d, s in split_of.items() if s == "train"} if split == "train" else None
    recs = []
    for i, (d, c) in enumerate(anchors):
        arow = _abstract_row(eng, d)
        if arow is None:
            continue
        pos_abs = eng.display_text[arow]
        negs = mine_hard_negatives(eng, c["text"], d, fam, eng.vectors[arow], allowed=train_only)
        neg_abs = [eng.display_text[_abstract_row(eng, n)] for n in negs]
        recs.append({"type": "claim->abstract", "doc": d, "anchor": c["text"], "positive": pos_abs,
                     "negatives": neg_abs, "negative_docs": negs, "data_status": eng.patents[d]["data_status"]})
        if split == "train":
            desc = [r for r in eng.doc_rows[d] if eng.kind[r] == "description" and eng.weight[r] >= 1.0]
            if desc:
                qv = eng.encoder.encode([c["text"]])[0]
                best = desc[int(np.argmax(eng.vectors[desc] @ qv))]
                recs.append({"type": "claim->passage", "doc": d, "anchor": c["text"],
                             "positive": eng.display_text[best], "negatives": neg_abs[:1], "negative_docs": negs[:1],
                             "data_status": eng.patents[d]["data_status"]})
                els = json.loads(c["elements"])
                if len(els) >= 3:
                    el = rng.choice(els)
                    ev = eng.encoder.encode([el])[0]
                    best_e = desc[int(np.argmax(eng.vectors[desc] @ ev))]
                    recs.append({"type": "element->passage", "doc": d, "anchor": el,
                                 "positive": eng.display_text[best_e], "negatives": [], "negative_docs": [],
                                 "data_status": eng.patents[d]["data_status"]})
        if (i + 1) % 100 == 0:
            log(f"  {split}: {i + 1}/{len(anchors)} anchors")
    return recs


def build_all(eng: Engine, out_dir: Path, seed: int = 13, max_train_anchors: int = 1200, log=print) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    fam = families(eng)
    split_of = assign_splits(fam, seed)
    queries = build_queries(eng, split_of, seed)
    (out_dir / "splits.json").write_text(json.dumps(
        {"seed": seed, "split_of": split_of, "family_of": fam,
         "counts": {s: sum(v == s for v in split_of.values()) for s, _ in SPLIT_SHARES},
         "n_families": len(set(fam.values()))}, indent=1))
    for split, qs in queries.items():
        with open(out_dir / f"queries_{split}.jsonl", "w") as f:
            for q in qs:
                f.write(json.dumps(q) + "\n")
        with open(out_dir / f"qrels_{split}.tsv", "w") as f:
            for q in qs:
                f.write(f"{q['qid']}\t0\t{q['source_doc']}\t1\n")
    stats = {"queries": {s: {t: sum(q["task"] == t for q in qs) for t in ("T1", "T2")} for s, qs in queries.items()}}
    for split, cap in (("dev", 150), ("test", 300), ("train", max_train_anchors)):
        log(f"mining pairs for {split} (up to {cap} anchors) ...")
        recs = build_pairs(eng, split_of, fam, split, seed, cap, log)
        name = "triplets_train.jsonl" if split == "train" else f"pairs_{split}.jsonl"
        with open(out_dir / name, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        stats[name] = {"records": len(recs), "with_hard_negative": sum(bool(r["negatives"]) for r in recs)}
    stats["splits"] = {s: sum(v == s for v in split_of.values()) for s, _ in SPLIT_SHARES}
    stats["families"] = len(set(fam.values()))
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats
