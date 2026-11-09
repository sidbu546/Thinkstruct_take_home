"""Calibrate element-coverage thresholds from the built index (no labels needed).

Positives: each element of an independent claim vs. its OWN patent's abstract + description
           passages (its own claim/element chunks are excluded, so the match is not trivial).
Negatives: the same element vs. patents in a DIFFERENT CPC subclass (B60B wheel vs B60C tire, etc.),
           which should not disclose it.
Hard negatives: other patents in the SAME subclass. Some genuinely disclose the element, so their
rate above a threshold is an upper bound on over-calling.

found = 97.5th percentile of same-subclass similarity  (strong evidence: ~2.5% of same-subclass
        patents reach it, ~0% of unrelated ones, while ~96% of own-patent evidence does)
weak  = 99th percentile of other-subclass similarity   (related, worth checking: ~1% of unrelated)

Usage:  .venv/bin/python scripts/calibrate_coverage.py [--index index] [--claims 300] [--seed 13]
Writes: reports/coverage_calibration[_<index>].json and <index>/coverage_calibration.json (read by the
        engine, so each index uses thresholds calibrated for its own model)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from patent_search.config import ROOT, get_config  # noqa: E402
from patent_search.search.engine import Engine  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", type=int, default=300)
    ap.add_argument("--negatives", type=int, default=20, help="other-subclass patents per element")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--index", default="index", help="index directory to calibrate")
    a = ap.parse_args()
    from dataclasses import replace
    eng = Engine(replace(get_config(), index_dir=ROOT / a.index))
    rng = random.Random(a.seed)
    element_rows = [r for r in range(eng.n_chunks) if eng.kind[r] == "element"]
    claim_keys = sorted({(eng.doc_of[r], eng.claim_number[r]) for r in element_rows})
    rng.shuffle(claim_keys)
    claim_keys = claim_keys[:a.claims]
    by_sub: dict[str, list[str]] = {}
    for d, p in eng.patents.items():
        by_sub.setdefault(p["cpc_subclass"] or "(none)", []).append(d)
    pos, neg, same = [], [], []
    for doc, cn in claim_keys:
        els = [r for r in eng.doc_rows[doc] if eng.kind[r] == "element" and eng.claim_number[r] == cn]
        own = [r for r in eng.doc_rows[doc] if eng.kind[r] in ("abstract", "description")]
        sub = eng.patents[doc]["cpc_subclass"]
        others = sorted(d for s, ds in by_sub.items() if s != sub for d in ds)
        peers = sorted(d for d in by_sub.get(sub, []) if d != doc)
        if not own or not others:
            continue
        E = eng.vectors[els]
        pos.extend((eng.vectors[own] @ E.T).max(axis=0).tolist())
        for d in rng.sample(others, min(a.negatives, len(others))):
            neg.extend((eng.vectors[eng.doc_rows[d]] @ E.T).max(axis=0).tolist())
        for d in rng.sample(peers, min(a.negatives, len(peers))):
            same.extend((eng.vectors[eng.doc_rows[d]] @ E.T).max(axis=0).tolist())
    pos, neg, same = np.array(pos), np.array(neg), np.array(same)
    found, weak = float(np.percentile(same, 97.5)), float(np.percentile(neg, 99))
    out = {
        "model": eng.manifest["embedding"]["model"], "revision": eng.manifest["embedding"]["revision"],
        "claims_sampled": len(claim_keys), "seed": a.seed,
        "positives": {"n": int(len(pos)), "p10": float(np.percentile(pos, 10)), "median": float(np.median(pos)),
                      "p90": float(np.percentile(pos, 90))},
        "negatives_other_subclass": {"n": int(len(neg)), "median": float(np.median(neg)),
                                     "p99": weak, "share_at_or_above_found": float((neg >= found).mean()),
                                     "share_at_or_above_weak": float((neg >= weak).mean())},
        "same_subclass_other_patents (some genuinely overlap)": {
            "n": int(len(same)), "median": float(np.median(same)), "p97_5": found,
            "share_at_or_above_found": float((same >= found).mean()),
            "share_at_or_above_weak": float((same >= weak).mean())},
        "recommended": {"coverage_found": round(found, 3), "coverage_weak": round(weak, 3)},
        "share_of_positives_found": float((pos >= found).mean()),
        "share_of_positives_at_least_weak": float((pos >= weak).mean()),
    }
    out["index"] = a.index
    (ROOT / "reports").mkdir(exist_ok=True)
    suffix = "" if a.index == "index" else f"_{a.index}"
    (ROOT / "reports" / f"coverage_calibration{suffix}.json").write_text(json.dumps(out, indent=2))
    (ROOT / a.index / "coverage_calibration.json").write_text(json.dumps(out, indent=2))   # read by Engine
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
