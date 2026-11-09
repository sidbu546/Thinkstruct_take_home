"""Serving latency on an idle machine with cold caches: phase 1 only vs the final two-phase pipeline
(phase 1 top 100 -> bge cross-encoder on the Apple GPU when available -> top 30).
Usage: .venv/bin/python scripts/latency_bench.py [--index index] [--n 20]
Writes reports/latency_bench_<index>.json"""
import argparse
import json
import statistics as st
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from patent_search.config import ROOT, get_config  # noqa: E402
from patent_search.schemas import SearchRequest  # noqa: E402
from patent_search.search.engine import Engine  # noqa: E402
from patent_search.search.rerank import ScoreCache  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="index")
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    eng = Engine(replace(get_config(), index_dir=ROOT / a.index))
    _ = eng.encoder
    _ = eng.reranker("bge")
    qs = [json.loads(line) for line in open(ROOT / "eval_data" / "queries_dev.jsonl")][:a.n]
    out = {}
    for label, kw in (("phase1", {"rerank": False}),
                      ("phase1_top100_then_bge_rerank_top30", {"rerank": True, "rerank_model": "bge", "top_k": 30})):
        eng.rerank_cache = ScoreCache()                    # cold, memory-only cache: what a new query costs
        tot, rr = [], []
        for q in qs:
            r = eng.search(SearchRequest(query=q["text"], mode="claim", **kw))
            tot.append(r.took_ms)
            rr.append(r.stage_ms.get("rerank", 0))
        out[label] = {"device": eng.reranker("bge").device if kw.get("rerank") else "cpu", "n": len(qs), "total_p50_ms": st.median(tot), "total_p95_ms": sorted(tot)[int(len(tot) * .95) - 1],
                      "rerank_p50_ms": st.median(rr)}
        print(label, out[label], flush=True)
    (ROOT / "reports" / f"latency_bench_{a.index}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
