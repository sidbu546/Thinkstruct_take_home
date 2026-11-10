"""Reproduce the Part 1 results in README §4.

  .venv/bin/python scripts/part1_checks.py                 # Part 1's original weights (as reported in §4)
  .venv/bin/python scripts/part1_checks.py --weights final # the dev-tuned weights now in config.py

Prints:
  1. what ingest found (§4.1, §4.2), read from index/manifest.json written by build-index
  2. claim self-retrieval (§4.4): 100 random independent claims (seed 1) searched in claim mode;
     share whose own patent ranks 1st and in the top 5. The claim's own chunks are NOT hidden here,
     so this is a sanity check, not the Part 3 evaluation
  3. latency per request type (§4.4), warm, model loaded. Machine-dependent: expect similar, not equal
"""
import argparse
import json
import random
import statistics as st
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from patent_search.config import get_config  # noqa: E402
from patent_search.schemas import SearchRequest  # noqa: E402
from patent_search.search.engine import Engine  # noqa: E402

PART1_WEIGHTS = {"abstract": 0.8, "claim": 1.0, "element": 1.0, "description": 0.6}
TEXT_QUERIES = ["tire with sound absorbing foam in the cavity", "carbon fiber wheel spoke", "trailer hitch coupling",
                "tire tread groove pattern for wet braking", "wheel hub bearing seal",
                "non-pneumatic tire with flexible spokes", "tire pressure sensor mounted on rim", "run flat insert"]


def p95(v):
    v = sorted(v)
    return v[max(0, int(len(v) * 0.95) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", choices=["part1", "final"], default="part1")
    a = ap.parse_args()
    cfg = get_config()
    if a.weights == "part1":
        cfg = replace(cfg, search=replace(cfg.search, kind_weights_claim=PART1_WEIGHTS))
    eng = Engine(cfg)
    _ = eng.encoder

    r = eng.manifest["report"]
    print("1) Ingest (from index/manifest.json)")
    print(f"   patents {r['patents_loaded']}, quarantined {r['records_quarantined']}, duplicates "
          f"{r['duplicates_exact_skipped']} exact / {r['duplicates_conflicting']} conflicting")
    print(f"   claims active {r['claims_active']}, canceled {r['claims_canceled_kept_as_status_rows']}; "
          f"empty description paragraphs {r['empty_paragraphs_structural']}")
    print(f"   chunks {sum(r['chunks_by_kind'].values())} {r['chunks_by_kind']}; claim tiers {r['claim_tiers']}")
    print(f"   data issues {r['data_issues_by_type']}")
    print(f"   data status {r['patents_by_data_status']}; build {eng.manifest['timings']}")

    rows = eng.con.execute("SELECT doc_number, text FROM claims WHERE status='active' AND depends_on='[]' "
                           "AND unresolved_refs='[]' ORDER BY doc_number").fetchall()
    random.seed(1)
    sample = random.sample(rows, 100)
    lat = {"text": [], "text + filters": [], "claim": [], "claim + filters": [], "patent": [], "filters only": []}
    top1 = top5 = 0
    for doc, text in sample:
        res = eng.search(SearchRequest(query=text, mode="claim", top_k=10))
        lat["claim"].append(res.took_ms)
        docs = [x.doc_number for x in res.results]
        top1 += docs[0] == doc
        top5 += doc in docs[:5]
        sub = eng.patents[doc]["cpc_subclass"]
        lat["claim + filters"].append(eng.search(SearchRequest(query=text, mode="claim", top_k=10,
                                                               filters={"cpc_prefix": [sub]})).took_ms)
    for q in TEXT_QUERIES * 3:
        lat["text"].append(eng.search(SearchRequest(query=q)).took_ms)
        lat["text + filters"].append(eng.search(SearchRequest(
            query=q, filters={"cpc_prefix": ["B60B"], "keywords": {"terms": ["wheel"], "field": "both"}})).took_ms)
        lat["filters only"].append(eng.search(SearchRequest(
            filters={"cpc_prefix": ["B60C"], "keywords": {"terms": ["tread"], "field": "abstract"}})).took_ms)
    for doc, _ in sample[:20]:
        lat["patent"].append(eng.search(SearchRequest(mode="patent", doc_number=doc)).took_ms)

    print(f"\n2) Claim self-retrieval ({a.weights} weights), 100 random independent claims: "
          f"top 1 {top1}%, top 5 {top5}%")
    print("\n3) Latency (warm, ms)")
    for k, v in lat.items():
        print(f"   {k:16s} n={len(v):3d}  p50 {st.median(v):6.0f}   p95 {p95(v):6.0f}")
    out = Path(__file__).resolve().parent.parent / "reports" / f"part1_checks_{a.weights}.json"
    out.write_text(json.dumps({"weights": a.weights, "self_retrieval": {"top1": top1, "top5": top5, "n": 100},
                               "latency_ms": {k: {"p50": st.median(v), "p95": p95(v)} for k, v in lat.items()}},
                              indent=2))
    print(f"\nsaved {out.relative_to(out.parent.parent)}")


if __name__ == "__main__":
    main()
