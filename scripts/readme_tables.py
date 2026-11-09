"""Print the final Part 3 result tables (test split) as Markdown, straight from reports/metrics.json,
so README numbers are never copied by hand. Usage: .venv/bin/python scripts/readme_tables.py"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
m = json.loads((ROOT / "reports" / "metrics.json").read_text())

BASE = "`gte-modernbert-base`"
FT = "`gte-modernbert-base`<br>fine-tuned (`gte-patent-ft`)"
HYB = "hybrid<br>dense + BM25, fused by RRF"
RR = "`bge-reranker-base`<br>top 100 → 30"
MATRIX = [  # (run, id, bi-encoder, chunking, retrieval, rerank)
    ("E0_test", "E0", "—", "adaptive", "BM25 only", "—"),
    ("E1_test", "E1", BASE, "fixed 200-word windows", "dense only", "—"),
    ("E2_test", "E2", BASE, "adaptive", "dense only", "—"),
    ("E3_test", "E3", BASE, "adaptive", HYB, "—"),
    ("E4_test", "E4", FT, "adaptive", HYB, "—"),
    ("E5_test", "E5", BASE, "adaptive", HYB, RR),
    ("E6_test", "**E6**", FT, "adaptive", HYB, RR),
]
EXTRA = [
    ("E3_part1w_test", "E3 · Part 1 weights", BASE, "adaptive", HYB + "<br>original hand-set claim-mode weights", "—"),
    ("E1h_test", "E1h", BASE, "fixed 200-word windows", HYB, "—"),
    ("E1d_test", "E1d", BASE, "fixed 200-word windows<br>abstract + description only", "dense only", "—"),
    ("E2d_test", "E2d", BASE, "adaptive<br>abstract + description only", "dense only", "—"),
]
COMPARE = [
    ("E3_part1w_test", "E3_test", "dev-tuned chunk weights vs Part 1 weights"),
    ("E1_test", "E2_test", "adaptive vs fixed chunks (dense, E2 vs E1)"),
    ("E1d_test", "E2d_test", "adaptive vs fixed, abstract + description only"),
    ("E0_test", "E3_test", "hybrid vs BM25 only (E3 vs E0)"),
    ("E3_test", "E4_test", "fine-tuned vs base bi-encoder (E4 vs E3)"),
    ("E3_test", "E5_test", "rerank on the base model (E5 vs E3)"),
    ("E4_test", "E6_test", "rerank on the fine-tuned model (E6 vs E4)"),
    ("E3_test", "E6_test", "full Part 3 pipeline vs tuned Part 1 (E6 vs E3)"),
]


def f(x):
    return f"{x:.3f}"


def row(run, *cols):
    if run not in m:
        return "| " + " | ".join(cols) + " | (not run) | | | |"
    o = m[run]["overall"]
    mrr = o["MRR@10"]
    return ("| " + " | ".join(cols) + f" | {f(mrr['mean'])} [{f(mrr['ci95'][0])}, {f(mrr['ci95'][1])}] | "
            f"{f(o['R@1']['mean'])} | {f(o['R@10']['mean'])} | {f(o['R@100']['mean'])} |")


hdr = ("| Exp | Bi-encoder | Chunking | Retrieval | Rerank | MRR@10 [95% CI] | R@1 | R@10 | R@100 |\n"
       "|---|---|---|---|---|---|---|---|---|")
print("**Experiment matrix (test, 306 queries)**\n")
print(hdr)
for r in MATRIX:
    print(row(*r))
print("\n**Ablations and reference rows (test)**\n")
print(hdr)
for r in EXTRA:
    print(row(*r))
print("\n**Paired differences on the same test queries** (ΔMRR@10, paired bootstrap 95% CI)\n")
print("| Change | Δ | 95% CI |\n|---|---|---|")
sys.path.insert(0, str(ROOT))
from patent_search.eval import runner  # noqa: E402
for a, b, label in COMPARE:
    if a in m and b in m:
        key = f"cmp_{a}_vs_{b}_MRR@10_test"
        if (runner.RUNS_DIR / f"{a}.jsonl").exists() and (runner.RUNS_DIR / f"{b}.jsonl").exists():
            d = runner.compare(a, b)                 # recompute from the per-query run files
            runner.save_metrics(key, d)
        else:
            d = m[key]                               # run files absent: use the stored comparison
        print(f"| {label} | {d['delta']:+.3f} | [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] |")
