#!/bin/bash
# Reproduce every Part 3 number, in order. Local only: CPU, plus the Apple GPU (mps) for the reranker when available. Times measured on an Apple M4 (CPU);
# steps that share the CPU run slower. Requires the Part 1 index (build-index) first.
set -e
cd "$(dirname "$0")/.."
P=".venv/bin/python -m patent_search"
$P eval build-sets                                                   # ~10 min: splits, queries, pairs, triplets
$P --index index_fixed build-index --chunking fixed                  # ~30 min: baseline index (fixed windows)
for e in E0 E1 E1h E2 E1d E2d E3; do $P eval run --exp $e --split dev; done
# dev results with Part 1's original claim-mode weights (the figures quoted before tuning)
for e in E3 E1 E2 E1d E2d; do
  $P eval run --exp $e --split dev --tag part1w --kind-weights "abstract=0.8,claim=1.0,element=1.0,description=0.6"
done
scripts/tune_kind_weights.sh                                         # ~15 min: claim-mode weights on dev
$P train                                                             # ~25-45 min: fine-tune the bi-encoder
$P --index index_ft build-index --model models/gte-patent-ft         # ~45 min: re-embed with it
.venv/bin/python scripts/calibrate_coverage.py --index index
.venv/bin/python scripts/calibrate_coverage.py --index index_ft
$P eval run --exp E4 --split dev                                     # selection rule: keep fine-tuning if dev MRR does not drop
$P eval pairs --pairs-index index --split dev
$P eval pairs --pairs-index index_ft --split dev
.venv/bin/python scripts/tune_rerank_final.py                        # ~45 min: bge alpha/beta grid on dev; best is set in config.py
scripts/run_test_phase1.sh                                           # ~25 min: all phase-1 test runs, T4, T5
scripts/run_test_rerank.sh                                           # ~1-2.5 h (Mac GPU): E6 and E5 on test
.venv/bin/python scripts/latency_bench.py --index index_ft
.venv/bin/python scripts/readme_tables.py                            # final tables for the README
$P eval report --split test                                          # -> reports/eval_report_test.md
