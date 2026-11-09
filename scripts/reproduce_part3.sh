#!/bin/bash
# Reproduce every Part 3 number, in order. Local only: CPU, plus the Apple GPU (mps) for the reranker when available. Times measured on an Apple M4 (CPU);
# steps that share the CPU run slower. Requires the Part 1 index (build-index) first.
set -e
cd "$(dirname "$0")/.."
P=".venv/bin/python -m patent_search"
$P eval build-sets                                                   # ~10 min: splits, queries, pairs, triplets
$P --index index_fixed build-index --chunking fixed                  # ~30 min: baseline index (fixed windows)
for e in E0 E1 E1h E2 E1d E2d E3; do $P eval run --exp $e --split dev; done
scripts/tune_kind_weights.sh                                         # ~15 min: claim-mode weights on dev
$P train                                                             # ~45 min: fine-tune the bi-encoder
$P --index index_ft build-index --model models/gte-patent-ft         # ~45 min: re-embed with it
.venv/bin/python scripts/calibrate_coverage.py --index index
.venv/bin/python scripts/calibrate_coverage.py --index index_ft
$P eval run --exp E4 --split dev
$P eval pairs --pairs-index index --split dev
$P eval pairs --pairs-index index_ft --split dev
.venv/bin/python scripts/tune_rerank_final.py                        # ~45 min: bge alpha/beta, top 100 -> 30, Mac GPU (dev)
scripts/run_test_phase1.sh                                           # ~25 min: all phase-1 test runs, T4, T5
scripts/run_test_rerank.sh                                           # ~2.5 h (Mac GPU): E6 and E5 on test
$P eval rescore --a E6_raw_test --b E6_test --alpha 0.3 --beta 0.1   # final E6 run name (same alpha/beta as config)
.venv/bin/python scripts/latency_bench.py --index index_ft
.venv/bin/python scripts/readme_tables.py                            # final tables for the README
$P eval report --split test                                          # -> reports/eval_report_test.md
