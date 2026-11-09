#!/bin/bash
# Test-split runs with phase 2: phase-1 top 100 -> bge-reranker-base (Apple GPU when available) -> top 30.
# The blend weights (alpha, beta) come from patent_search/config.py, where the dev-tuned values are set.
set -e
cd "$(dirname "$0")/.."
P=".venv/bin/python -u -m patent_search"
$P eval run --exp E6 --split test --rerank-model bge --depth 100
$P eval run --exp E5 --split test --rerank-model bge --depth 100
