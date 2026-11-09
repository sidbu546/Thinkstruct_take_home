#!/bin/bash
# Test-split runs with phase 2 (top 100 -> bge -> top 30, Apple GPU when available). E6 saves
# per-candidate scores so the dev-tuned alpha/beta can be applied exactly afterwards (eval rescore).
set -e
cd "$(dirname "$0")/.."
P=".venv/bin/python -u -m patent_search"
$P eval run --exp E6 --split test --rerank-model bge --depth 100 --tag raw
$P eval run --exp E5 --split test --rerank-model bge --depth 100
