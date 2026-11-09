#!/bin/bash
# Test-split runs for everything that does not depend on phase-2 settings. Choices were all made on dev.
set -e
cd "$(dirname "$0")/.."
P=".venv/bin/python -m patent_search"
for e in E0 E1 E1h E2 E1d E2d E3 E4; do $P eval run --exp $e --split test; done
# Part 1's original (pre-tuning) claim-mode weights, for the record
$P eval run --exp E3 --split test --tag part1w --kind-weights "abstract=0.8,claim=1.0,element=1.0,description=0.6"
$P eval pairs --pairs-index index --split test
$P eval pairs --pairs-index index_ft --split test
$P eval filters --split dev
