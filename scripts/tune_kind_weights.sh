#!/bin/bash
# Dev-split grid over claim-mode chunk-kind weights for phase 1 (E3). Output: reports/kind_weight_grid_dev.log
cd "$(dirname "$0")/.."
for c in 1.0 0.6 0.3; do for e in 1.0 0.3; do for d in 0.6 1.0; do
  t="kw_c${c}_e${e}_d${d}"
  .venv/bin/python -m patent_search eval run --exp E3 --split dev --tag $t \
      --kind-weights "claim=$c,element=$e,description=$d,abstract=1.0" 2>&1 \
    | grep -v -E 'Redirects|Loading weights|s/query' \
    | .venv/bin/python -c "import json,sys; d=json.load(sys.stdin); o=d['overall']; print('$t', 'MRR@10', o['MRR@10'], 'R@10', o['R@10'], 'R@100', o['R@100'], flush=True)"
done; done; done
