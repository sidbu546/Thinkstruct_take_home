"""reports/metrics.json -> reports/eval_report.md (the tables that go into the README)."""
from __future__ import annotations

import json

from .runner import EXPERIMENTS, METRICS_PATH

COLS = ("R@1", "R@10", "R@100", "MRR@10", "nDCG@10")


def _cell(m: dict, k: str) -> str:
    if k not in m:
        return "–"
    v = m[k]
    return f"{v['mean']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]"


def _row(label: str, agg: dict) -> str:
    return f"| {label} | {agg.get('n', 0)} | " + " | ".join(_cell(agg, k) for k in COLS) + " |"


def build(split: str = "test") -> str:
    m = json.loads(METRICS_PATH.read_text())
    out = [f"# Evaluation report ({split} split)", "",
           "Silver labels: the relevant patent for a claim query is the claim's own patent, with that patent's own "
           "claim and element chunks hidden, so it must be found through its abstract and description. "
           "95% bootstrap CIs (1,000 resamples over queries).", ""]
    hdr = "| Experiment | n | " + " | ".join(COLS) + " |\n|---|---|" + "---|" * len(COLS)
    for task in ("T1", "T2"):
        out += [f"## {task}: {'independent claim' if task == 'T1' else '2–3 elements of a claim'} → its patent", "",
                hdr]
        extra = {"E3_part1w": "E3 with Part 1's original hand-set claim-mode weights"}
        for exp, label in [(k, v["label"]) for k, v in EXPERIMENTS.items()] + list(extra.items()):
            name = f"{exp}_{split}"
            if name in m and task in m[name]["by_task"]:
                out.append(_row(f"{exp} {label}", m[name]["by_task"][task]))
        out.append("")
    for exp in ("E3", "E6", "E5", "E4"):
        name = f"{exp}_{split}"
        if name in m:
            out += [f"## Breakdown for {exp} ({EXPERIMENTS[exp]['label']})", "", hdr.replace("Experiment", "Group")]
            for key in ("by_tier", "by_data_status"):
                for g, agg in m[name][key].items():
                    note = " (provisional: data gaps)" if key == "by_data_status" and g != "complete" else ""
                    out.append(_row(f"{key[3:]}={g}{note}", agg))
            out.append("")
    pairs = {k: v for k, v in m.items() if k.startswith("T4_") and k.endswith(split)}
    if pairs:
        out += ["## T4: positives vs hard negatives (bi-encoder only)", "",
                "| Index / model | anchors | triplet accuracy | ROC-AUC | mean cos pos | mean cos neg | margin |",
                "|---|---|---|---|---|---|---|"]
        for k, v in pairs.items():
            out.append(f"| {v['index']} ({v['model'].split('/')[-1]}) | {v['anchors']} | {v['triplet_accuracy']:.3f} | "
                       f"{v['roc_auc']:.3f} | {v['mean_cos_pos']:.3f} | {v['mean_cos_neg']:.3f} | {v['margin']:.3f} |")
        out.append("")
    comps = {k: v for k, v in m.items() if k.startswith("cmp_") and k.endswith(split)}
    if comps:
        out += ["## Paired differences (same queries)", "", "| Comparison | metric | Δ (b − a) | 95% CI |", "|---|---|---|---|"]
        for k, v in comps.items():
            out.append(f"| {v['b']} vs {v['a']} | {v['metric']} | {v['delta']:+.3f} | "
                       f"[{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] |")
        out.append("")
    lat_rows = [(k, v["latency_ms"]) for k, v in m.items() if k.endswith(split) and "latency_ms" in v]
    if lat_rows:
        out += ["## Latency per query (CPU, ms)", "",
                "Measured during the evaluation runs, often with other jobs sharing the CPU. Idle, cold-cache "
                "serving numbers are in reports/latency_bench_index.json.", "",
                "| Run | total p50 | total p95 | phase1 p50 | rerank p50 |",
                "|---|---|---|---|---|"]
        for k, lat in lat_rows:
            out.append(f"| {k} | {lat['total']['p50']:.0f} | {lat['total']['p95']:.0f} | "
                       f"{lat.get('phase1', {}).get('p50', 0):.0f} | {lat.get('rerank', {}).get('p50', 0):.0f} |")
        out.append("")
    if "T5_filters" in m:
        t = m["T5_filters"]
        out += ["## T5: filter correctness", "",
                f"{t['results_checked']} results checked over {t['filter_cases']} filter cases × {t['queries']} queries, "
                f"with and without reranking: **{t['violations']} violations**.", ""]
    return "\n".join(out)


def write(split: str = "test") -> str:
    md = build(split)
    (METRICS_PATH.parent / f"eval_report_{split}.md").write_text(md)
    return md
