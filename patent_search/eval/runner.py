"""Experiment runner (DESIGN.md Part 3A.4). Every experiment runs the SAME Engine.search that users
call, so the evaluation measures the real system, not a copy of it."""
from __future__ import annotations

import json
import statistics as st
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..config import ROOT, Config
from ..schemas import SearchRequest
from ..search.engine import Engine
from .metrics import aggregate, pair_metrics, paired_delta

EVAL_DIR = ROOT / "eval_data"
RUNS_DIR = ROOT / "runs"
METRICS_PATH = ROOT / "reports" / "metrics.json"

EXPERIMENTS = {
    "E0": {"index": "index", "retrieval": "bm25", "rerank": False, "label": "BM25 only (adaptive chunks)"},
    "E1": {"index": "index_fixed", "retrieval": "dense", "rerank": False, "label": "dense, FIXED 200-word chunks"},
    "E1h": {"index": "index_fixed", "retrieval": "hybrid", "rerank": False, "label": "hybrid, FIXED chunks"},
    "E2": {"index": "index", "retrieval": "dense", "rerank": False, "label": "dense, ADAPTIVE chunks"},
    "E1d": {"index": "index_fixed", "retrieval": "dense", "rerank": False, "kinds": ["abstract", "description"],
            "label": "ablation: dense, fixed, abstract+description only"},
    "E2d": {"index": "index", "retrieval": "dense", "rerank": False, "kinds": ["abstract", "description"],
            "label": "ablation: dense, adaptive, abstract+description only"},
    "E3": {"index": "index", "retrieval": "hybrid", "rerank": False, "label": "hybrid, adaptive, dev-tuned chunk weights"},
    "E4": {"index": "index_ft", "retrieval": "hybrid", "rerank": False, "label": "hybrid, FINE-TUNED bi-encoder"},
    "E5": {"index": "index", "retrieval": "hybrid", "rerank": True, "label": "hybrid + cross-encoder rerank"},
    "E6": {"index": "index_ft", "retrieval": "hybrid", "rerank": True, "label": "fine-tuned + rerank"},
}

_ENGINES: dict[str, Engine] = {}


def engine_for(index: str, base: Config) -> Engine:
    if index not in _ENGINES:
        _ENGINES[index] = Engine(replace(base, index_dir=ROOT / index))
    return _ENGINES[index]


def load_queries(split: str) -> list[dict]:
    return [json.loads(line) for line in open(EVAL_DIR / f"queries_{split}.jsonl")]


def _with_search(eng: Engine, **kw):
    old = eng.cfg
    eng.cfg = replace(old, search=replace(old.search, **kw)) if kw else old
    return old


def run(exp: str, split: str, base: Config, rerank_model: str | None = None, alpha: float | None = None,
        beta: float | None = None, depth: int | None = None, tag: str | None = None, limit: int | None = None,
        log=print, kind_weights: dict | None = None) -> dict:
    spec = EXPERIMENTS[exp]
    eng = engine_for(spec["index"], base)
    overrides = {k: v for k, v in (("rerank_alpha", alpha), ("rerank_beta", beta),
                                   ("kind_weights_claim", kind_weights)) if v is not None}
    # No time budget during evaluation: a budget makes results depend on machine load (a busy CPU
    # would rerank fewer candidates). Latency is reported separately; the budget applies in serving.
    overrides["rerank_budget_ms"] = 10 ** 9
    old = _with_search(eng, **overrides)
    queries = load_queries(split)[:limit] if limit else load_queries(split)
    RUNS_DIR.mkdir(exist_ok=True)
    name = f"{exp}{('_' + tag) if tag else ''}_{split}"
    lines, ranks, p1ranks, lat = [], [], [], defaultdict(list)
    partial = RUNS_DIR / f"{name}.partial.jsonl"           # written line by line: a crash keeps the work
    done = {}
    if partial.exists():
        for line in open(partial):
            x = json.loads(line)
            done[x["qid"]] = x
        log(f"  {name}: resuming, {len(done)} queries already done")
    pf = open(partial, "a")
    t0 = time.time()
    try:
        for i, q in enumerate(queries):
            if q["qid"] in done:
                x = done[q["qid"]]
                lines.append(x)
                ranks.append(x["rank"])
                p1ranks.append(x["phase1_rank"])
                lat["total"].append(x["took_ms"])
                for k, v in x["stage_ms"].items():
                    lat[k].append(v)
                continue
            req = SearchRequest(query=q["text"], mode="claim", top_k=100, facets=False, retrieval=spec["retrieval"],
                                rerank=spec["rerank"], rerank_model=rerank_model, rerank_depth=depth,
                                hide_claims_of=[q["source_doc"]], chunk_kinds=spec.get("kinds"))
            r = eng.search(req)
            docs = [x.doc_number for x in r.results]
            rank = docs.index(q["source_doc"]) + 1 if q["source_doc"] in docs else None
            p1 = next((x.signals.phase1_rank for x in r.results if x.doc_number == q["source_doc"]), None)
            ranks.append(rank)
            p1ranks.append(p1)
            lat["total"].append(r.took_ms)
            for k, v in r.stage_ms.items():
                lat[k].append(v)
            cands = None
            if spec["rerank"] and getattr(eng, "last_phase2", None) is not None:
                cands = [{"doc": o["doc"], "p1": float(o["p1"].score),
                          "rerank": None if o["rerank"] is None else float(o["rerank"]),
                          "coverage": None if o["coverage"] is None else float(o["coverage"])}
                         for o in eng.last_phase2[:depth or eng.cfg.search.rerank_depth]]
                eng.last_phase2 = None
            lines.append({"candidates": cands, "qid": q["qid"], "exp": exp, "tag": tag, "task": q["task"], "tier": q["tier"],
                          "data_status": q["data_status"], "source_doc": q["source_doc"], "rank": rank,
                          "phase1_rank": p1, "ranking": docs, "took_ms": r.took_ms, "stage_ms": r.stage_ms,
                          "rerank": r.rerank, "notices": r.pipeline_notices})
            pf.write(json.dumps(lines[-1]) + "\n")
            pf.flush()
            if (i + 1) % 50 == 0:
                log(f"  {name}: {i + 1}/{len(queries)} ({(time.time() - t0) / (i + 1):.2f} s/query)")
    finally:
        eng.cfg = old
        pf.close()
    with open(RUNS_DIR / f"{name}.jsonl", "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    partial.unlink()
    res = summarize(lines)
    res["latency_ms"] = {k: {"p50": float(st.median(v)), "p95": float(sorted(v)[max(0, int(len(v) * .95) - 1)])}
                         for k, v in lat.items()}
    res["config"] = {"exp": exp, **spec, "split": split, "rerank_model": (rerank_model or base.search.rerank_model)
                     if spec["rerank"] else None, "alpha": alpha, "beta": beta, "depth": depth, "tag": tag,
                     "kind_weights_claim": kind_weights or eng.cfg.search.kind_weights_claim,
                     "model": eng.manifest["embedding"]["model"]}
    if spec["rerank"]:
        moved = [(a, b) for a, b in zip(p1ranks, ranks) if a is not None and b is not None]
        res["rank_movement_of_relevant"] = {"n": len(moved), "mean_phase1_rank": float(np.mean([a for a, _ in moved])),
                                            "mean_final_rank": float(np.mean([b for _, b in moved])),
                                            "improved": sum(b < a for a, b in moved),
                                            "worsened": sum(b > a for a, b in moved)}
    save_metrics(name, res)
    return res


def rescore(raw_name: str, out_name: str, alpha: float, beta: float) -> dict:
    """Apply alpha/beta to the per-candidate scores saved by a rerank run, with exactly the formula of
    search/rerank.py (z-scores over the scored candidates, ties broken by doc number). The cross-encoder
    scores themselves do not depend on alpha/beta, so this equals re-running the engine."""
    from ..search.rerank import _z
    lines = load_run(raw_name)
    out = []
    for x in lines:
        c = [k for k in x["candidates"] if k["rerank"] is not None]
        r = np.array([k["rerank"] for k in c])
        p1 = np.array([k["p1"] for k in c])
        final = alpha * _z(r) + (1 - alpha) * _z(p1)
        if beta:
            final = final + beta * np.array([k["coverage"] or 0.0 for k in c])
        order = sorted(range(len(c)), key=lambda i: (-final[i], c[i]["doc"]))
        docs = [c[i]["doc"] for i in order] + [k["doc"] for k in x["candidates"] if k["rerank"] is None]
        y = dict(x)
        y["ranking"] = docs
        y["rank"] = docs.index(x["source_doc"]) + 1 if x["source_doc"] in docs else None
        y["rescored_from"] = {"run": raw_name, "alpha": alpha, "beta": beta}
        out.append(y)
    with open(RUNS_DIR / f"{out_name}.jsonl", "w") as f:
        for line in out:
            f.write(json.dumps(line) + "\n")
    res = summarize(out)
    raw = json.loads(METRICS_PATH.read_text()).get(raw_name, {})
    res["latency_ms"] = raw.get("latency_ms", {})
    res["config"] = {**raw.get("config", {}), "alpha": alpha, "beta": beta, "rescored_from": raw_name}
    moved = [(x["phase1_rank"], x["rank"]) for x in out if x["phase1_rank"] is not None and x["rank"] is not None]
    if moved:
        res["rank_movement_of_relevant"] = {"n": len(moved), "improved": sum(b < a for a, b in moved),
                                            "worsened": sum(b > a for a, b in moved)}
    save_metrics(out_name, res)
    return res


def summarize(lines: list[dict]) -> dict:
    out = {"overall": aggregate([x["rank"] for x in lines])}
    for key in ("task", "tier", "data_status"):
        groups = defaultdict(list)
        for x in lines:
            groups[x[key]].append(x["rank"])
        out[f"by_{key}"] = {g: aggregate(v) for g, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}
    return out


def load_run(name: str) -> list[dict]:
    return [json.loads(line) for line in open(RUNS_DIR / f"{name}.jsonl")]


def compare(a: str, b: str, metric: str = "MRR@10") -> dict:
    ra = {x["qid"]: x["rank"] for x in load_run(a)}
    rb = {x["qid"]: x["rank"] for x in load_run(b)}
    q = sorted(set(ra) & set(rb))
    return {"a": a, "b": b, "n": len(q), **paired_delta([ra[k] for k in q], [rb[k] for k in q], metric)}


def pairs_eval(index: str, split: str, base: Config) -> dict:
    """T4: does the bi-encoder of this index rate positives above hard negatives?"""
    eng = engine_for(index, base)
    recs = [json.loads(line) for line in open(EVAL_DIR / f"pairs_{split}.jsonl")]
    recs = [r for r in recs if r["negatives"]]
    texts = sorted({t for r in recs for t in [r["anchor"], r["positive"], *r["negatives"]]})
    vec = dict(zip(texts, eng.encoder.encode(texts)))
    pos = [float(vec[r["anchor"]] @ vec[r["positive"]]) for r in recs]
    negs = [[float(vec[r["anchor"]] @ vec[n]) for n in r["negatives"]] for r in recs]
    res = {"index": index, "model": eng.manifest["embedding"]["model"], "split": split, "anchors": len(recs),
           **pair_metrics(pos, negs)}
    save_metrics(f"T4_{index}_{split}", res)
    return res


FILTER_CASES = [
    {"cpc_prefix": ["B60B"]}, {"cpc_prefix": ["B60C11"]}, {"cpc_exclude": ["B60C"]},
    {"keywords": {"terms": ["tread"], "field": "abstract"}}, {"title": {"value": "TIRE", "mode": "exact"}},
    {"week_from": "2024-06-01", "week_to": "2024-09-30"}, {"cpc_prefix": ["B60D"], "keywords": {"terms": ["hitch"]}},
]


def filter_correctness(base: Config, split: str = "dev", n_queries: int = 20, rerank: bool = True) -> dict:
    """T5: no result may violate a filter, in phase 1 or after phase 2."""
    eng = engine_for("index", base)
    from ..search.filters import resolve
    from ..schemas import Filters
    queries = load_queries(split)[:n_queries]
    checked = violations = 0
    for f in FILTER_CASES:
        allowed = resolve(eng.con, Filters(**f)).allowed
        for q in queries:
            for rr in ((False, True) if rerank else (False,)):
                r = eng.search(SearchRequest(query=q["text"], mode="claim", top_k=30, facets=False, filters=f,
                                             rerank=rr, rerank_model="bge", rerank_depth=10))
                for x in r.results:
                    checked += 1
                    violations += x.doc_number not in allowed
    res = {"filter_cases": len(FILTER_CASES), "queries": len(queries), "results_checked": checked,
           "violations": violations}
    save_metrics("T5_filters", res)
    return res


def save_metrics(name: str, res: dict):
    METRICS_PATH.parent.mkdir(exist_ok=True)
    allm = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    allm[name] = res
    METRICS_PATH.write_text(json.dumps(allm, indent=1))
