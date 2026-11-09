"""Retrieval and pair metrics, with bootstrap confidence intervals. Single relevant document per
query (silver labels), so nDCG@10 = 1/log2(rank+1) when the relevant patent is in the top 10."""
from __future__ import annotations

import math

import numpy as np


def per_query(rank: int | None) -> dict[str, float]:
    """rank is 1-based position of the relevant patent, or None if not retrieved."""
    r = rank
    return {
        "R@1": float(r is not None and r <= 1),
        "R@10": float(r is not None and r <= 10),
        "R@30": float(r is not None and r <= 30),
        "R@100": float(r is not None and r <= 100),
        "MRR@10": 1.0 / r if r is not None and r <= 10 else 0.0,
        "nDCG@10": 1.0 / math.log2(r + 1) if r is not None and r <= 10 else 0.0,
    }


METRICS = ("R@1", "R@10", "R@30", "R@100", "MRR@10", "nDCG@10")


def aggregate(ranks: list[int | None], n_boot: int = 1000, seed: int = 13) -> dict:
    if not ranks:
        return {"n": 0}
    rows = np.array([[per_query(r)[m] for m in METRICS] for r in ranks])
    mean = rows.mean(axis=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(rows), size=(n_boot, len(rows)))
    boots = rows[idx].mean(axis=1)
    lo, hi = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)
    return {"n": len(ranks), **{m: {"mean": float(mean[i]), "ci95": [float(lo[i]), float(hi[i])]}
                                for i, m in enumerate(METRICS)}}


def paired_delta(ranks_a: list, ranks_b: list, metric: str = "MRR@10", n_boot: int = 1000, seed: int = 13) -> dict:
    """Mean (b - a) over the same queries with a paired bootstrap 95% CI."""
    a = np.array([per_query(r)[metric] for r in ranks_a])
    b = np.array([per_query(r)[metric] for r in ranks_b])
    d = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return {"metric": metric, "delta": float(d.mean()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def pair_metrics(pos: list[float], negs: list[list[float]]) -> dict:
    """pos[i] = cos(anchor_i, positive_i); negs[i] = cos(anchor_i, each hard negative)."""
    trip = [p > n for p, ns in zip(pos, negs) for n in ns]
    all_neg = [n for ns in negs for n in ns]
    if all_neg:
        p = np.array(pos)[:, None]
        n = np.array(all_neg)[None, :]
        auc = float((p > n).mean() + 0.5 * (p == n).mean())
    else:
        auc = float("nan")
    return {"triplets": len(trip), "triplet_accuracy": float(np.mean(trip)) if trip else float("nan"),
            "roc_auc": auc, "mean_cos_pos": float(np.mean(pos)) if pos else float("nan"),
            "mean_cos_neg": float(np.mean(all_neg)) if all_neg else float("nan"),
            "margin": float(np.mean(pos) - np.mean(all_neg)) if pos and all_neg else float("nan")}
