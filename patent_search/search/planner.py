"""Dense retrieval planner (DESIGN.md §4.1).

small allowed set  -> gather only those rows of the vector matrix, exact dot product
large / no filter  -> full-matrix exact scan, disallowed rows masked out
(At 10M patents the second path becomes ANN + over-fetch + post-filter; see SCALE.md.)
Both paths here are exact, so the choice affects speed, never results."""
from __future__ import annotations

import numpy as np


def dense_topk(vectors: np.ndarray, queries: np.ndarray, allowed_rows: np.ndarray | None,
               k: int, exact_max: int) -> tuple[list[list[tuple[int, float]]], str]:
    """Return, per query vector, the top-k (row, cosine) pairs, and the path taken."""
    q = np.atleast_2d(queries).astype(np.float32)
    if allowed_rows is not None and len(allowed_rows) == 0:
        return [[] for _ in range(len(q))], "empty"
    if allowed_rows is not None and len(allowed_rows) <= exact_max:
        sub = vectors[allowed_rows]
        scores = q @ sub.T                                   # (nq, |allowed|)
        rows = allowed_rows
        path = "exact_subset"
    else:
        scores = q @ vectors.T
        if allowed_rows is not None:
            mask = np.full(vectors.shape[0], -np.inf, dtype=np.float32)
            mask[allowed_rows] = 0.0
            scores = scores + mask
        rows = None
        path = "full_scan_masked" if allowed_rows is not None else "full_scan"
    out = []
    kk = min(k, scores.shape[1])
    for s in scores:
        top = np.argpartition(-s, kk - 1)[:kk]
        top = top[np.argsort(-s[top], kind="stable")]
        out.append([(int(rows[i] if rows is not None else i), float(s[i])) for i in top if np.isfinite(s[i])])
    return out, path
