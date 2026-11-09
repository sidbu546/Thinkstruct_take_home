"""Hybrid retrieval (DESIGN.md §4): dense + BM25 over chunks, both restricted to the allowed
set, fused with Reciprocal Rank Fusion, rolled up to patents."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..ingest.normalize import content_tokens
from .planner import dense_topk
from .query import AnalyzedQuery


@dataclass
class RankedPatent:
    doc_number: str
    score: float
    best_row: int
    rrf: float
    dense_rank: int | None = None
    bm25_rank: int | None = None


@dataclass
class Retrieval:
    query_vecs: np.ndarray
    chunk_scores: dict[int, float]
    ranked: list[RankedPatent]
    dense_path: str
    doc_rows_hit: dict[str, list[int]] = field(default_factory=dict)


def _quote(t: str) -> str:
    return '"' + t.replace('"', '""') + '"'


def bm25_rows(con, text: str, allowed_rows: np.ndarray | None, n_total: int, k: int) -> list[int]:
    toks = list(dict.fromkeys(content_tokens(text)))[:64]
    if not toks:
        return []
    q = " OR ".join(_quote(t) for t in toks)
    base = "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?"
    if allowed_rows is None:
        return [r[0] for r in con.execute(base + " ORDER BY bm25(chunks_fts) LIMIT ?", (q, k))]
    if len(allowed_rows) < 0.5 * n_total:            # selective: restrict inside the query
        con.execute("CREATE TEMP TABLE IF NOT EXISTS allowed_rows (id INTEGER PRIMARY KEY)")
        con.execute("DELETE FROM temp.allowed_rows")
        con.executemany("INSERT INTO temp.allowed_rows VALUES (?)", ((int(r),) for r in allowed_rows))
        # JOIN, not "rowid IN (subquery)": FTS5 evaluates the IN form per match (measured ~2 s vs
        # ~3 ms on this index for identical results).
        sql = ("SELECT f.rowid FROM chunks_fts f JOIN temp.allowed_rows a ON a.id = f.rowid "
               "WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?")
        return [r[0] for r in con.execute(sql, (q, k))]
    allowed = set(int(r) for r in allowed_rows)       # broad: over-fetch, post-filter
    over = k + (n_total - len(allowed))
    rows = [r[0] for r in con.execute(base + " ORDER BY bm25(chunks_fts) LIMIT ?", (q, over))]
    return [r for r in rows if r in allowed][:k]


def retrieve(engine, aq: AnalyzedQuery, mode: str, allowed_rows: np.ndarray | None,
             retrieval: str = "hybrid") -> Retrieval:
    """retrieval: hybrid (dense + BM25, the default) | dense | bm25 (the last two for evaluation)."""
    sc = engine.cfg.search
    texts = [aq.text] + aq.elements
    qv = engine.encoder.encode(texts)
    lists: list[tuple[float, list[int]]] = []
    path, dense_first = "skipped", []
    if retrieval in ("hybrid", "dense"):
        dense_lists, path = dense_topk(engine.vectors, qv, allowed_rows, sc.retrieve_per_list, sc.exact_max_chunks)
        dense_first = [r for r, _ in dense_lists[0]]
        lists.append((1.0, dense_first))
        if aq.elements:
            w = 1.0 / len(aq.elements)
            lists += [(w, [r for r, _ in dl]) for dl in dense_lists[1:]]
    bm: list[int] = []
    if retrieval in ("hybrid", "bm25"):
        bm = bm25_rows(engine.con, aq.text, allowed_rows, engine.n_chunks, sc.retrieve_per_list)
        lists.append((1.0, bm))

    kind_w = sc.kind_weights_claim if mode in ("claim", "patent") else sc.kind_weights_text
    raw: dict[int, float] = defaultdict(float)
    for w, rows in lists:
        for rank, row in enumerate(rows):
            raw[row] += w / (sc.rrf_k + rank + 1)
    chunk_scores = {row: s * kind_w[engine.kind[row]] * engine.weight[row] for row, s in raw.items()}

    per_doc: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for row, s in chunk_scores.items():
        per_doc[engine.doc_of[row]].append((s, row))

    def first_rank(rows: list[int]) -> dict[str, int]:
        out: dict[str, int] = {}
        for row in rows:
            d = engine.doc_of[row]
            if d not in out:
                out[d] = len(out) + 1
        return out

    dense_rank, bm_rank = first_rank(dense_first), first_rank(bm)
    ranked = []
    for d, items in per_doc.items():
        items.sort(key=lambda x: (-x[0], x[1]))
        s1 = items[0][0]
        s2 = items[1][0] if len(items) > 1 else 0.0
        score = s1 + sc.second_hit_bonus * s2
        ranked.append(RankedPatent(d, score, items[0][1], s1, dense_rank.get(d), bm_rank.get(d)))
    ranked.sort(key=lambda r: (-r.score, r.doc_number))
    return Retrieval(qv, chunk_scores, ranked, path,
                     {d: [r for _, r in items] for d, items in per_doc.items()})
