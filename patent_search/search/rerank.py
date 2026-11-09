"""Phase 2 of two-phase search (DESIGN.md Part 3B).

Phase 1 (hybrid retrieval, filters applied) hands over its top `depth` patents with their best
chunks. A cross-encoder reads (query, passage) together and scores each pair. Patent score =
max over its pairs; final = alpha * z(rerank) + (1 - alpha) * z(phase-1) [+ beta * coverage in claim
mode]. Phase 2 can only reorder phase-1 candidates, never add any, so filters always hold.

Local only: the cross-encoder runs in-process on CPU (pinned revision), like the bi-encoder."""
from __future__ import annotations

import hashlib
import re
import time
from collections import OrderedDict

import numpy as np

from ..config import Config

_WORD = re.compile(r"[a-z0-9]+")


class FakeReranker:
    """Token-overlap scorer for tests only (deterministic, no model download)."""
    name, revision, device = "fake-overlap", "v1", "cpu"

    def score(self, pairs, batch_size: int = 16) -> np.ndarray:
        out = []
        for q, p in pairs:
            a, b = set(_WORD.findall(q.lower())), set(_WORD.findall(p.lower()))
            out.append(len(a & b) / (len(a | b) or 1))
        return np.asarray(out, dtype=np.float64)

    def n_tokens(self, q: str, p: str) -> int:
        return len(_WORD.findall(q.lower())) + len(_WORD.findall(p.lower()))


class LocalReranker:
    def __init__(self, name: str, revision: str, device: str, max_length: int):
        import torch
        from sentence_transformers import CrossEncoder
        self.name, self.revision, self.max_length, self.device = name, revision, max_length, device
        self.model = CrossEncoder(name, revision=revision, device=device, max_length=max_length,
                                  model_kwargs={"dtype": torch.float32})
        self.tok = self.model.tokenizer

    def score(self, pairs, batch_size: int = 16) -> np.ndarray:
        import torch
        with torch.inference_mode():
            s = self.model.predict(list(pairs), batch_size=batch_size, show_progress_bar=False)
        return np.asarray(s, dtype=np.float64)

    def n_tokens(self, q: str, p: str) -> int:
        return len(self.tok(q, p, truncation=False)["input_ids"])


def get_reranker(cfg: Config, which: str):
    if cfg.embed_backend == "fake":
        return FakeReranker()
    name, rev = cfg.search.rerankers[which]
    return LocalReranker(name, rev, rerank_device(cfg), cfg.search.rerank_max_length)


def rerank_device(cfg: Config) -> str:
    d = cfg.search.rerank_device
    if d != "auto":
        return d
    try:
        import torch
        return "mps" if torch.backends.mps.is_available() else "cpu"
    except Exception:
        return "cpu"


def _z(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-12 else np.zeros_like(x)


class ScoreCache:
    """Cross-encoder pair scores, keyed by (model revision, device, sha1(query + passage)).

    In memory (LRU), optionally backed by a SQLite file so scores survive restarts and are shared by
    every index that shows the same passage text (e.g. base and fine-tuned indexes)."""

    def __init__(self, cap: int = 200_000, path=None):
        self.d: OrderedDict = OrderedDict()
        self.cap = cap
        self.con = None
        if path is not None:
            import sqlite3
            from pathlib import Path
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.con = sqlite3.connect(str(path), check_same_thread=False)
            self.con.execute("CREATE TABLE IF NOT EXISTS s (k TEXT PRIMARY KEY, v REAL)")

    def key(self, rev: str, device: str, query: str, passage: str) -> str:
        return f"{rev}|{device}|" + hashlib.sha1((query + "\0" + passage).encode()).hexdigest()

    def get(self, k):
        v = self.d.get(k)
        if v is not None:
            self.d.move_to_end(k)
            return v
        if self.con is not None:
            row = self.con.execute("SELECT v FROM s WHERE k = ?", (k,)).fetchone()
            if row is not None:
                self._mem(k, row[0])
                return row[0]
        return None

    def _mem(self, k, v):
        self.d[k] = v
        if len(self.d) > self.cap:
            self.d.popitem(last=False)

    def put_many(self, items: list[tuple[str, float]]):
        for k, v in items:
            self._mem(k, v)
        if self.con is not None and items:
            self.con.executemany("INSERT OR REPLACE INTO s VALUES (?, ?)", items)
            self.con.commit()


def pair_text(engine, row: int) -> str:
    """Passage shown to the cross-encoder: the claim for element hits (more context), plus title."""
    doc = engine.doc_of[row]
    if engine.kind[row] == "element" and engine.parent[row] is not None:
        row = engine.parent[row]
    return f"{engine.patents[doc]['title']}. {engine.display_text[row]}"


def phase2(engine, reranker, query: str, candidates: list, ret, depth: int, chunks_per_doc: int,
           budget_ms: int, alpha: float, beta: float, coverage_frac: dict[str, float] | None):
    """candidates: phase-1 RankedPatent list (already filtered, best first).
    Returns (ordered list of dicts, info dict, notices list)."""
    sc_cache: ScoreCache = engine.rerank_cache
    head = candidates[:depth]
    pairs, owners = [], []
    for c in head:
        for row in ret.doc_rows_hit[c.doc_number][:chunks_per_doc]:
            pairs.append((query, pair_text(engine, row)))
            owners.append((c.doc_number, row))
    t0 = time.perf_counter()
    scores = np.full(len(pairs), np.nan)
    batch = engine.cfg.search.rerank_batch
    truncated = 0
    for i in range(0, len(pairs), batch):
        if (time.perf_counter() - t0) * 1000 > budget_ms:
            break
        todo = []
        for j in range(i, min(i + batch, len(pairs))):
            v = sc_cache.get(sc_cache.key(reranker.revision, reranker.device, *pairs[j]))
            if v is None:
                todo.append(j)
            else:
                scores[j] = v
        if todo:
            out = reranker.score([pairs[j] for j in todo], batch_size=batch)
            new = []
            for j, v in zip(todo, out):
                scores[j] = float(v)
                new.append((sc_cache.key(reranker.revision, reranker.device, *pairs[j]), float(v)))
            sc_cache.put_many(new)
    max_len = getattr(reranker, "max_length", None)
    if max_len:
        truncated = sum(reranker.n_tokens(q, p) > max_len for q, p in pairs[:8])  # sampled, cheap
    per_doc: dict[str, tuple[float, int]] = {}
    for (doc, row), s in zip(owners, scores):
        if np.isnan(s):
            continue
        if doc not in per_doc or s > per_doc[doc][0]:
            per_doc[doc] = (s, row)
    scored = [c for c in head if c.doc_number in per_doc]
    rest = [c for c in candidates if c.doc_number not in per_doc]
    out = []
    if scored:
        r = np.array([per_doc[c.doc_number][0] for c in scored])
        p1 = np.array([c.score for c in scored])
        final = alpha * _z(r) + (1 - alpha) * _z(p1)
        if coverage_frac is not None and beta:
            final = final + beta * np.array([coverage_frac.get(c.doc_number, 0.0) for c in scored])
        order = sorted(range(len(scored)), key=lambda i: (-final[i], scored[i].doc_number))
        for i in order:
            c = scored[i]
            out.append({"doc": c.doc_number, "score": float(final[i]), "row": per_doc[c.doc_number][1], "p1": c,
                        "rerank": float(r[i]), "coverage": (coverage_frac or {}).get(c.doc_number)})
    for c in rest:
        out.append({"doc": c.doc_number, "score": c.score, "row": c.best_row, "p1": c, "rerank": None,
                    "coverage": (coverage_frac or {}).get(c.doc_number)})
    notices = []
    if len(scored) < len(head):
        notices.append(f"rerank covered {len(scored)} of {len(head)} candidates within the {budget_ms} ms time "
                       f"budget; the rest are in phase-1 order")
    info = {"model": reranker.name, "revision": reranker.revision, "device": reranker.device,
            "depth": len(head), "scored": len(scored),
            "pairs": int(np.sum(~np.isnan(scores))), "alpha": alpha, "beta": beta,
            "pairs_over_max_length_in_sample": truncated}
    return out, info, notices
