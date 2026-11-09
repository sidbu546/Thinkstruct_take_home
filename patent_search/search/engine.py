"""Search engine: loads the index once, answers SearchRequest -> SearchResponse."""
from __future__ import annotations

import json
import threading
import time
import uuid
from collections import Counter, defaultdict

import numpy as np

from ..config import Config
from ..embed.encoder import get_encoder
from ..index import db
from ..manifest import load_manifest
from ..notices import issue_notice, response_notice
from ..schemas import (BestMatch, ClaimMapEntry, ClaimMatch, DataIssueOut, Facets, QueryInfo, Result,
                       SearchRequest, SearchResponse, Signals)
from .coverage import element_coverage, highlight_spans
from .filters import resolve
from .hybrid import retrieve
from .query import analyze
from .rerank import ScoreCache, get_reranker, phase2


class IndexMismatch(RuntimeError):
    pass


class Engine:
    def __init__(self, cfg: Config, encoder=None):
        self.cfg = cfg
        self.manifest = load_manifest(cfg)
        built = self.manifest["embedding"]
        if built["backend"] != cfg.embed_backend:
            raise IndexMismatch(f"index was built with the {built['backend']} backend, config asks for "
                                f"{cfg.embed_backend}. Rebuild the index or change PS_EMBED_BACKEND.")
        # queries must be encoded with the model that built this index (base or fine-tuned)
        from dataclasses import replace
        cfg = replace(cfg, model_name=built["model"], model_revision=built["revision"])
        # coverage thresholds are model-specific (cosine scales differ): use the ones calibrated for
        # THIS index by scripts/calibrate_coverage.py when present
        cal = cfg.index_dir / "coverage_calibration.json"
        self.coverage_calibration = None
        if cal.exists():
            import json as _json
            rec = _json.loads(cal.read_text())["recommended"]
            cfg = replace(cfg, search=replace(cfg.search, coverage_found=rec["coverage_found"],
                                              coverage_weak=rec["coverage_weak"]))
            self.coverage_calibration = rec
        self.cfg = cfg
        self.con = db.connect(cfg.db_path, readonly=True)
        self.vectors = np.load(cfg.vectors_path)
        rows = self.con.execute(
            "SELECT id, doc_number, kind, claim_number, element_idx, section, para_span, parent_chunk_id, "
            "tier, context_claims, display_text, weight FROM chunks ORDER BY id").fetchall()
        self.n_chunks = len(rows)
        if self.n_chunks != self.vectors.shape[0]:
            raise IndexMismatch(f"{self.n_chunks} chunks in the database but {self.vectors.shape[0]} vectors")
        self.doc_of = [r["doc_number"] for r in rows]
        self.kind = [r["kind"] for r in rows]
        self.claim_number = [r["claim_number"] for r in rows]
        self.element_idx = [r["element_idx"] for r in rows]
        self.section = [r["section"] for r in rows]
        self.para_span = [r["para_span"] for r in rows]
        self.parent = [r["parent_chunk_id"] for r in rows]
        self.tier = [r["tier"] for r in rows]
        self.context = [r["context_claims"] for r in rows]
        self.display_text = [r["display_text"] for r in rows]
        self.weight = np.array([r["weight"] for r in rows], dtype=np.float32)
        by_doc: dict[str, list[int]] = defaultdict(list)
        for r in rows:
            by_doc[r["doc_number"]].append(r["id"])
        self.doc_rows = {d: np.array(v, dtype=np.int64) for d, v in by_doc.items()}
        self.patents = {r["doc_number"]: dict(r) for r in self.con.execute("SELECT * FROM patents")}
        self.issues: dict[str, list[dict]] = defaultdict(list)
        for r in self.con.execute("SELECT * FROM data_issues ORDER BY id"):
            self.issues[r["doc_number"]].append(dict(r))
        self.inferred = {(r["doc_number"], r["claim_number"]): bool(r["number_inferred"])
                         for r in self.con.execute("SELECT doc_number, claim_number, number_inferred FROM claims")}
        self._encoder = encoder
        self._rerankers: dict = {}
        self.rerank_cache = ScoreCache(path=cfg.rerank_cache_path)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ helpers
    @property
    def encoder(self):
        if self._encoder is None:
            self._encoder = get_encoder(self.cfg)
        return self._encoder

    def reranker(self, which: str | None = None):
        which = which or self.cfg.search.rerank_model
        if which not in self._rerankers:
            self._rerankers[which] = get_reranker(self.cfg, which)
        return self._rerankers[which]

    def _allowed_rows(self, allowed_docs, exclude_docs, hide_claims_of=(), kinds=None) -> np.ndarray | None:
        if allowed_docs is None and not exclude_docs and not hide_claims_of and not kinds:
            return None
        docs = set(self.doc_rows) if allowed_docs is None else set(allowed_docs)
        docs -= set(exclude_docs)
        hide = set(hide_claims_of)
        parts = []
        for d in sorted(docs):
            rows = self.doc_rows.get(d)
            if rows is None:
                continue
            if d in hide:                          # evaluation: own claims must not be the evidence
                rows = np.array([r for r in rows if self.kind[r] not in ("claim", "element")], dtype=np.int64)
            parts.append(rows)
        rows = np.sort(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.int64)
        if kinds:
            keep = set(kinds)
            rows = np.array([r for r in rows if self.kind[r] in keep], dtype=np.int64)
        return rows

    def best_match(self, row: int, query: str | None) -> BestMatch:
        kind = self.kind[row]
        doc = self.doc_of[row]
        cn = self.claim_number[row]
        text, element_text = self.display_text[row], None
        if kind == "element" and self.parent[row] is not None:
            element_text, text = text, self.display_text[self.parent[row]]
        ctx = [int(x) for x in self.context[row].split(",")] if self.context[row] else []
        return BestMatch(kind=kind, chunk_id=row, claim_number=cn,
                         claim_number_inferred=self.inferred.get((doc, cn)) if cn is not None else None,
                         element_idx=self.element_idx[row], element_text=element_text,
                         para_span=self.para_span[row], section=self.section[row], tier=self.tier[row],
                         context_claims=ctx, display_text=text,
                         highlights=highlight_spans(text, query) if query else [])

    def issues_out(self, doc: str) -> list[DataIssueOut]:
        return [DataIssueOut(id=i["id"], type=i["issue_type"], severity=i["severity"], status=i["status"],
                             location=i["location"], notice=issue_notice(i["issue_type"], i["location"], i["detail"]))
                for i in self.issues.get(doc, []) if i["status"] == "open" and i["severity"] != "info"]

    def _result(self, rank, doc, score, row, signals, query, matched_filters, coverage=None) -> Result:
        p = self.patents[doc]
        return Result(rank=rank, doc_number=doc, title=p["title"], classification=p["classification"],
                      filing_week=p["filing_week"], record_version=p["record_version"], score=round(score, 6),
                      signals=signals, best_match=self.best_match(row, query) if row is not None else None,
                      coverage=coverage, data_status=p["data_status"], data_issues=self.issues_out(doc),
                      matched_filters=matched_filters)

    def _facets(self, docs) -> Facets:
        ps = [self.patents[d] for d in docs]
        return Facets(
            cpc_subclass=dict(Counter(p["cpc_subclass"] or "(none)" for p in ps).most_common()),
            cpc_code=dict(Counter(p["classification"] or "(none)" for p in ps).most_common(50)),
            filing_week=dict(sorted(Counter(p["filing_week"] or "(none)" for p in ps).items())),
            data_status=dict(Counter(p["data_status"] for p in ps).most_common()))

    def _first_row(self, doc: str, kinds=("abstract",)) -> int | None:
        for r in self.doc_rows.get(doc, []):
            if self.kind[r] in kinds:
                return int(r)
        rows = self.doc_rows.get(doc)
        return int(rows[0]) if rows is not None and len(rows) else None

    # ------------------------------------------------------------------ public
    def search(self, req: SearchRequest) -> SearchResponse:
        with self._lock:
            return self._search(req)

    def _search(self, req: SearchRequest) -> SearchResponse:
        if (req.query and req.query.strip()) or req.mode == "patent":
            _ = self.encoder                  # one-time model load is not part of query latency
        t0 = time.perf_counter()
        fr = resolve(self.con, req.filters)
        pipeline = ["filter"] if fr.allowed is not None else []
        excl = list(req.exclude_docs)
        if req.mode == "patent":
            if not req.doc_number:
                raise ValueError("mode=patent needs doc_number")
            if req.doc_number not in self.patents:
                raise LookupError(f"patent {req.doc_number} is not in the index")
            excl.append(req.doc_number)
        allowed_rows = self._allowed_rows(fr.allowed, excl, req.hide_claims_of, req.chunk_kinds)
        claim_map = None
        coverage_by_doc: dict[str, list] = {}
        stage_ms: dict[str, int] = {"filter": int((time.perf_counter() - t0) * 1000)}
        rerank_info, p_notices = None, []
        sc = self.cfg.search
        do_rerank = sc.rerank_default if req.rerank is None else req.rerank

        if req.mode == "patent":
            results, pool, qinfo, claim_map, p2 = self._patent_mode(req, allowed_rows, fr.descriptions)
            pipeline += p2
            if do_rerank and req.rerank:
                p_notices.append("rerank is not applied in patent mode (its cost grows with the number of "
                                 "claims); results are in phase-1 order")
        elif not (req.query and req.query.strip()):
            if fr.allowed is None:
                raise ValueError("give a query, filters, or both")
            docs = sorted((d for d in fr.allowed if d not in excl),
                          key=lambda d: (self.patents[d]["filing_week"] or "", d), reverse=True)
            pool = docs
            qinfo = QueryInfo(mode=req.mode)
            pipeline.append("order:filing_week_desc")
            results = [self._result(i + 1, d, 0.0, self._first_row(d), Signals(), None, fr.descriptions)
                       for i, d in enumerate(docs[:req.top_k])]
        else:
            aq = analyze(req.query, req.mode, self.cfg.chunking)
            t1 = time.perf_counter()
            ret = retrieve(self, aq, req.mode, allowed_rows, req.retrieval)
            stage_ms["phase1"] = int((time.perf_counter() - t1) * 1000)
            pipeline += {"hybrid": [f"dense:{ret.dense_path}", "bm25", "rrf"], "dense": [f"dense:{ret.dense_path}"],
                         "bm25": ["bm25"]}[req.retrieval]
            pool = [r.doc_number for r in ret.ranked]
            qinfo = QueryInfo(mode=req.mode, text=aq.text, tier=aq.tier, elements=aq.elements)
            depth = req.rerank_depth or sc.rerank_depth
            phase1_rank = {r.doc_number: i + 1 for i, r in enumerate(ret.ranked)}
            cov_frac = None
            if req.mode == "claim":
                t2 = time.perf_counter()
                elements = aq.elements or [aq.text]
                evecs = ret.query_vecs[1:] if aq.elements else ret.query_vecs[:1]
                n_cov = max(sc.coverage_top_results, depth if do_rerank else 0, req.top_k if do_rerank else 0)
                for r in ret.ranked[:n_cov]:
                    coverage_by_doc[r.doc_number] = element_coverage(self, r.doc_number, elements, evecs)
                cov_frac = {d: sum(c.status == "found" for c in cov) / len(cov) for d, cov in coverage_by_doc.items()}
                stage_ms["coverage"] = int((time.perf_counter() - t2) * 1000)
            if do_rerank and ret.ranked:
                t3 = time.perf_counter()
                rr = self.reranker(req.rerank_model)
                ordered, rerank_info, notices = phase2(
                    self, rr, aq.text, ret.ranked, ret, depth, sc.rerank_chunks_per_doc, sc.rerank_budget_ms,
                    sc.rerank_alpha, sc.rerank_beta if req.mode == "claim" else 0.0, cov_frac)
                p_notices += notices
                self.last_phase2 = ordered           # kept for evaluation: lets alpha/beta be re-applied exactly
                pipeline.append(f"rerank:{rerank_info['model']}@{rerank_info['revision'][:7]}")
                stage_ms["rerank"] = int((time.perf_counter() - t3) * 1000)
            else:
                ordered = [{"doc": r.doc_number, "score": r.score, "row": r.best_row, "p1": r, "rerank": None,
                            "coverage": (cov_frac or {}).get(r.doc_number)} for r in ret.ranked]
            if req.mode == "claim":
                pipeline.append("coverage")
            results = []
            for i, o in enumerate(ordered[:req.top_k]):
                r = o["p1"]
                sig = Signals(dense_rank=r.dense_rank, bm25_rank=r.bm25_rank, rrf=round(r.rrf, 6),
                              phase1_rank=phase1_rank[o["doc"]],
                              rerank=None if o["rerank"] is None else round(o["rerank"], 6),
                              coverage=None if o["coverage"] is None else round(o["coverage"], 4))
                cov = coverage_by_doc.get(o["doc"]) if i < max(sc.coverage_top_results, req.top_k if do_rerank else 0) \
                    else None
                results.append(self._result(i + 1, o["doc"], o["score"], o["row"], sig, aq.text, fr.descriptions, cov))

        affected = sum(r.data_status != "complete" for r in results)
        return SearchResponse(
            request_id="q_" + uuid.uuid4().hex[:12], took_ms=int((time.perf_counter() - t0) * 1000),
            pipeline=pipeline, query=qinfo, filters_applied=req.filters.model_dump(exclude_defaults=True, mode="json"),
            total_matched=len(pool), data_notice=response_notice(affected, len(results)),
            filter_notices=fr.notices, results=results, facets=self._facets(pool) if req.facets else None,
            claim_map=claim_map, stage_ms=stage_ms, rerank=rerank_info, pipeline_notices=p_notices)

    def _patent_mode(self, req, allowed_rows, matched_filters):
        doc = req.doc_number
        claims = self.con.execute(
            "SELECT claim_number, text, tier FROM claims WHERE doc_number = ? AND status = 'active' "
            "AND depends_on = '[]' AND unresolved_refs = '[]' ORDER BY claim_number", (doc,)).fetchall()
        k = self.cfg.search.rrf_k
        fused: dict[str, float] = defaultdict(float)
        best: dict[str, tuple[float, int, str]] = {}
        entries = []
        queries = [(c["claim_number"], c["text"], c["tier"]) for c in claims][:10]
        if not queries and self.patents[doc]["abstract"]:
            queries = [(0, self.patents[doc]["abstract"], None)]       # no independent claim available
        path = "none"
        for cn, text, tier in queries:
            aq = analyze(text, "claim", self.cfg.chunking)
            ret = retrieve(self, aq, "claim", allowed_rows, req.retrieval)
            path = ret.dense_path
            elements = aq.elements or [aq.text]
            evecs = ret.query_vecs[1:] if aq.elements else ret.query_vecs[:1]
            for rank, r in enumerate(ret.ranked[:100]):
                fused[r.doc_number] += 1.0 / (k + rank + 1)
                if r.doc_number not in best or r.score > best[r.doc_number][0]:
                    best[r.doc_number] = (r.score, r.best_row, aq.text)
            matches = []
            for i, r in enumerate(ret.ranked[:5]):
                claim_rows = [x for x in ret.doc_rows_hit[r.doc_number] if self.kind[x] in ("claim", "element")]
                row = claim_rows[0] if claim_rows else r.best_row
                cov = element_coverage(self, r.doc_number, elements, evecs) if i < 3 else None
                matches.append(ClaimMatch(doc_number=r.doc_number, title=self.patents[r.doc_number]["title"],
                                          score=round(r.score, 6), match=self.best_match(row, aq.text), coverage=cov,
                                          data_status=self.patents[r.doc_number]["data_status"]))
            entries.append(ClaimMapEntry(claim_number=cn, claim_text=text, tier=tier, matches=matches))
        order = sorted(fused, key=lambda d: (-fused[d], d))
        results = [self._result(i + 1, d, fused[d], best[d][1], Signals(rrf=round(fused[d], 6)), best[d][2],
                                matched_filters) for i, d in enumerate(order[:req.top_k])]
        qinfo = QueryInfo(mode="patent", text=f"patent {doc}: {len(queries)} independent claim(s)",
                          elements=[f"claim {cn}" for cn, _, _ in queries])
        return results, order, qinfo, entries, [f"dense:{path}", "bm25", "rrf", "per-claim", "rrf-across-claims",
                                                "coverage"]

    def patent(self, doc: str) -> dict:
        if doc not in self.patents:
            raise LookupError(f"patent {doc} is not in the index")
        p = dict(self.patents[doc])
        p["claims"] = [dict(r) | {"depends_on": json.loads(r["depends_on"]),
                                  "unresolved_refs": json.loads(r["unresolved_refs"]),
                                  "elements": json.loads(r["elements"])}
                       for r in self.con.execute("SELECT * FROM claims WHERE doc_number = ? ORDER BY claim_number",
                                                 (doc,))]
        p["passages"] = [{"chunk_id": int(r), "kind": self.kind[r], "section": self.section[r],
                          "para_span": self.para_span[r], "text": self.display_text[r]}
                         for r in self.doc_rows.get(doc, []) if self.kind[r] == "description"]
        p["data_issues"] = [dict(i) | {"notice": issue_notice(i["issue_type"], i["location"], i["detail"])}
                            for i in self.issues.get(doc, [])]
        return p

    def data_issues(self, status: str | None = "open", severity: str | None = None, issue_type: str | None = None,
                    limit: int = 200) -> dict:
        where, params = [], []
        for col, val in (("status", status), ("severity", severity), ("issue_type", issue_type)):
            if val:
                where.append(f"{col} = ?")
                params.append(val)
        w = (" WHERE " + " AND ".join(where)) if where else ""
        summary = [dict(r) for r in self.con.execute(
            f"SELECT issue_type, severity, COUNT(*) n FROM data_issues{w} GROUP BY 1, 2 ORDER BY n DESC", params)]
        items = [dict(r) for r in self.con.execute(
            f"SELECT * FROM data_issues{w} ORDER BY id LIMIT ?", params + [limit])]
        return {"summary": summary, "items": items,
                "quarantine": [dict(r) for r in self.con.execute("SELECT * FROM quarantine")]}
