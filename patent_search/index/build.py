"""build-index: JSON files -> SQLite (+FTS5) + vectors.npy + manifest.json."""
from __future__ import annotations

import collections
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from ..config import Config
from ..embed.cache import EmbeddingCache
from ..embed.encoder import get_encoder
from ..ingest.chunk import abstract_chunk, claim_chunks, description_chunks, fixed_chunks
from ..ingest.claims import assign_tier, parse_claims
from ..ingest.load import load_dir
from ..ingest.normalize import normalize_title
from ..ingest.validate import data_status, description_word_count, detect_issues
from ..manifest import write_manifest
from . import db


def prepare(data_dir: Path, cfg: Config):
    """Everything up to (not including) embedding. Deterministic."""
    lr = load_dir(data_dir)
    patents, all_chunks, all_issues = [], [], []
    for p in lr.patents:
        claims = parse_claims(p.claims)
        if cfg.chunking.mode == "fixed":
            body = fixed_chunks(p.doc_number, p.title, claims, p.paragraphs, cfg.chunking)
        else:
            body = (claim_chunks(p.doc_number, p.title, claims, cfg.chunking)
                    + description_chunks(p.doc_number, p.title, p.paragraphs, cfg.chunking))
        chunks = abstract_chunk(p.doc_number, p.title, p.abstract, cfg.chunking) + body
        if cfg.chunking.mode == "fixed":            # adaptive mode sets tiers inside claim_chunks
            for c in claims:
                if c.status == "active":
                    c.tier = assign_tier(c, cfg.chunking)
        dwords = description_word_count(p.paragraphs)
        issues = detect_issues(p, claims, dwords)
        patents.append((p, claims, data_status(issues), dwords))
        all_chunks.extend(chunks)
        all_issues.extend(issues)
    return lr, patents, all_chunks, all_issues


def embed_all(texts: list[str], cfg: Config, encoder, use_cache: bool = True, log=print) -> np.ndarray:
    """Encode each distinct text once (identical chunks get identical vectors), then scatter back."""
    first: dict[str, int] = {}
    inverse = np.array([first.setdefault(t, len(first)) for t in texts], dtype=np.int64)
    unique = list(first)
    if len(unique) < len(texts):
        log(f"  {len(texts) - len(unique)} chunks share text with another chunk; encoding {len(unique)} distinct")
    return _embed_unique(unique, cfg, encoder, use_cache, log)[inverse]


def _embed_unique(texts: list[str], cfg: Config, encoder, use_cache: bool, log) -> np.ndarray:
    vecs = np.zeros((len(texts), encoder.dim), dtype=np.float32)
    cache = EmbeddingCache(cfg.cache_path, encoder.name, encoder.revision, encoder.dim) if use_cache else None
    todo = list(range(len(texts)))
    if cache:
        hit = cache.get_many(texts)
        for i, v in hit.items():
            vecs[i] = v
        todo = [i for i in todo if i not in hit]
        log(f"  embedding cache: {len(hit)} hits, {len(todo)} to encode")
    # sort by length so batches pad little; stable -> deterministic batch composition
    todo.sort(key=lambda i: (len(texts[i]), i))
    group = 256
    t0 = time.time()
    for g in range(0, len(todo), group):
        idx = todo[g:g + group]
        out = encoder.encode([texts[i] for i in idx], batch_size=cfg.embed_batch_size)
        vecs[idx] = out
        if cache:
            cache.put_many([texts[i] for i in idx], out)       # checkpoint: a crash resumes here
        done = g + len(idx)
        rate = done / max(time.time() - t0, 1e-9)
        log(f"  encoded {done}/{len(todo)}  ({rate:.1f} chunks/s, "
            f"eta {(len(todo) - done) / max(rate, 1e-9) / 60:.1f} min)")
    if cache:
        cache.close()
    return vecs


def build_index(data_dir: Path, cfg: Config, use_cache: bool = True, log=print) -> dict:
    t_start = time.time()
    cfg.index_dir.mkdir(parents=True, exist_ok=True)
    log(f"loading + parsing + chunking {data_dir} ...")
    lr, patents, chunks, issues = prepare(data_dir, cfg)
    t_prep = time.time() - t_start

    log(f"loading encoder ({cfg.embed_backend}: {cfg.model_name if cfg.embed_backend == 'local' else 'fake'}, "
        f"device={cfg.device}) ...")
    encoder = get_encoder(cfg)

    con = db.create(cfg.db_path)
    for p, claims, status, dwords in patents:
        n_act = sum(c.status == "active" for c in claims)
        db.insert_patent(con, p, status, n_act, len(claims) - n_act, dwords, normalize_title(p.title))
        db.insert_claims(con, p.doc_number, claims)
    key_to_id = {c.key: i for i, c in enumerate(chunks)}
    texts = [c.embed_text for c in chunks]
    n_tokens = encoder.count_tokens(texts)
    db.insert_chunks(con, [
        (i, c.key, c.doc_number, c.kind, c.claim_number, c.element_idx, c.section, c.para_span,
         key_to_id.get(c.parent_key) if c.parent_key else None, c.tier, n_tokens[i], c.context_claims,
         c.embed_text, c.display_text, c.weight) for i, c in enumerate(chunks)])
    db.insert_issues(con, issues)
    db.insert_quarantine(con, lr.quarantine)
    db.rebuild_fts(con)
    con.commit()

    log(f"embedding {len(texts)} chunks locally ...")
    t_emb = time.time()
    vecs = embed_all(texts, cfg, encoder, use_cache=use_cache, log=log)
    t_emb = time.time() - t_emb
    np.save(cfg.vectors_path, vecs)
    con.close()

    report = {
        "patents_loaded": len(patents),
        "records_quarantined": len(lr.quarantine),
        "quarantine": lr.quarantine,
        "duplicates_exact_skipped": lr.duplicates_exact,
        "duplicates_conflicting": lr.duplicates_conflict,
        "claims_active": sum(sum(c.status == "active" for c in cl) for _, cl, _, _ in patents),
        "claims_canceled_kept_as_status_rows": sum(sum(c.status == "canceled" for c in cl)
                                                   for _, cl, _, _ in patents),
        "empty_paragraphs_structural": sum(sum(1 for x in p.paragraphs if not str(x or "").strip())
                                           for p, _, _, _ in patents),
        "chunks_by_kind": dict(collections.Counter(c.kind for c in chunks)),
        "claim_tiers": dict(collections.Counter(c.tier for c in chunks if c.kind == "claim")),
        "chunks_over_max_seq_length": sum(n > cfg.max_seq_length for n in n_tokens),
        "tokens_total": int(sum(n_tokens)),
        "data_issues_by_type": dict(collections.Counter(i["issue_type"] for i in issues)),
        "data_issues_by_severity": dict(collections.Counter(i["severity"] for i in issues)),
        "patents_by_data_status": dict(collections.Counter(s for _, _, s, _ in patents)),
    }
    timings = {"prepare_s": round(t_prep, 1), "embed_s": round(t_emb, 1),
               "total_s": round(time.time() - t_start, 1),
               "chunks_per_s": round(len(texts) / max(t_emb, 1e-9), 1)}
    manifest = write_manifest(cfg, encoder, lr.files, report, timings, chunks_digest(chunks))
    (cfg.index_dir / "ingest_report.json").write_text(json.dumps(report, indent=2))
    return manifest


def chunks_digest(chunks) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.key.encode())
        h.update(b"\0")
        h.update(c.embed_text.encode())
        h.update(b"\1")
    return h.hexdigest()
