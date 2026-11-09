"""Process one ingestion job: read -> tolerant parse (salvage) -> Part 1 ingest (claims, adaptive chunks,
data issues) -> optional embed sample -> one transaction: dedupe + upsert + issues + quarantine + job done.

Idempotent: re-running a job rewrites the same rows. A record already ingested from the same source
(same uri + position + content) is a no-op, not a duplicate."""
from __future__ import annotations

import hashlib
import json
import random
import time
import zipfile
from pathlib import Path

from patent_search.config import ChunkingConfig
from patent_search.ingest.chunk import abstract_chunk, claim_chunks, description_chunks
from patent_search.ingest.claims import parse_claims
from patent_search.ingest.load import EXPECTED_KEYS, RawPatent, _as_list, _clean_str, _week
from patent_search.ingest.validate import data_status, description_word_count, detect_issues

from . import queue

CHUNK_CFG = ChunkingConfig()


class InjectedFailure(RuntimeError):
    pass


def read_input(uri: str) -> bytes:
    if uri.startswith("zip://"):
        zpath, member = uri[len("zip://"):].split("!", 1)
        with zipfile.ZipFile(zpath) as z, z.open(member) as f:
            return f.read()
    if uri.startswith("file://"):
        return Path(uri[len("file://"):]).read_bytes()
    raise ValueError(f"unknown input uri scheme: {uri}")


def source_name(uri: str) -> str:
    return uri.split("!", 1)[-1] if "!" in uri else uri[len("file://"):]


def parse_tolerant(data: bytes) -> tuple[list, list[dict], bool]:
    """Returns (records, quarantine rows, salvaged?). A file that is not valid JSON is decoded record by
    record from the top-level array; everything from the first unparseable byte on is quarantined."""
    try:
        obj = json.loads(data.decode("utf-8"))
        return (obj if isinstance(obj, list) else [obj]), [], False
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass
    text = data.decode("utf-8", errors="replace")
    dec = json.JSONDecoder()
    i = text.find("[")
    if i < 0:
        return [], [{"position": 0, "byte_offset": 0, "reason": "no JSON array found", "excerpt": text[:200]}], True
    i += 1
    recs, quar = [], []
    n = len(text)
    while True:
        while i < n and text[i] in " \t\r\n,":
            i += 1
        if i >= n or text[i] == "]":
            if i >= n:
                quar.append({"position": len(recs), "byte_offset": len(data), "reason": "file ends mid-array "
                             "(truncated)", "excerpt": ""})
            break
        try:
            obj, j = dec.raw_decode(text, i)
        except json.JSONDecodeError as e:
            quar.append({"position": len(recs), "byte_offset": len(text[:i].encode("utf-8")),
                         "reason": f"unparseable from here on ({e.msg}); {len(text) - i} chars not recoverable",
                         "excerpt": text[i:i + 200]})
            break
        recs.append(obj)
        i = j
    return recs, quar, True


def _raw_patent(r: dict, uri: str, pos: int) -> RawPatent:
    xml = _clean_str(r.get("filename"))
    return RawPatent(
        doc_number=_clean_str(r.get("doc_number")), title=_clean_str(r.get("title")),
        abstract=_clean_str(r.get("abstract")), classification=_clean_str(r.get("classification")) or None,
        filing_week=_week(Path(source_name(uri)), xml), source_file=source_name(uri), position=pos, source_xml=xml,
        bibtex=_clean_str(r.get("bibtex")), claims=_as_list(r.get("claims")),
        paragraphs=_as_list(r.get("detailed_description")),
        present_keys=frozenset(k for k in EXPECTED_KEYS if k in r))


def process_job(con, job, worker_id: str, fail_rate: float = 0.0, embed_sample: int = 0, encoder=None,
                seed: int = 13) -> dict:
    t0 = time.perf_counter()
    uri, job_id = job["input_uri"], job["job_id"]
    data = read_input(uri)
    recs, quar, salvaged = parse_tolerant(data)
    del data
    t_parse = time.perf_counter()

    prepared, issues = [], []
    for pos, r in enumerate(recs):
        if not isinstance(r, dict):
            quar.append({"position": pos, "byte_offset": -1, "reason": "record is not a JSON object",
                         "excerpt": json.dumps(r)[:200]})
            continue
        p = _raw_patent(r, uri, pos)
        if not p.doc_number:
            quar.append({"position": pos, "byte_offset": -1, "reason": "no doc_number (pending data research)",
                         "excerpt": (p.title or "")[:200]})
            continue
        claims = parse_claims(p.claims)
        chunks = (abstract_chunk(p.doc_number, p.title, p.abstract, CHUNK_CFG)
                  + claim_chunks(p.doc_number, p.title, claims, CHUNK_CFG)
                  + description_chunks(p.doc_number, p.title, p.paragraphs, CHUNK_CFG))
        iss = detect_issues(p, claims, description_word_count(p.paragraphs))
        filled, size = p.completeness()
        prepared.append((p, claims, chunks, iss, data_status(iss), p.content_hash(), filled * 10**9 + size))
        issues.extend(iss)
    t_chunk = time.perf_counter()

    emb_chunks = emb_tokens = 0
    if embed_sample and encoder is not None and prepared:
        texts = [c.embed_text for (_, _, ch, *_rest) in prepared[:embed_sample] for c in ch]
        emb_tokens = int(sum(encoder.count_tokens(texts)))
        encoder.encode(texts, batch_size=16)
        emb_chunks = len(texts)
    t_embed = time.perf_counter()

    rng = random.Random(hash((seed, job_id, job["attempts"])) & 0xFFFFFFFF)
    if fail_rate and rng.random() < fail_rate:
        raise InjectedFailure(f"injected transient failure (attempt {job['attempts']})")

    written = dup_exact = dup_conflict = n_issues = 0
    con.execute("BEGIN IMMEDIATE")
    try:
        for p, claims, chunks, iss, status, h, completeness in prepared:
            cur = con.execute("SELECT content_sha256, source_uri, source_position FROM patents WHERE doc_number=?",
                              (p.doc_number,)).fetchone()
            con.execute("INSERT OR IGNORE INTO patent_versions VALUES (?,?,?,?,?,0,?)",
                        (p.doc_number, h, uri, p.position, completeness, time.time()))
            is_new_or_same_source = cur is None or (cur["source_uri"] == uri and cur["source_position"] == p.position)
            if not is_new_or_same_source:
                if cur["content_sha256"] == h:
                    dup_exact += 1                      # same content seen before from another file: skip
                    continue
                dup_conflict += 1                       # same doc_number, different content
                best = con.execute("SELECT content_sha256, source_uri, source_position FROM patent_versions "
                                   "WHERE doc_number=? ORDER BY completeness DESC, seen_at DESC LIMIT 1",
                                   (p.doc_number,)).fetchone()
                con.execute("INSERT OR REPLACE INTO data_issues VALUES (?,?,?,?,?,?,?)",
                            (p.doc_number, "duplicate_conflict", "record", "medium", "open",
                             json.dumps({"versions": con.execute("SELECT COUNT(*) FROM patent_versions WHERE "
                                                                 "doc_number=?", (p.doc_number,)).fetchone()[0],
                                         "canonical": f"{source_name(best['source_uri'])}[{best['source_position']}]"}),
                             job_id))
                n_issues += 1
                if best["content_sha256"] != h:
                    continue                            # the existing record stays canonical
            con.execute("UPDATE patent_versions SET is_canonical = (content_sha256 = ?) WHERE doc_number=?",
                        (h, p.doc_number))
            con.execute("INSERT OR REPLACE INTO patents VALUES (?,?,?,?,?,?,?,?,?,?,?,1)",
                        (p.doc_number, p.title, p.classification, status, h, uri, p.position, job_id,
                         sum(c.status == "active" for c in claims), len(chunks),
                         int(sum(c.n_tokens_est for c in chunks))))
            for i in iss:
                con.execute("INSERT OR REPLACE INTO data_issues VALUES (?,?,?,?,?,?,?)",
                            (i["doc_number"], i["issue_type"], i["location"], i["severity"], i["status"],
                             i["detail"], job_id))
            n_issues += len(iss)
            written += 1
        for q in quar:
            con.execute("INSERT OR IGNORE INTO quarantine VALUES (?,?,?,?,?,?)",
                        (uri, q["position"], q["byte_offset"], q["reason"], q["excerpt"], job_id))
        if salvaged:
            con.execute("INSERT OR REPLACE INTO data_issues VALUES (?,?,?,?,?,?,?)",
                        (f"(file) {source_name(uri)}", "corrupt_source", uri, "high", "open",
                         json.dumps({"records_salvaged": len(recs), "quarantined_rows": len(quar)}), job_id))
        t_write = time.perf_counter()
        stats = {"records_in": len(recs),
                 "patents_written": written, "duplicates_exact": dup_exact, "duplicates_conflict": dup_conflict,
                 "quarantined": len(quar), "issues_opened": n_issues, "chunks": sum(len(x[2]) for x in prepared),
                 "parse_s": round(t_parse - t0, 3), "chunk_s": round(t_chunk - t_parse, 3),
                 "embed_s": round(t_embed - t_chunk, 3), "write_s": round(t_write - t_embed, 3),
                 "total_s": round(t_write - t0, 3), "embed_chunks": emb_chunks, "embed_tokens": emb_tokens}
        status = "partial" if quar else "succeeded"
        if not queue.complete_in_tx(con, job_id, worker_id, status, stats):
            con.execute("ROLLBACK")             # lease lost: another worker owns the job now
            return {"outcome": "lease_lost"}
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return {"outcome": status, **stats}


def idempotency_key(member: str, crc: int, size: int, pipeline: str) -> str:
    return hashlib.sha256(f"ingest|{member}|{crc:08x}|{size}|{pipeline}".encode()).hexdigest()
