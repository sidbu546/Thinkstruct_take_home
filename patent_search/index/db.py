"""SQLite storage: metadata, claims, chunks, data issues, FTS5 indexes (DESIGN.md §3)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE patents (
  doc_number TEXT PRIMARY KEY, title TEXT, title_norm TEXT, abstract TEXT,
  classification TEXT, cpc_subclass TEXT, filing_week TEXT, source_file TEXT,
  source_xml TEXT, bibtex TEXT, data_status TEXT, record_version INTEGER DEFAULT 1,
  n_claims_active INTEGER, n_claims_canceled INTEGER, description_words INTEGER
);
CREATE INDEX ix_pat_cpc ON patents(classification);
CREATE INDEX ix_pat_week ON patents(filing_week);
CREATE INDEX ix_pat_title ON patents(title_norm);

CREATE TABLE claims (
  doc_number TEXT, claim_number INTEGER, number_inferred INTEGER, status TEXT,
  depends_on TEXT, unresolved_refs TEXT, depth INTEGER, n_elements INTEGER, tier TEXT,
  preamble TEXT, elements TEXT, text TEXT,
  PRIMARY KEY (doc_number, claim_number)
);

CREATE TABLE chunks (
  id INTEGER PRIMARY KEY, chunk_key TEXT UNIQUE, doc_number TEXT, kind TEXT,
  claim_number INTEGER, element_idx INTEGER, section TEXT, para_span TEXT,
  parent_chunk_id INTEGER, tier TEXT, n_tokens INTEGER, context_claims TEXT,
  embed_text TEXT, display_text TEXT, weight REAL
);
CREATE INDEX ix_chunk_doc ON chunks(doc_number);

CREATE TABLE data_issues (
  id INTEGER PRIMARY KEY, doc_number TEXT, location TEXT, issue_type TEXT, severity TEXT,
  status TEXT, detail TEXT, detected_at TEXT DEFAULT CURRENT_TIMESTAMP, resolved_at TEXT,
  resolution_source TEXT
);
CREATE INDEX ix_issue_doc ON data_issues(doc_number, status);

CREATE TABLE quarantine (id INTEGER PRIMARY KEY, file TEXT, position INTEGER, reason TEXT, detail TEXT);

CREATE VIRTUAL TABLE patents_fts USING fts5(title, abstract, content='patents',
  tokenize='porter unicode61');
CREATE VIRTUAL TABLE chunks_fts USING fts5(display_text, content='chunks', content_rowid='id',
  tokenize='porter unicode61');
"""


def connect(path: Path, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    else:
        con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con


def create(path: Path) -> sqlite3.Connection:
    if path.exists():
        path.unlink()
    con = connect(path)
    con.executescript(SCHEMA)
    return con


def insert_patent(con, p, status: str, n_active: int, n_canceled: int, desc_words: int, title_norm: str):
    cpc = p.classification
    con.execute(
        "INSERT INTO patents VALUES (?,?,?,?,?,?,?,?,?,?,?,1,?,?,?)",
        (p.doc_number, p.title, title_norm, p.abstract, cpc, cpc[:4] if cpc else None, p.filing_week,
         p.source_file, p.source_xml, p.bibtex, status, n_active, n_canceled, desc_words))


def insert_claims(con, doc: str, claims):
    con.executemany(
        "INSERT INTO claims VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        [(doc, c.number, int(c.number_inferred), c.status, json.dumps(c.depends_on),
          json.dumps(c.unresolved_refs), c.depth, len(c.elements), c.tier, c.preamble,
          json.dumps(c.elements), c.text) for c in claims])


def insert_chunks(con, rows: list[tuple]):
    con.executemany("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)


def insert_issues(con, issues: list[dict]):
    con.executemany(
        "INSERT INTO data_issues (doc_number, location, issue_type, severity, status, detail) "
        "VALUES (?,?,?,?,?,?)",
        [(i["doc_number"], i["location"], i["issue_type"], i["severity"], i["status"], i["detail"])
         for i in issues])


def insert_quarantine(con, rows: list[dict]):
    con.executemany("INSERT INTO quarantine (file, position, reason, detail) VALUES (?,?,?,?)",
                    [(q.get("file"), q.get("position"), q.get("reason"),
                      json.dumps({k: v for k, v in q.items() if k not in ("file", "position", "reason")}))
                     for q in rows])


def rebuild_fts(con):
    con.execute("INSERT INTO patents_fts(patents_fts) VALUES('rebuild')")
    con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
