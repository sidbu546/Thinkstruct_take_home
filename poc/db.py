"""PoC state store (SCALE.md §3, §10): one SQLite file in WAL mode standing in for Postgres.

jobs          the work queue (claim with lease, attempts, dead state) — stands in for SQS + jobs table
patents       canonical record per doc_number (+ chunk stats; chunk text itself would live in Parquet)
patent_versions  every distinct content seen per doc_number (duplicates with different content)
data_issues   Part 1 data-issue rows (never dropped, pending data research)
quarantine    records/bytes that could not be parsed, with file and byte offset
workers       heartbeats
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "poc_data" / "poc.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  job_id INTEGER PRIMARY KEY,
  input_uri TEXT NOT NULL,          -- zip://<zip>!<member>  or  file://<path>
  idempotency_key TEXT UNIQUE,      -- hash(stage, input identity (CRC32+size), pipeline version)
  status TEXT NOT NULL DEFAULT 'queued',   -- queued | running | succeeded | partial | dead
  attempts INTEGER NOT NULL DEFAULT 0,
  max_attempts INTEGER NOT NULL DEFAULT 3,
  worker_id TEXT, lease_until REAL, not_before REAL,
  created_at REAL, started_at REAL, finished_at REAL,
  records_in INTEGER, patents_written INTEGER, duplicates_exact INTEGER, duplicates_conflict INTEGER,
  quarantined INTEGER, issues_opened INTEGER, chunks INTEGER,
  parse_s REAL, chunk_s REAL, embed_s REAL, write_s REAL, total_s REAL,
  embed_chunks INTEGER, embed_tokens INTEGER,
  error TEXT, history TEXT                    -- JSON list of {attempt, worker, outcome, error}
);
CREATE INDEX IF NOT EXISTS ix_jobs_status ON jobs(status, not_before);

CREATE TABLE IF NOT EXISTS patents (
  doc_number TEXT PRIMARY KEY, title TEXT, classification TEXT, data_status TEXT,
  content_sha256 TEXT, source_uri TEXT, source_position INTEGER, job_id INTEGER,
  n_claims INTEGER, n_chunks INTEGER, chunk_tokens_est INTEGER, record_version INTEGER DEFAULT 1
);
CREATE TABLE IF NOT EXISTS patent_versions (
  doc_number TEXT, content_sha256 TEXT, source_uri TEXT, source_position INTEGER, completeness INTEGER,
  is_canonical INTEGER, seen_at REAL, PRIMARY KEY (doc_number, content_sha256)
);
CREATE TABLE IF NOT EXISTS data_issues (
  doc_number TEXT, issue_type TEXT, location TEXT, severity TEXT, status TEXT, detail TEXT, job_id INTEGER,
  PRIMARY KEY (doc_number, issue_type, location)
);
CREATE TABLE IF NOT EXISTS quarantine (
  input_uri TEXT, position INTEGER, byte_offset INTEGER, reason TEXT, excerpt TEXT, job_id INTEGER,
  PRIMARY KEY (input_uri, position, byte_offset)
);
CREATE TABLE IF NOT EXISTS workers (
  worker_id TEXT PRIMARY KEY, pid INTEGER, started_at REAL, last_heartbeat REAL, current_job INTEGER,
  jobs_done INTEGER DEFAULT 0, state TEXT
);
"""


def connect(path: Path = DEFAULT_DB) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), timeout=60, isolation_level=None)   # autocommit; explicit BEGINs
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA busy_timeout=60000")
    con.executescript(SCHEMA)
    return con


def now() -> float:
    return time.time()
