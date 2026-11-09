"""Job queue with leases (SCALE.md §6.1): the semantics of SQS visibility timeouts on SQLite.

claim     atomically takes the oldest runnable job: queued (and past its backoff), or running with an
          EXPIRED lease (its worker died). One UPDATE ... RETURNING inside BEGIN IMMEDIATE, so two workers
          can never claim the same job.
renew     heartbeat: extends the lease while a job is being processed.
complete  marks succeeded / partial. The worker does this in the SAME transaction as its data writes,
          so a crash can never leave "data written but job not done" (or the reverse).
fail      transient failure: back to queued with exponential backoff, or dead after max_attempts.
"""
from __future__ import annotations

import json

from .db import now

LEASE_S = 30.0


def enqueue(con, input_uri: str, idempotency_key: str, max_attempts: int = 3) -> bool:
    cur = con.execute("INSERT OR IGNORE INTO jobs (input_uri, idempotency_key, max_attempts, created_at) "
                      "VALUES (?, ?, ?, ?)", (input_uri, idempotency_key, max_attempts, now()))
    return cur.rowcount == 1


def claim(con, worker_id: str, lease_s: float = LEASE_S):
    """Return the claimed job row, or None if nothing is runnable right now."""
    while True:
        t = now()
        con.execute("BEGIN IMMEDIATE")          # takes the write lock: no other worker can claim in between
        try:
            row = con.execute(
                """SELECT job_id, status, worker_id, attempts, max_attempts FROM jobs
                   WHERE (status='queued' AND (not_before IS NULL OR not_before <= ?))
                      OR (status='running' AND lease_until < ?)
                   ORDER BY job_id LIMIT 1""", (t, t)).fetchone()
            if row is None:
                con.execute("COMMIT")
                return None
            if row["status"] == "running":     # its worker stopped heartbeating (crashed / killed)
                _append_history(con, row["job_id"], row["attempts"], row["worker_id"],
                                "lease expired (worker stopped responding)", None)
                if row["attempts"] >= row["max_attempts"]:
                    con.execute("UPDATE jobs SET status='dead', error='lease expired on final attempt', "
                                "finished_at=?, lease_until=NULL WHERE job_id=?", (t, row["job_id"]))
                    con.execute("COMMIT")
                    continue
            job = con.execute(
                """UPDATE jobs SET status='running', worker_id=?, lease_until=?, attempts=attempts+1,
                          started_at=COALESCE(started_at, ?)
                   WHERE job_id=? RETURNING *""", (worker_id, t + lease_s, t, row["job_id"])).fetchone()
            con.execute("COMMIT")
            return job
        except Exception:
            con.execute("ROLLBACK")
            raise


def renew(con, job_id: int, worker_id: str, lease_s: float = LEASE_S) -> bool:
    cur = con.execute("UPDATE jobs SET lease_until=? WHERE job_id=? AND worker_id=? AND status='running'",
                      (now() + lease_s, job_id, worker_id))
    return cur.rowcount == 1


def _append_history(con, job_id, attempt, worker, outcome, error):
    h = con.execute("SELECT history FROM jobs WHERE job_id=?", (job_id,)).fetchone()[0]
    hist = json.loads(h or "[]")
    hist.append({"attempt": attempt, "worker": worker, "outcome": outcome, "error": error, "at": round(now(), 1)})
    con.execute("UPDATE jobs SET history=? WHERE job_id=?", (json.dumps(hist), job_id))


def complete_in_tx(con, job_id: int, worker_id: str, status: str, stats: dict) -> bool:
    """Call inside the caller's open transaction, after its data writes. Returns False if this worker
    no longer owns the job (its lease expired and someone else took it): the caller must roll back."""
    owner = con.execute("SELECT worker_id, status, attempts FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    if owner is None or owner["worker_id"] != worker_id or owner["status"] != "running":
        return False
    cols = ", ".join(f"{k}=?" for k in stats)
    con.execute(f"UPDATE jobs SET status=?, finished_at=?, lease_until=NULL, error=NULL, {cols} WHERE job_id=?",
                (status, now(), *stats.values(), job_id))
    _append_history(con, job_id, owner["attempts"], worker_id, status, None)
    return True


def fail(con, job_id: int, worker_id: str, error: str, base_backoff_s: float = 1.0):
    con.execute("BEGIN IMMEDIATE")
    try:
        row = con.execute("SELECT attempts, max_attempts, worker_id FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None or row["worker_id"] != worker_id:
            con.execute("ROLLBACK")
            return
        _append_history(con, job_id, row["attempts"], worker_id, "failed", error[:300])
        if row["attempts"] >= row["max_attempts"]:
            con.execute("UPDATE jobs SET status='dead', error=?, finished_at=?, lease_until=NULL WHERE job_id=?",
                        (error[:500], now(), job_id))
        else:
            backoff = base_backoff_s * 2 ** (row["attempts"] - 1)
            con.execute("UPDATE jobs SET status='queued', error=?, not_before=?, lease_until=NULL, worker_id=NULL "
                        "WHERE job_id=?", (error[:500], now() + backoff, job_id))
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise


def runnable_or_running(con) -> int:
    return con.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
