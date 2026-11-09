import json

import pytest

from patent_search.config import ROOT
from poc import queue
from poc.db import connect
from poc.process import parse_tolerant, process_job
from poc.status import summary

DATA = ROOT / "data" / "patent_data_small"


@pytest.fixture
def con(tmp_path):
    return connect(tmp_path / "poc.db")


@pytest.fixture(scope="module")
def records():
    if not DATA.exists():
        pytest.skip("small dataset not extracted")
    return json.loads(sorted(DATA.glob("*.json"))[0].read_text())[:6]


def _job(con, path, key):
    queue.enqueue(con, f"file://{path}", key)


def test_enqueue_is_idempotent(con):
    assert queue.enqueue(con, "file:///x", "k1") and not queue.enqueue(con, "file:///x", "k1")
    assert con.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1


def test_claim_lease_expiry_and_reclaim(con):
    queue.enqueue(con, "file:///x", "k1")
    a = queue.claim(con, "wA")
    assert a["status"] == "running" and a["attempts"] == 1
    assert queue.claim(con, "wB") is None                       # leased: nobody else gets it
    con.execute("UPDATE jobs SET lease_until = 0")              # wA died: lease expires
    b = queue.claim(con, "wB")
    assert b["job_id"] == a["job_id"] and b["worker_id"] == "wB" and b["attempts"] == 2
    hist = json.loads(con.execute("SELECT history FROM jobs").fetchone()[0])
    assert hist[-1]["outcome"].startswith("lease expired") and hist[-1]["worker"] == "wA"


def test_fail_backoff_then_dead(con):
    queue.enqueue(con, "file:///x", "k1", max_attempts=2)
    j = queue.claim(con, "w")
    queue.fail(con, j["job_id"], "w", "boom")
    row = con.execute("SELECT status, not_before FROM jobs").fetchone()
    assert row["status"] == "queued" and row["not_before"] > 0
    assert queue.claim(con, "w") is None                        # still in backoff
    con.execute("UPDATE jobs SET not_before = 0")
    j = queue.claim(con, "w")
    queue.fail(con, j["job_id"], "w", "boom again")
    assert con.execute("SELECT status FROM jobs").fetchone()[0] == "dead"


def test_parse_tolerant_salvages_truncated_json(records):
    data = json.dumps(records).encode()
    recs, quar, salvaged = parse_tolerant(data[: int(len(data) * 0.7)])
    assert salvaged and 0 < len(recs) < len(records) and quar
    assert recs == records[:len(recs)]
    ok, q2, s2 = parse_tolerant(data)
    assert ok == records and not q2 and not s2


def test_process_is_idempotent_and_dedupes(con, tmp_path, records):
    a = tmp_path / "a.json"
    a.write_text(json.dumps(records[:4]))
    _job(con, a, "ka")
    r1 = process_job(con, queue.claim(con, "w"), "w")
    assert r1["outcome"] == "succeeded" and r1["patents_written"] == 4
    # re-run the same job (e.g. after a lost lease): same rows, no duplicates counted
    con.execute("UPDATE jobs SET status='queued'")
    r2 = process_job(con, queue.claim(con, "w"), "w")
    assert r2["patents_written"] == 4 and r2["duplicates_exact"] == 0
    assert con.execute("SELECT COUNT(*) FROM patents").fetchone()[0] == 4
    # another file: exact copy, conflicting copy, a record without id, a non-object
    changed = dict(records[1])
    changed["abstract"] = changed["abstract"] + " CHANGED"
    no_id = dict(records[2])
    no_id["doc_number"] = ""
    b = tmp_path / "b.json"
    b.write_text(json.dumps([records[0], changed, no_id, "junk"]))
    _job(con, b, "kb")
    r3 = process_job(con, queue.claim(con, "w"), "w")
    assert r3["outcome"] == "partial" and r3["duplicates_exact"] == 1 and r3["duplicates_conflict"] == 1
    assert r3["quarantined"] == 2
    assert con.execute("SELECT COUNT(*) FROM data_issues WHERE issue_type='duplicate_conflict'").fetchone()[0] == 1
    assert con.execute("SELECT COUNT(*) FROM patents").fetchone()[0] == 4
    s = summary(con)
    assert s["jobs"]["succeeded"] == 1 and s["jobs"]["partial"] == 1 and s["duplicates"]["conflicting"] == 1


def test_lost_lease_writes_nothing(con, tmp_path, records):
    a = tmp_path / "a.json"
    a.write_text(json.dumps(records[:2]))
    _job(con, a, "ka")
    job = queue.claim(con, "wA")
    con.execute("UPDATE jobs SET worker_id='wB'")               # someone else took it over
    assert process_job(con, job, "wA")["outcome"] == "lease_lost"
    assert con.execute("SELECT COUNT(*) FROM patents").fetchone()[0] == 0
