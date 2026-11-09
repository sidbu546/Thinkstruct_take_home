"""Make bad inputs for the demo and enqueue them (SCALE.md §6.1):

  truncated_<batch>.json        a real batch file cut at 60% of its bytes -> salvage + quarantine, job 'partial'
  duplicates_and_bad.json       records copied from a batch that IS in the queue: one exact copy (counted as a
                                duplicate), one with a changed abstract (duplicate_conflict), plus a record with
                                no doc_number and a non-object record (both quarantined)
  missing.json (not created)    a file that does not exist -> fails every attempt -> 'dead' after 3

  python -m poc.faults --zip coding_challenge_large_data.zip --out poc_data/faults
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from . import queue
from .db import DEFAULT_DB, connect
from .enqueue import PIPELINE_VERSION, batch_members


def _key(path: Path) -> str:
    h = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"
    return hashlib.sha256(f"ingest|file|{path.name}|{h}|{PIPELINE_VERSION}".encode()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="poc.faults")
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("poc_data/faults"))
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--jobs", default="job_000")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    con = connect(Path(a.db))
    queued = [r["input_uri"].split("!", 1)[-1] for r in con.execute("SELECT input_uri FROM jobs WHERE "
                                                                   "input_uri LIKE 'zip://%'")]
    members = batch_members(a.zip, a.jobs.split(","))
    outside = [m for m in members if m.filename not in queued]
    inside = [m for m in members if m.filename in queued]
    if not outside or not inside:
        raise SystemExit("enqueue some (but not all) batch files of the job folder first")
    with zipfile.ZipFile(a.zip) as z:
        src = outside[-1]                                    # not otherwise queued: salvaged records are new
        data = z.read(src.filename)
        trunc = a.out / f"truncated_{Path(src.filename).name}"
        trunc.write_bytes(data[: int(len(data) * 0.6)])
        recs = json.loads(z.read(inside[0].filename))       # this batch IS queued
    exact, changed = dict(recs[0]), dict(recs[1])
    changed["abstract"] = (changed.get("abstract") or "") + " [conflicting revision of this record]"
    no_id = dict(recs[2])
    no_id["doc_number"] = ""
    bad = a.out / "duplicates_and_bad.json"
    bad.write_text(json.dumps([exact, changed, no_id, "this is not a patent object"]))
    missing = a.out / "missing.json"
    added = 0
    for p in (trunc, bad, missing):
        added += queue.enqueue(con, f"file://{p.resolve()}", _key(p))
    print(f"wrote {trunc.name} ({trunc.stat().st_size / 1e6:.1f} MB, cut at 60% of {src.filename}), "
          f"{bad.name} (exact + conflicting duplicate of {inside[0].filename}, no-id record, non-object); "
          f"enqueued {added} fault jobs (incl. a missing file)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
