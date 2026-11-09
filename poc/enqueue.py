"""Enqueue one job per batch file of the large dataset, straight from the zip (no extraction).

  python -m poc.enqueue --zip coding_challenge_large_data.zip --jobs job_000 --limit 20

The idempotency key is hash(stage, member name, CRC32, size, pipeline version): running it again
inserts nothing new, and a changed file or a new pipeline version gets a new job."""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from patent_search.config import get_config

from . import queue
from .db import DEFAULT_DB, connect
from .process import idempotency_key

PIPELINE_VERSION = "poc-ingest-1/" + get_config().hash()


def batch_members(zip_path: Path, jobs: list[str]) -> list[zipfile.ZipInfo]:
    with zipfile.ZipFile(zip_path) as z:
        out = [i for i in z.infolist() if i.filename.endswith(".json") and "/patents_batch_" in i.filename
               and any(f"/{j}/" in i.filename for j in jobs)]
    return sorted(out, key=lambda i: i.filename)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="poc.enqueue")
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--jobs", default="job_000", help="comma-separated job folders inside the zip")
    ap.add_argument("--limit", type=int, help="first N batch files (sorted)")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    a = ap.parse_args(argv)
    con = connect(Path(a.db))
    members = batch_members(a.zip, a.jobs.split(","))[:a.limit]
    zabs = a.zip.resolve()
    new = sum(queue.enqueue(con, f"zip://{zabs}!{m.filename}",
                            idempotency_key(m.filename, m.CRC, m.file_size, PIPELINE_VERSION)) for m in members)
    total = con.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    print(f"{len(members)} batch files considered, {new} new jobs enqueued, {len(members) - new} already "
          f"queued (idempotent); {total} jobs in the queue")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
