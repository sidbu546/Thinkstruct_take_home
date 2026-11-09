"""Worker processes (SCALE.md §10): N independent local processes stand in for ECS containers.

  python -m poc.worker --workers 4 --fail-rate 0.05 --embed-sample 3 [--kill-one-after 40]

Each worker loops: claim a job (lease) -> process it while a heartbeat thread renews the lease ->
commit data + "done" in one transaction, or report a failure (retry with backoff, dead after 3).
--kill-one-after S SIGKILLs worker 0 after S seconds, mid-job, to show that its job is re-claimed by
another worker once the lease expires, with no duplicate rows."""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import signal
import threading
import time
import traceback
from pathlib import Path

from . import queue
from .db import DEFAULT_DB, connect, now
from .process import process_job


def _heartbeat(db_path, job_id, worker_id, stop: threading.Event):
    con = connect(db_path)
    while not stop.wait(queue.LEASE_S / 3):
        queue.renew(con, job_id, worker_id)
        con.execute("UPDATE workers SET last_heartbeat=? WHERE worker_id=?", (now(), worker_id))
    con.close()


def worker_main(idx: int, db_path: str, fail_rate: float, embed_sample: int, seed: int):
    worker_id = f"w{idx}-{os.getpid()}"
    con = connect(Path(db_path))
    con.execute("INSERT OR REPLACE INTO workers VALUES (?,?,?,?,NULL,0,'starting')",
                (worker_id, os.getpid(), now(), now()))
    encoder = None
    if embed_sample:
        import torch
        torch.set_num_threads(2)                     # 4 workers x 2 threads on a 10-core laptop
        from patent_search.config import get_config
        from patent_search.embed.encoder import get_encoder
        encoder = get_encoder(get_config(device="cpu"))
    con.execute("UPDATE workers SET state='idle', last_heartbeat=? WHERE worker_id=?", (now(), worker_id))
    while True:
        job = queue.claim(con, worker_id)
        if job is None:
            if queue.runnable_or_running(con) == 0:
                break                                 # nothing queued and nothing in flight: done
            con.execute("UPDATE workers SET state='waiting', last_heartbeat=?, current_job=NULL WHERE worker_id=?",
                        (now(), worker_id))
            time.sleep(1.0)                           # jobs in backoff, or leases held by other workers
            continue
        con.execute("UPDATE workers SET state='busy', current_job=?, last_heartbeat=? WHERE worker_id=?",
                    (job["job_id"], now(), worker_id))
        stop = threading.Event()
        hb = threading.Thread(target=_heartbeat, args=(Path(db_path), job["job_id"], worker_id, stop), daemon=True)
        hb.start()
        try:
            res = process_job(con, job, worker_id, fail_rate, embed_sample, encoder, seed)
            if res["outcome"] != "lease_lost":
                con.execute("UPDATE workers SET jobs_done=jobs_done+1 WHERE worker_id=?", (worker_id,))
        except Exception as e:                        # transient or permanent: the queue decides
            queue.fail(con, job["job_id"], worker_id, f"{type(e).__name__}: {e}")
            if not isinstance(e, (OSError, ValueError, RuntimeError)):
                traceback.print_exc()
        finally:
            stop.set()
            hb.join(timeout=5)
    con.execute("UPDATE workers SET state='exited', current_job=NULL, last_heartbeat=? WHERE worker_id=?",
                (now(), worker_id))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="poc.worker")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fail-rate", type=float, default=0.0, help="probability a job attempt fails (injected)")
    ap.add_argument("--embed-sample", type=int, default=0,
                    help="patents per job whose chunks are embedded on CPU, to measure embed throughput")
    ap.add_argument("--kill-one-after", type=float, help="SIGKILL worker 0 after this many seconds")
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args(argv)
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker_main, args=(i, a.db, a.fail_rate, a.embed_sample, a.seed), daemon=False)
             for i in range(a.workers)]
    t0 = time.time()
    for p in procs:
        p.start()
    print(f"started {len(procs)} workers: {[p.pid for p in procs]}", flush=True)
    killed = False
    while any(p.is_alive() for p in procs):
        if a.kill_one_after and not killed and time.time() - t0 > a.kill_one_after and procs[0].is_alive():
            os.kill(procs[0].pid, signal.SIGKILL)
            killed = True
            print(f"[{time.time() - t0:.0f}s] SIGKILLed worker 0 (pid {procs[0].pid}) mid-run", flush=True)
        time.sleep(1)
    print(f"all workers exited after {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
