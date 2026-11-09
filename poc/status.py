"""Status summary (SCALE.md §7), shared by the terminal view and the dashboard.

  python -m poc.status [--json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .db import DEFAULT_DB, connect, now

TARGET_PATENTS = 10_000_000
GPU_TOKENS_PER_S = 50_000        # SCALE.md §4.1 assumption for one L4 GPU (not measured here)


def summary(con) -> dict:
    q = lambda sql, *p: con.execute(sql, p).fetchall()  # noqa: E731
    one = lambda sql, *p: con.execute(sql, p).fetchone()[0]  # noqa: E731
    jobs = {r[0]: r[1] for r in q("SELECT status, COUNT(*) FROM jobs GROUP BY status")}
    n_jobs = sum(jobs.values())
    done = q("SELECT * FROM jobs WHERE status IN ('succeeded','partial')")
    attempts = one("SELECT COALESCE(SUM(attempts),0) FROM jobs")
    started = one("SELECT COUNT(*) FROM jobs WHERE attempts > 0")
    patents = one("SELECT COUNT(*) FROM patents")
    by_status = {r[0]: r[1] for r in q("SELECT data_status, COUNT(*) FROM patents GROUP BY 1")}
    issues = [{"type": r[0], "severity": r[1], "rows": r[2]} for r in
              q("SELECT issue_type, severity, COUNT(*) FROM data_issues GROUP BY 1, 2 ORDER BY 3 DESC")]
    missing_cpc = one("SELECT COUNT(*) FROM data_issues WHERE issue_type='missing_field' AND location='classification'")
    quarantine = [{"reason": r[0][:90], "rows": r[1]} for r in
                  q("SELECT reason, COUNT(*) FROM quarantine GROUP BY 1 ORDER BY 2 DESC")]
    conflicts = one("SELECT COUNT(*) FROM (SELECT doc_number FROM patent_versions GROUP BY 1 HAVING COUNT(*) > 1)")

    rec = sum(r["records_in"] or 0 for r in done)
    pc_s = sum((r["parse_s"] or 0) + (r["chunk_s"] or 0) for r in done)
    emb_s = sum(r["embed_s"] or 0 for r in done)
    emb_tok = sum(r["embed_tokens"] or 0 for r in done)
    emb_ch = sum(r["embed_chunks"] or 0 for r in done)
    chunks = sum(r["chunks"] or 0 for r in done)
    t_first = one("SELECT MIN(started_at) FROM jobs")
    t_last = one("SELECT MAX(finished_at) FROM jobs WHERE status IN ('succeeded','partial','dead')")
    wall = (t_last - t_first) if (t_first and t_last) else None
    per_worker = rec / pc_s if pc_s else None                    # patents per second of one worker (parse+chunk)
    chunks_per_patent = chunks / rec if rec else None
    tok_per_chunk = emb_tok / emb_ch if emb_ch else None
    tok_per_patent = (tok_per_chunk * chunks_per_patent) if (tok_per_chunk and chunks_per_patent) else None
    cpu_tok_s = emb_tok / emb_s if emb_s else None

    workers = []
    for r in q("SELECT * FROM workers ORDER BY worker_id"):
        w = dict(r) | {"heartbeat_age_s": round(now() - (r["last_heartbeat"] or now()), 1)}
        if w["state"] != "exited" and w["heartbeat_age_s"] > 2 * 30:      # 2 x lease: presumed dead
            w["state"] = f"unresponsive (was {w['state']}; presumed dead)"
        workers.append(w)
    active = sum(1 for w in workers if w["state"] in ("busy", "waiting", "idle") and w["heartbeat_age_s"] < 60)
    remaining = jobs.get("queued", 0) + jobs.get("running", 0)
    avg_job_s = sum(r["total_s"] or 0 for r in done) / len(done) if done else None
    eta = remaining * avg_job_s / max(active, 1) if (avg_job_s and remaining) else 0

    notable = [dict(job_id=r["job_id"], input=r["input_uri"].split("!")[-1].split("/")[-1], status=r["status"],
                    attempts=r["attempts"], error=r["error"], history=json.loads(r["history"] or "[]"),
                    quarantined=r["quarantined"])
               for r in q("SELECT * FROM jobs WHERE attempts > 1 OR status IN ('dead','partial') ORDER BY job_id")]
    extrap = None
    if per_worker:
        extrap = {"target_patents": TARGET_PATENTS,
                  "parse_chunk_patents_per_s_per_worker": round(per_worker, 1),
                  "parse_chunk_worker_hours": round(TARGET_PATENTS / per_worker / 3600, 1)}
        if tok_per_patent and cpu_tok_s:
            extrap |= {"chunks_per_patent": round(chunks_per_patent, 1), "tokens_per_patent": round(tok_per_patent),
                       "embed_cpu_tokens_per_s_per_worker": round(cpu_tok_s),
                       "embed_cpu_worker_hours": round(TARGET_PATENTS * tok_per_patent / cpu_tok_s / 3600),
                       "embed_gpu_hours_at_assumed_50k_tok_s": round(TARGET_PATENTS * tok_per_patent
                                                                     / GPU_TOKENS_PER_S / 3600)}
    return {
        "jobs": {"total": n_jobs, **jobs, "attempts": attempts, "retries": attempts - started},
        "progress": {"remaining_jobs": remaining, "eta_s": round(eta), "wall_s": round(wall, 1) if wall else None,
                     "patents_per_s_overall": round(rec / wall, 1) if wall else None, "active_workers": active},
        "patents": {"stored": patents, "records_processed": rec, "by_data_status": by_status,
                    "classification_missing_share": round(missing_cpc / patents, 3) if patents else None},
        "duplicates": {"exact_skipped": sum(r["duplicates_exact"] or 0 for r in done),
                       "conflicting": sum(r["duplicates_conflict"] or 0 for r in done),
                       "doc_numbers_with_multiple_versions": conflicts},
        "data_issues": issues, "quarantine": quarantine, "notable_jobs": notable, "workers": workers,
        "extrapolation": extrap,
    }


def render(s: dict) -> str:
    j, p, pa = s["jobs"], s["progress"], s["patents"]
    out = [f"Jobs: {j['total']} total · " + " · ".join(f"{k} {j[k]}" for k in ("succeeded", "partial", "running",
                                                                              "queued", "dead") if j.get(k))
           + f" · attempts {j['attempts']} (retries {j['retries']})",
           f"Progress: {p['remaining_jobs']} remaining, ETA {p['eta_s']} s, {p['active_workers']} active workers, "
           f"wall {p['wall_s']} s, {p['patents_per_s_overall']} patents/s overall",
           f"Patents: {pa['stored']} stored from {pa['records_processed']} records · data status {pa['by_data_status']}"
           f" · classification missing {pa['classification_missing_share']:.1%}" if pa["classification_missing_share"]
           is not None else "Patents: none yet",
           f"Duplicates: {s['duplicates']['exact_skipped']} exact (skipped), {s['duplicates']['conflicting']} "
           f"conflicting ({s['duplicates']['doc_numbers_with_multiple_versions']} doc numbers with several versions)",
           "Data issues (open, pending data research): " + ", ".join(f"{i['type']} {i['rows']}" for i in s["data_issues"]),
           "Quarantine: " + (", ".join(f"{q['rows']} × {q['reason']}" for q in s["quarantine"]) or "none")]
    for n in s["notable_jobs"]:
        hist = " → ".join(f"{h['outcome']}" for h in n["history"])
        out.append(f"  job {n['job_id']} {n['input']}: {n['status']} after {n['attempts']} attempt(s) [{hist}]"
                   + (f" · {n['quarantined']} quarantined" if n["quarantined"] else "")
                   + (f" · {n['error'][:80]}" if n["status"] == "dead" and n["error"] else ""))
    for w in s["workers"]:
        out.append(f"  worker {w['worker_id']}: {w['state']}, {w['jobs_done']} jobs, heartbeat {w['heartbeat_age_s']} s ago")
    e = s["extrapolation"]
    if e:
        out.append(f"Extrapolation to {e['target_patents']:,} patents: parse+chunk {e['parse_chunk_patents_per_s_per_worker']}"
                   f" patents/s per worker → {e['parse_chunk_worker_hours']:,} worker-hours")
        if "tokens_per_patent" in e:
            out.append(f"  embed: {e['tokens_per_patent']:,} tokens/patent ({e['chunks_per_patent']} chunks); CPU "
                       f"{e['embed_cpu_tokens_per_s_per_worker']:,} tok/s per worker → {e['embed_cpu_worker_hours']:,} "
                       f"worker-hours; GPU at an assumed 50k tok/s → {e['embed_gpu_hours_at_assumed_50k_tok_s']:,} GPU-hours")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="poc.status")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    s = summary(connect(Path(a.db)))
    print(json.dumps(s, indent=1, default=str) if a.json else render(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
