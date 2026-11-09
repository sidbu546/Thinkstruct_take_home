"""Live status dashboard (SCALE.md §7, §10).  python -m poc.dashboard  ->  http://localhost:8001

One page polling /api/status every 2 s; the same summary as `python -m poc.status`."""
from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from .db import DEFAULT_DB, connect
from .status import summary

PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ingestion Status</title><style>
:root{--bg:#f7f7f5;--card:#fff;--fg:#1d1d1b;--mute:#6b6b66;--line:#e3e3de;--ok:#2f7d4f;--warn:#b26b00;--bad:#b3261e}
@media (prefers-color-scheme: dark){:root{--bg:#161615;--card:#1f1f1d;--fg:#ececea;--mute:#a3a39e;--line:#33332f;--ok:#5cbf83;--warn:#e0a040;--bad:#ef6b62}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:20px 16px}h1{font-size:20px;margin:0 0 4px}.sub{color:var(--mute);margin-bottom:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin-bottom:14px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.tile b{display:block;font-size:22px;font-variant-numeric:tabular-nums}.tile span{color:var(--mute);font-size:12px}
section{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:12px;overflow-x:auto}
h2{font-size:14px;margin:0 0 8px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
td,th{text-align:left;padding:4px 8px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--mute);font-weight:500}
.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}.bar{height:8px;background:var(--line);border-radius:4px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--ok)}
</style></head><body><main>
<h1>Ingestion status</h1><div class="sub" id="sub">loading…</div>
<div class="bar"><i id="bar" style="width:0"></i></div><br>
<div class="grid" id="tiles"></div>
<section><h2>Extrapolation to 10M patents (measured here, scaled)</h2><div id="extrap"></div></section>
<section><h2>Workers</h2><table id="workers"></table></section>
<section><h2>Jobs with retries, salvage or failure</h2><table id="jobs"></table></section>
<section><h2>Data issues: open, pending data research (never dropped)</h2><table id="issues"></table></section>
<section><h2>Quarantine</h2><table id="quar"></table></section>
</main><script>
const esc=s=>String(s??"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const tile=(v,l,c="")=>`<div class="tile"><b class="${c}">${v}</b><span>${l}</span></div>`;
async function tick(){
 try{const s=await (await fetch("/api/status")).json();const j=s.jobs,p=s.progress,pa=s.patents,d=s.duplicates;
 const fin=(j.succeeded||0)+(j.partial||0)+(j.dead||0);document.getElementById("bar").style.width=(j.total?100*fin/j.total:0)+"%";
 document.getElementById("sub").textContent=`${fin} of ${j.total} jobs finished · ${p.remaining_jobs} remaining · ETA ${p.eta_s}s · ${p.active_workers} active workers · updated ${new Date().toLocaleTimeString()}`;
 document.getElementById("tiles").innerHTML=[tile(j.succeeded||0,"succeeded","ok"),tile(j.partial||0,"partial (salvaged)","warn"),
  tile(j.running||0,"running"),tile(j.queued||0,"queued"),tile(j.dead||0,"dead","bad"),tile(j.retries,"retries"),
  tile(pa.stored.toLocaleString(),"patents stored"),tile(p.patents_per_s_overall??"–","patents/s overall"),
  tile(d.exact_skipped,"exact duplicates skipped"),tile(d.conflicting,"conflicting duplicates","warn"),
  tile(pa.classification_missing_share!=null?(100*pa.classification_missing_share).toFixed(1)+"%":"–","classification missing","warn")].join("");
 const e=s.extrapolation;document.getElementById("extrap").innerHTML=!e?"waiting for finished jobs…":
  `<table><tr><td>Parse + validate + chunk</td><td>${e.parse_chunk_patents_per_s_per_worker} patents/s per worker</td><td><b>${e.parse_chunk_worker_hours.toLocaleString()}</b> worker-hours for 10M</td></tr>`+
  (e.tokens_per_patent?`<tr><td>Embed (CPU, measured on a sample)</td><td>${e.embed_cpu_tokens_per_s_per_worker.toLocaleString()} tokens/s per worker · ${e.tokens_per_patent.toLocaleString()} tokens/patent</td><td><b>${e.embed_cpu_worker_hours.toLocaleString()}</b> CPU worker-hours</td></tr>`+
  `<tr><td>Embed (GPU, assumed 50k tokens/s per L4)</td><td>same tokens/patent</td><td><b>${e.embed_gpu_hours_at_assumed_50k_tok_s.toLocaleString()}</b> GPU-hours</td></tr>`:"")+`</table>`;
 document.getElementById("workers").innerHTML="<tr><th>worker</th><th>state</th><th>jobs done</th><th>job</th><th>heartbeat</th></tr>"+
  s.workers.map(w=>`<tr><td>${esc(w.worker_id)}</td><td class="${w.state.startsWith("unresponsive")?"bad":""}">${esc(w.state)}</td><td>${w.jobs_done}</td><td>${w.current_job??""}</td><td>${w.heartbeat_age_s}s ago</td></tr>`).join("");
 document.getElementById("jobs").innerHTML="<tr><th>job</th><th>input</th><th>status</th><th>attempts</th><th>history</th></tr>"+
  s.notable_jobs.map(n=>`<tr><td>${n.job_id}</td><td>${esc(n.input)}</td><td class="${n.status==="dead"?"bad":n.status==="partial"?"warn":"ok"}">${n.status}${n.quarantined?` (${n.quarantined} quarantined)`:""}</td><td>${n.attempts}</td><td>${n.history.map(h=>esc(h.outcome)+(h.error?": "+esc(h.error.slice(0,70)):"")).join(" → ")}</td></tr>`).join("");
 document.getElementById("issues").innerHTML="<tr><th>issue</th><th>severity</th><th>rows</th></tr>"+s.data_issues.map(i=>`<tr><td>${esc(i.type)}</td><td>${i.severity}</td><td>${i.rows.toLocaleString()}</td></tr>`).join("");
 document.getElementById("quar").innerHTML="<tr><th>reason</th><th>rows</th></tr>"+(s.quarantine.map(q=>`<tr><td>${esc(q.reason)}</td><td>${q.rows}</td></tr>`).join("")||"<tr><td>none</td><td></td></tr>");
 }catch(err){document.getElementById("sub").textContent="status unavailable: "+err}
}
tick();setInterval(tick,2000);
</script></body></html>"""


def make_app(db_path: Path) -> FastAPI:
    app = FastAPI(title="PoC ingestion status")

    @app.get("/", response_class=HTMLResponse)
    def page():
        return PAGE

    @app.get("/api/status")
    def status():
        con = connect(db_path)
        try:
            return summary(con)
        finally:
            con.close()

    return app


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="poc.dashboard")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--port", type=int, default=8001)
    a = ap.parse_args(argv)
    import socket
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", a.port)) == 0:
            print(f"Port {a.port} is already in use: a dashboard is probably already running at "
                  f"http://localhost:{a.port}. Open that in a browser, or stop the other process first.")
            return 1
    import uvicorn
    print(f"Dashboard running at http://localhost:{a.port}  (open it in a browser; it refreshes every 2 s).\n"
          f"This command keeps running while the dashboard is up. Press Ctrl+C to stop it.", flush=True)
    uvicorn.run(make_app(Path(a.db)), host="127.0.0.1", port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
