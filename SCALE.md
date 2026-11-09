# Running the Patent Search Engine at 10 Million Patents

This document describes how the search engine in [DESIGN.md](DESIGN.md) would run over the full
patent database. It uses the terms defined there: adaptive chunks, data issues, phase 1 and phase 2,
and the JSON response.

**Goal:** the same engine (hybrid search with filters, adaptive chunks, element coverage, optional
reranking, data-gap notices) over **10 million patents**, updated weekly, at interactive speed.

**Approach:** widely used managed services, one source of truth, batch jobs that are safe to re-run,
and versioned indexes. Section 9 lists the problems this design leaves open. Section 10 describes a
proof-of-concept that ran on real data; the sizes and throughput used throughout come from it.

---

## 1. Sizing: what 10M patents means

Measured on an 8,000-patent sample of the large dataset (one batch from each of the 8 jobs):

| Per patent | Measured | × 10M |
|---|---|---|
| Raw JSON | 69 KB | **~690 GB** raw |
| Words (title + abstract + claims + description) | mean 10.9k, median 8.0k | ~110B words ≈ **140B tokens** |
| Claims | mean 19.2 (≈ 1,000 words) | ~190M claims |
| Non-empty description paragraphs | mean 114 | ~1.1B paragraphs |
| Chunks after adaptive chunking (measured by the PoC on 20,571 records, §10) | **~96** | **~960M chunks** |
| Tokens to embed, model tokenizer (text + claim-chain context duplication; measured, §10) | **~19.4k** | **~194B tokens** |

Data-quality rates seen in the same sample, which the pipeline must handle:

| Issue | Rate in sample | At 10M |
|---|---|---|
| `classification = "N/A"` | **37%** | ~3.7M patents that no CPC filter can match until researched |
| Exact duplicate records | 4 in 8,000 (README warns of more) | ≥ 5k |
| Empty detailed description | 0.8% | ~80k |
| No date in filename (`US20200187828A1.XML`) | 100% of the large set | filing/publication date must come from another source (§4.5) |
| Corrupted entries / files | 0 in the sample; README says they exist | must be survivable (§6) |

**Load assumptions** (to be revised once real usage is known): ~200 concurrent
examiner users, **10 queries/s peak**, p95 latency target **< 1.5 s** with reranking, and
**≤ 200k new or changed patents per week**.

---

## 2. Architecture

```
                         ┌──────────────────────── INGESTION (batch) ─────────────────────────┐
 USPTO bulk XML / JSON   │                                                                     │
 weekly drops  ─────────►│  S3 raw/ (immutable) ─► queue ─► parse/validate workers (CPU)        │
                         │                                   │   ├─► Postgres (metadata, claims, │
                         │                                   │   │    data_issues, jobs)         │
                         │                                   │   └─► S3 normalized/ (Parquet)    │
                         │                         queue ─► chunk workers (CPU) ─► S3 chunks/    │
                         │                         queue ─► embed workers (GPU) ─► S3 vectors/   │
                         │                         index builder ─► OpenSearch (new version)     │
                         │                                       ─► chunk-vector store (new ver) │
                         │                         validation gate (counts + eval) ─► alias swap │
                         └─────────────────────────────────────────────────────────────────────┘

                         ┌──────────────────────── SERVING (online) ──────────────────────────┐
 user / UI / CLI ──────► │  Search API (stateless, autoscaled CPU)                               │
                         │    1. filters + BM25 + patent-level kNN ──► OpenSearch  → 1,000 docs  │
                         │    2. fetch chunk vectors for those docs ─► chunk-vector store        │
                         │       exact chunk scoring + RRF ─────────────────────────→ 100 docs   │
                         │    3. rerank ───────────────────────────► Reranker service (GPU)      │
                         │       + element coverage + data notices ─────────────────→ 30 docs    │
                         │    4. JSON response (schema from DESIGN.md §6)                        │
                         └─────────────────────────────────────────────────────────────────────┘

 Ops: metrics + logs + traces → dashboards / alerts; status dashboard reads the jobs table.
```

### 2.1 Components

| Component | Choice | Holds / does | Why this one |
|---|---|---|---|
| Object storage | **S3** | raw files (immutable), normalized Parquet, chunks, all vectors (archive) | cheap, durable, the replay source for everything |
| Relational DB | **Postgres** (RDS) | patents metadata, claims, `data_issues`, `jobs`, `index_versions`, `saved_results` | source of truth for state; transactions; easy to query for status |
| Queue | **SQS** (one queue per stage + dead-letter queues) | work items = one input file or one shard of patents | managed, at-least-once, visibility timeout = lease |
| Batch workers | containers on **ECS/Fargate** (CPU) and **ECS on EC2 GPU** (embedding) | parse/validate, chunk, embed, index build | same image as Part 1's code; scale by queue depth |
| Search engine | **OpenSearch** (managed) | BM25 over title/abstract/claims, keyword/CPC/date filters, **patent-level vectors** (kNN, faiss HNSW) | one engine does lexical + filters + vectors; supports filtered kNN |
| Chunk-vector store | **LanceDB on local NVMe** of the search API nodes (synced from S3), keyed by `doc_number` | ~960M chunk vectors, fetched **by document**, not searched globally | avoids a ~1B-vector ANN; fetch-by-key is fast and cheap |
| Reranker service (optional) | small GPU service (L4), `bge-reranker-base` | phase 2: top 100 patents × 2 chunks ≈ 200 pairs → top 30 | opt-in, as in Part 3 (small measured gain); an L4 is expected to be far faster than the ~7 s per query measured on the Mac GPU, but that is not measured |
| Search API | FastAPI, stateless | orchestration, JSON contract | same code as Part 1 |
| Observability | CloudWatch or Prometheus + Grafana, structured logs, OpenTelemetry traces | pipeline and query health | see §7 |

### 2.2 The key scaling decision: coarse-to-fine vectors

~960M chunk vectors in one global ANN index would need ~1–2 TB of RAM even with compression, and it
would be slow to rebuild. Instead the vector search is split in two tiers, the same idea as Part 1's
"filter first, then score only the allowed set":

| Tier | What | Count | Stored as | Used for |
|---|---|---|---|---|
| **Patent-level (hot)** | abstract vector + one vector per independent claim (≈ 3/patent) | **~40M** | 256-dim (Matryoshka-truncated) fp16 in OpenSearch HNSW: 40M × 512 B ≈ 20 GB + graph ≈ **~25 GB RAM** across the cluster | find the top 1,000 candidate patents, with filters applied inside the kNN |
| **Chunk-level (warm)** | every adaptive chunk (claims, elements, passages) | **~960M** | 256-dim int8 = 256 B each ≈ **~246 GB** on NVMe (LanceDB), plus a 768-dim fp16 archive in S3 (~1.5 TB) for re-quantization | exact scoring of the ~96k chunks belonging to the 1,000 candidates (≈ 25 MB read, tens of ms) |

256-dim vectors require training the bi-encoder with `MatryoshkaLoss`, which makes the first 256
dimensions usable on their own. The current model is 768-dim and was not trained this way (DESIGN.md
Part 3C), so this is a step to take before scaling. The evaluation from DESIGN.md Part 3A must then
confirm that 256 dimensions lose little recall; if they lose too much, use 384 or 512 and recompute the
RAM and cost numbers.

---

## 3. Data model at scale

Postgres keeps the Part 1 tables (`patents`, `claims`, `data_issues`) with three additions on
every row: `snapshot_id`, `source_uri`, `source_sha256`. Long text (descriptions, chunk text) lives in
S3 Parquet, not Postgres: about 200 GB compressed, versus roughly 80 GB for abstracts and claims in
Postgres.

New tables:

```sql
CREATE TABLE jobs (
  job_id          BIGSERIAL PRIMARY KEY,
  pipeline        TEXT,          -- backfill | weekly | research | reembed
  stage           TEXT,          -- parse | chunk | embed | index
  input_uri       TEXT,
  input_sha256    TEXT,
  pipeline_version TEXT,         -- git sha of the code
  idempotency_key TEXT UNIQUE,   -- hash(stage, input_sha256, pipeline_version, model_revision)
  status          TEXT,          -- queued | running | succeeded | partial | failed | dead
  attempts        INT DEFAULT 0,
  worker_id       TEXT,
  lease_until     TIMESTAMPTZ,
  started_at      TIMESTAMPTZ,
  finished_at     TIMESTAMPTZ,
  records_in      INT, records_out INT, records_quarantined INT, issues_opened INT,
  error           TEXT
);

CREATE TABLE quarantine (                       -- records that could not be parsed or validated
  id BIGSERIAL PRIMARY KEY, job_id BIGINT, input_uri TEXT, byte_offset BIGINT,
  reason TEXT, raw_excerpt TEXT, status TEXT    -- open | fixed | confirmed_bad
);

CREATE TABLE patent_versions (                  -- duplicates with different content
  doc_number TEXT, content_sha256 TEXT, source_uri TEXT, seen_at TIMESTAMPTZ,
  is_canonical BOOLEAN, PRIMARY KEY (doc_number, content_sha256)
);

CREATE TABLE index_versions (
  version_id TEXT PRIMARY KEY,                  -- e.g. 2026-11-08_gte-ft-v2_chunk-v3
  snapshot_id TEXT, model TEXT, model_revision TEXT, chunking_version TEXT,
  n_patents BIGINT, n_patent_vectors BIGINT, n_chunks BIGINT,
  eval_metrics JSONB,                           -- Part 3 metrics on the fixed eval set
  status TEXT,                                  -- building | validating | live | retired | failed
  created_at TIMESTAMPTZ, promoted_at TIMESTAMPTZ
);
```

**Snapshots.** Every ingestion run writes under a new `snapshot_id`. Indexes are only built from a
**closed** snapshot, so OpenSearch, the chunk-vector store and Postgres always describe the same data.

---

## 4. Pipelines

All stages are **idempotent**: re-running a job with the same `idempotency_key` produces the same
output and overwrites it. Chunk ids are deterministic (`doc_number`, kind, claim or paragraph position,
content hash), and writes are upserts. This is what makes retries safe.

### 4.1 Backfill (one-time, 10M patents)

1. **Land:** copy source files to `s3://…/raw/{source}/{date}/` once. They are never modified.
2. **Enqueue:** one `parse` job per raw file (the large dataset's 1,000-patent batch files are a natural unit).
3. **Parse + validate** (CPU): tolerant streaming JSON parse, schema validation, normalization, claim parsing, data-issue detection, dedupe (§6.3). Writes Postgres rows plus `normalized/*.parquet`. Then enqueues `chunk` jobs per 1,000 patents.
4. **Chunk** (CPU): adaptive chunking (DESIGN.md §3.3), the same code as the engine. Passage merge decisions use TF-IDF similarity between neighbouring paragraphs, which is cheap and needs no embeddings. Writes `chunks/*.parquet` and enqueues `embed` jobs.
5. **Embed** (GPU): batch by token length (sort, then bucket) to avoid padding waste. Writes 768-dim fp16 to `vectors/full/` and 256-dim int8 / fp16 to `vectors/serving/`.
6. **Index build:** bulk-load OpenSearch into a **new index** (`patents_v{version}`) and build the LanceDB table for the same version.
7. **Validate:** count reconciliation (Postgres = OpenSearch = vector store, per snapshot), the Part 3 eval set (metrics must not drop beyond tolerance compared with the live version), and 20 smoke queries.
8. **Promote:** atomically move the OpenSearch alias `patents_live` and the API's config pointer to the new version. The old version is kept for 2 weeks for rollback.

**Throughput math (backfill):**

| Stage | Rate | Time for 10M |
|---|---|---|
| Parse + validate + chunk | **~19 patents/s per worker process (measured, §10)**; adaptive chunking is ~99% of it | 526k worker-s ≈ **~147 worker-hours** → ~2.3 h on 64 vCPUs |
| Embed (149M model, fp16, L4) | ~50k tokens/s per GPU (**assumed**; the PoC measured only CPU: ~655 tokens/s per 2-thread worker) | 194B / 50k ≈ 3.9M s ≈ **~1,075 GPU-hours** → ~2.2 days on 20 GPUs |
| OpenSearch bulk load | ~5–10k docs/s | ~20–30 min, plus HNSW build for 40M vectors (~1–3 h) |

**Embedding is the bottleneck**, by roughly 15×. Everything else is sized around it.

### 4.2 Weekly incremental

Triggered when the weekly USPTO drop lands (scheduler: EventBridge cron).

- Same stages 1–5, only for new or changed documents. A changed document is one whose `content_sha256` differs from the canonical version.
- **Indexing without a full rebuild:** upsert into the live OpenSearch index (it supports document updates and HNSW inserts) and append to the live LanceDB table. Deleted or withdrawn documents get `status = withdrawn` and are filtered out.
- At ~200k patents/week, that's ~3.9B tokens ≈ 22 GPU-hours, so about 1 hour on 20 spot GPUs.
- After a week's upserts, the eval set and the reconciliation counts run again. A failure pages the on-call; it doesn't roll back automatically.

### 4.3 Data research (closing data gaps)

This is the data-research workflow of DESIGN.md §3.2, at scale.

- A nightly job groups open `data_issues` by type and source. For each fixable type it runs a fetcher: CPC from the USPTO master classification file for the 37% `N/A`; publication and filing dates from the full-text XML; lost claim preambles and thin descriptions from the full-text XML.
- Each fixed patent goes back through stages 3–5 **for that patent only**. Its `record_version` goes up, its issues are marked `resolved`, and `saved_results` that referenced the old version are marked **"record updated — re-examine"**.
- Until an issue is resolved, every search output that touches the record shows the notice (DESIGN.md §6.3). At scale, the response also states when the *filter itself* can't see records: e.g. a CPC filter adds `"data_notice": "3.7M patents have no classification yet (pending data research) and cannot match this CPC filter. Results may be incomplete; re-run after the update."`

### 4.4 Re-embedding (new model or chunking version)

- This is a full build: stages 4–8 into a **new index version**, in parallel with the live one (blue/green). Raw and normalized data are reused; parsing is not repeated.
- Cost ≈ one backfill's embedding cost (§5). Do it when the eval gain justifies it, not on every model tweak.
- The promotion gate (§4.1 step 7) compares the new version's eval metrics with the live version's.

### 4.5 Date fields at scale

The large dataset's filenames carry no date, unlike the small set (`…A1-20240215.XML`). The pipeline
therefore takes `publication_date` / `filing_date` from the XML (`<publication-reference>`,
`<application-reference>`). If those are absent, it opens a data issue `date_missing` and derives the
year only from the doc number. The week-range filter and the week facet then say so in their notice.

### 4.6 Query pipeline (online)

| Step | Where | Output | Budget (p95) |
|---|---|---|---|
| 0. Embed query (and its elements, for complex claims) | API node (CPU, model loaded) or GPU reranker host | 256-dim vector(s) | 50 ms |
| 1. Filters + BM25 + patent-level kNN, fused | OpenSearch, one hybrid query with filters applied inside kNN | top 1,000 patents | 250 ms |
| 2. Fetch chunk vectors for those 1,000 patents, exact scoring, roll-up, RRF | API node + local NVMe LanceDB | top 100 patents, 2 evidence chunks each | 150 ms |
| 3. Rerank 100 × 2 pairs (opt-in) | Reranker service (GPU) | top 30 | 300 ms (L4 target, not measured) |
| 4. Coverage for the top 10, data notices, facets | API + Postgres (issues by doc) | JSON | 150 ms |
| **Total** | | | **~0.9 s p95** (target < 1.5 s) |

Facets (CPC and date counts) come from OpenSearch aggregations over the **filtered** set in step 1,
so they count the whole result space, not just the top 30.

---

## 5. Cost (rough, AWS us-east-1 on-demand list prices; verify before budgeting)

### 5.1 One-time backfill

| Item | Quantity | Unit price (≈) | Cost (≈) |
|---|---|---|---|
| Embedding GPUs (g6.xlarge, L4) | ~1,075 GPU-h | $0.80/h on-demand (~$0.35 spot) | **$860** (≈ $380 on spot) |
| Parse/chunk CPU (Fargate) | ~150 vCPU-h + memory (measured rate) | ~$0.05/vCPU-h all-in | < $10 |
| OpenSearch bulk build | runs on the steady-state cluster | — | (in 5.2) |
| S3 PUT/GET requests | ~millions | — | < $20 |
| **Total one-time** | | | **≈ $400–900** |

### 5.2 Steady state, per month

| Item | Sizing | Cost / month (≈) |
|---|---|---|
| OpenSearch | 3 data nodes r6g.2xlarge.search (64 GB RAM each; ~25 GB vectors + BM25 over title/abstract/claims ~150 GB on EBS) + 3 small master nodes + 3 × 500 GB gp3 | **$2,000** |
| Postgres (RDS) | db.r6g.xlarge + 300 GB gp3, single availability zone in the first version | $400 |
| Search API | 2 × c7g.2xlarge with 1 TB local NVMe each (i4g or similar, to hold the 246 GB chunk-vector table) | $600 |
| Reranker | 1 × g6.xlarge (L4) always on; a 2nd at peak via autoscaling | $600–1,200 |
| S3 | raw 690 GB + Parquet ~400 GB + vectors ~1.8 TB ≈ 2.9 TB | $65 |
| Weekly incremental embedding | ~22 GPU-h/week on spot | $35 |
| Observability, NAT, data transfer | — | $200 |
| **Total** | | **≈ $3,900–4,500 / month** |

**Cheaper variant** (if about $4k per month is too much for a first version): drop the reranker GPU (Part 3 found reranking
adds no significant gain on its test set), run OpenSearch on 2 nodes, and use spot for all batch
work. That's ≈ $2,000/month, at the cost of rerank quality and no high availability.

**Biggest cost drivers:** OpenSearch RAM (patent vectors plus BM25), the always-on GPU, and a full
re-embed (~$300–700 each time). The vector-tier split (§2.2) is what keeps the first one from being
10× larger.

---

## 6. Error handling

### 6.1 Failure classes

| Failure | Detection | Action | Visible where |
|---|---|---|---|
| Corrupted file (truncated or invalid JSON) | parser exception | stream-parse with `ijson`, **salvage** every valid record, quarantine the rest with byte offsets; job ends `partial` | `jobs.status = partial`, `quarantine` rows, dashboard |
| Bad record (schema fails, wrong types) | Pydantic validation | record to `quarantine`; the batch continues | dashboard, `quarantine` |
| Defective content (missing fields, lost preamble, `N/A` CPC, thin description…) | DESIGN.md §3.2 rules | record **kept and indexed**, `data_issues` row, `data_status` set | every search output that touches it (§5.3 of DESIGN.md) |
| Exact duplicate | same `doc_number` + same `content_sha256` | skip, count | job counters |
| Conflicting duplicate | same `doc_number`, different content | keep both in `patent_versions`; canonical = most complete, then newest source; open `duplicate_conflict` issue | issue list; output notice on that patent |
| Transient (DB timeout, S3 5xx, GPU preempted) | exception class | SQS redelivers after the visibility timeout; exponential backoff; max 3 attempts → DLQ | `attempts`, DLQ depth alert |
| GPU out of memory | CUDA OOM | halve the batch and retry in-process; a single over-long chunk is split again | worker log, counter |
| Chunk that still can't be embedded | repeated failure | chunk marked `embed_failed` (data issue); the patent stays searchable via BM25 and its other chunks | output notice on that patent |
| Worker crash mid-job | lease (`lease_until`) expires | another worker picks the job up; idempotent writes make the re-run safe | `attempts`, dashboard |
| Index/DB disagreement | nightly reconciliation counts per snapshot | block promotion; alert | dashboard, alert |
| Eval regression on a new index version | promotion gate | version marked `failed`, live version untouched | `index_versions` |

### 6.2 Serving failures

| Failure | Behaviour |
|---|---|
| Reranker down or slow | return phase-1 order with `pipeline_notice` (DESIGN.md 3B.4); never silent |
| Chunk-vector store slow | fall back to patent-level scores; evidence chunk comes from BM25 highlight; notice in the response |
| OpenSearch down | 503 with a clear message; no partial results presented as complete |
| Postgres slow (issue lookup) | results returned with `data_status: "unknown"` and a notice to retry, rather than claiming "complete" |

### 6.3 Dedupe rule

`doc_number` is the key. A content hash over normalized title + abstract + claims + description
detects exact copies. Families and continuations (different `doc_number`, near-identical text) are
**not** merged. They are linked with MinHash (≥ 0.8) into a `family_id`, so results can be collapsed
by family in the UI, and so evaluation splits (Part 3) keep families together.

---

## 7. Tracking contents and status

**Status dashboard** (reads Postgres, refreshes every few seconds):
- jobs by pipeline × stage × status, throughput (patents/s, tokens/s), ETA, queue depth per stage, DLQ depth, workers alive (last heartbeat);
- data quality: open issues by type and severity, quarantine count, duplicates, % of patents `complete / partial / incomplete`, research progress (resolved per day);
- index versions: live version, its snapshot, counts, eval metrics, and the next version's build progress.

**Metrics and alerts:**

| Metric | Alert when |
|---|---|
| DLQ depth | > 0 for 30 min |
| Job failure rate per stage | > 2% over 1 h |
| Embed throughput | < 50% of baseline (GPU problem) |
| Reconciliation mismatch | any |
| Search p95 latency per stage | > budget (§4.6) for 10 min |
| Rerank budget-exhausted rate | > 5% of queries |
| Data-issue open count | rising week over week (research falling behind) |

**Lineage:** every patent row records `source_uri`, `source_sha256`, `snapshot_id`,
`pipeline_version`; every vector records `model_revision` and `chunking_version`. So "why does
this result look like this" can always be traced back to a file and a code version.

---

## 8. Implementation plan (milestones)

| # | Milestone | Done when |
|---|---|---|
| M1 | Jobs table + queue + one worker running the Part 1 parse/validate code on files from S3 | 10 batch files processed; killing a worker mid-job leads to a correct re-run; quarantine works on a deliberately truncated file |
| M2 | Chunk + embed stages, Parquet outputs, deterministic ids | re-running a job yields identical chunk ids and vectors (hash match) |
| M3 | OpenSearch index with filters + BM25 + patent-level kNN; LanceDB chunk table | the Part 1 test queries return the same top 10 as the local engine on the same 640 patents |
| M4 | Search API wired to OpenSearch + LanceDB + reranker; JSON contract unchanged | Part 3 eval metrics on the 640-patent set within tolerance of the local engine |
| M5 | Snapshots, index versions, validation gate, alias swap | a bad build (forced eval drop) is refused and live stays up |
| M6 | Weekly incremental + data-research job + dashboard + alerts | a simulated weekly drop lands end to end; a backfilled record flips saved results to "re-examine" |
| M7 | Scale test on the 480k large dataset (5% of target) | measured throughput and cost per 1k patents; extrapolation to 10M updated in §4.1 / §5 |

### 8.1 Where to start

**Before writing code:** read DESIGN.md §3–§6 (ingestion, storage, search, the JSON response), then
run the local engine and the proof-of-concept (README §3 and §6). Both run on a laptop.

**Start from the proof-of-concept.** `poc/` already implements milestone M1's behaviour on one
machine: the job queue with leases, retries and a dead state (`poc/queue.py`), salvage, dedupe,
data issues and an idempotent single-transaction write (`poc/process.py`), and status tracking
(`poc/status.py`, `poc/dashboard.py`). M1 is mostly a port of it:

| Proof-of-concept | Production (M1) |
|---|---|
| `jobs` table in SQLite | `jobs` table in Postgres (§3), with SQS carrying the job ids |
| `queue.enqueue / claim / renew / complete_in_tx / fail` | the same five functions over Postgres + SQS: enqueue = insert the `jobs` row + send the message; claim = receive message + set lease; renew = extend the visibility timeout; fail = let the message return, then dead-letter after 3 attempts |
| `zip://…!member` and `file://…` inputs | `s3://…` inputs (`read_input` in `poc/process.py` is the only function that changes) |
| worker processes (`poc/worker.py`) | the same worker loop in a container on ECS |
| `tests/test_poc.py` | keep these tests as the acceptance tests for the ported queue |

**Reuse the engine's code; do not rewrite it.** Each pipeline stage calls existing modules:

| Stage (§4.1) | Existing code to call |
|---|---|
| Parse + validate | `patent_search/ingest/load.py`, `normalize.py`, `claims.py`, `validate.py` (as `poc/process.py` already does per record) |
| Chunk | `patent_search/ingest/chunk.py` (adaptive chunking) |
| Embed | `patent_search/embed/encoder.py` (pinned model; set the device to `cuda` on GPU workers) and `embed/cache.py` |
| Data-gap notices | `patent_search/notices.py` (fixed wording) |
| Search API | `patent_search/search/engine.py`, `api/app.py`, `schemas.py`. The JSON contract stays the same; only the retrieval back-end changes (`search/filters.py` and `search/hybrid.py` → OpenSearch queries, `search/planner.py` → LanceDB fetch by document) |
| Rerank | `patent_search/search/rerank.py` |
| Validation gate (M5) | `patent_search/eval/runner.py` and `metrics.py` with the sets in `eval_data/` |
| Index versions | `patent_search/manifest.py` and `index/verify.py` (what a build contains, and a check that it is reproducible) |

**Job message** (the body of an SQS message; the full state lives in the `jobs` row):

```json
{
  "job_id": 1234,
  "pipeline": "backfill",
  "stage": "parse",
  "input_uri": "s3://patents-raw/uspto/2026-11-05/patents_batch_0030.json",
  "input_sha256": "9f2c…",
  "snapshot_id": "2026-11-10",
  "pipeline_version": "git:bb7b2dc",
  "model_revision": null,
  "idempotency_key": "sha256(stage, input_sha256, pipeline_version, model_revision)"
}
```

**Files passed between stages** (Parquet, one file per job, keyed by `snapshot_id`):

| Output | Columns |
|---|---|
| `normalized/` | the `patents` and `claims` columns of DESIGN.md §4, plus `snapshot_id`, `source_uri`, `source_sha256` |
| `chunks/` | `chunk_key, doc_number, kind, claim_number, element_idx, section, para_span, parent_chunk_key, tier, n_tokens, embed_text, display_text, weight` |
| `vectors/` | `chunk_key, model_revision, vec_768` (float16, archive) and `vec_256` (int8, serving) |

**Develop locally before touching the cloud.** Run Postgres and a single-node OpenSearch in Docker,
and LocalStack for S3 and SQS, so the whole pipeline (M1–M4) runs on a laptop against the 640-patent
dataset. Compare its search results with the local engine's (M3), and only then point it at AWS.

**First week:**
1. Run `pytest`, the local engine and the proof-of-concept; read DESIGN.md §3–§6.
2. Port `poc/queue.py` to Postgres + SQS behind the same five functions; make `tests/test_poc.py` pass against it.
3. Change `read_input` to read from S3 (LocalStack locally) and write the `normalized/` Parquet output.
4. Process 10 batch files end to end, kill a worker mid-job, and confirm the M1 "done when" test.

---

## 9. Major challenges (acknowledged, not solved)

1. **Embedding cost and time dominate.** ~1,075 GPU-hours per full build makes model iteration expensive. Mitigations: re-embed only claims and abstracts (≈ 15% of tokens) for experiments, Matryoshka dims, distilled models.
2. **Filtered vector search with very selective filters.** A CPC group can be 0.01% of 10M. HNSW with a restrictive filter can return too few results or degrade to slow traversal. OpenSearch's filtered kNN switches to exact search below a threshold (the same planner idea as Part 1); the threshold needs tuning, and our patent-level tier keeps exact search affordable (≤ ~40M vectors × selectivity).
3. **Classification gaps.** 37% `N/A` in the large sample means CPC filters silently miss about a third of patents unless research backfills them. The design makes this visible in every filtered response, but recall under CPC filters is incomplete until the backfill finishes.
4. **BM25 over descriptions.** ~110B words of description would make the lexical index ~1 TB. The first version indexes title/abstract/claims lexically and covers descriptions with dense vectors only. Exact-phrase search inside descriptions is therefore missing at scale; adding it is a separate index and roughly doubles the OpenSearch cost.
5. **Consistency across three stores** (Postgres, OpenSearch, chunk-vector store). Snapshots and reconciliation reduce this but add moving parts. Weekly upserts between full builds can drift; nightly reconciliation is the safety net.
6. **Domain shift.** The large data covers all technology (only ~2% vehicles). A bi-encoder fine-tuned on vehicle patents (Part 3) may hurt other areas. Either fine-tune on a mix across CPC sections or keep the general model for non-B60 queries; the eval set must then be stratified by CPC section.
7. **Duplicates and families.** Without family data, MinHash linking is approximate; continuations can crowd the top 30. Collapsing by family is a UI decision that needs examiner input.
8. **Reranker capacity.** At 10 QPS × 200 pairs, a single L4 is likely near its limit. Since reranking is opt-in, peaks can fall back to phase-1 results (with a notice) instead of autoscaling GPUs.
9. **Saved-result invalidation** at scale: one research fix can touch thousands of saved claim maps. Marking them "re-examine" must be a batch job, not a trigger.

---

## 10. Proof of concept: local job queue, parallel workers and a live status dashboard

**What it demonstrates:** the backbone of §4 and §6–7 (idempotent jobs, leases, retries,
quarantine and salvage, dedupe, data issues, status tracking) on **real files from
`coding_challenge_large_data.zip`**, and it measures the throughput numbers that §4.1 and §5 use.

**Why ingestion:** at 10M patents, ingestion is where things actually break (corrupted files,
duplicates, crashed workers, data gaps), and it produces the throughput numbers behind the cost
estimate. A proof-of-concept that measures them is more useful than one that only stands up a database.

**Runs locally with only Python and pip.** Docker and Postgres are not required, so anyone can run it
on an ordinary laptop.

| At-scale component | PoC stand-in | Same semantics? |
|---|---|---|
| SQS + `jobs` table in Postgres | `jobs` table in SQLite (WAL mode); a job is claimed with an atomic `UPDATE … RETURNING` inside `BEGIN IMMEDIATE`, with a `lease_until` | yes: lease, visibility timeout, attempts, dead state |
| ECS worker containers | N local worker **processes** (`--workers 4`) | yes: independent, crash-able, stateless |
| S3 raw files | the zip itself, read **member by member** without extracting (the zip's CRC32 is part of the idempotency key) | yes: immutable source |
| Postgres patents / issues / quarantine | `poc.db` (SQLite) using the Part 1 schema | yes |
| GPU embed workers | optional embed stage on CPU for a **sample** of patents per job (`--embed-sample 3`), to measure tokens/s | partial: proves the stage and measures speed; not full embedding |
| Grafana dashboard | FastAPI page polling `/api/status` every 2 s | yes |

**Components (`poc/`):**

```
poc/
  db.py          SQLite (WAL) schema: jobs, patents, patent_versions, data_issues, quarantine, workers
  queue.py       claim (BEGIN IMMEDIATE: select + update, lease), renew, complete (inside the data transaction),
                 fail (exponential backoff; dead after 3 attempts); re-claims jobs whose lease expired
  enqueue.py     one job per batch file in the zip, read member by member; key = hash(member, CRC32, size,
                 pipeline version), so re-running inserts nothing
  process.py     read -> tolerant parse (salvage a truncated/corrupt array record by record, quarantine the rest
                 with its byte offset) -> Part 1 ingest per record (claims, adaptive chunks, data issues)
                 -> optional embed sample -> ONE transaction: dedupe + upserts + issues + quarantine + job done
  worker.py      N worker processes; heartbeat thread renews the lease; --fail-rate injects failures;
                 --kill-one-after SIGKILLs worker 0 mid-run
  faults.py      truncated copy of a real batch (60% of its bytes), a file with an exact duplicate, a
                 conflicting duplicate, a record without doc_number and a non-object, plus a missing file
  status.py      shared summary (terminal: python -m poc.status)
  dashboard.py   live page on http://localhost:8001 polling /api/status every 2 s
tests/test_poc.py  claim/lease/expiry, backoff -> dead, salvage, idempotent re-run, dedupe, lost lease
```

Duplicates: same `doc_number` + same content from another file is skipped and counted; different
content is kept in `patent_versions`, a `duplicate_conflict` data issue is opened, and the most complete
version is canonical. A record re-processed from the *same* file position is an idempotent no-op.

**Commands** (run from the project folder):

```
.venv/bin/python -m poc.enqueue --zip coding_challenge_large_data.zip --jobs job_000 --limit 20
.venv/bin/python -m poc.faults --zip coding_challenge_large_data.zip --out poc_data/faults
.venv/bin/python -m poc.dashboard            # http://localhost:8001
.venv/bin/python -m poc.worker --workers 4 --fail-rate 0.05 --embed-sample 3 --kill-one-after 60
.venv/bin/python -m poc.status
```

`--embed-sample 3` embeds the full chunk set of 3 patents per job on CPU (2 threads per worker), only to
measure embedding throughput and tokens per patent.

**Results** (Apple M4, 4 worker processes, 20 real batch files from `job_000` + 3 fault inputs; one run):

| What | Result |
|---|---|
| Jobs | 23: **20 succeeded, 2 partial, 1 dead**; 27 attempts (4 retries) |
| Idempotent enqueue | enqueuing the same 20 files twice added 0 jobs the second time |
| Killed worker | worker 0 SIGKILLed at 60 s mid-job; its job was re-claimed when the lease expired and succeeded (history: `lease expired (worker stopped responding) → succeeded`) |
| Injected failure | 1 job attempt failed (5% rate) and succeeded on retry after backoff |
| Missing file | failed 3 times (FileNotFoundError), then **dead**, with the error recorded |
| Truncated file (60% of a batch) | **567 records salvaged**, 565 stored; the rest quarantined from byte offset 42,839,753; job `partial`, `corrupt_source` issue opened |
| Bad records | a record without `doc_number` and a non-object record, both quarantined; job `partial` |
| No duplicate rows | 20,539 patent rows = 20,539 distinct doc numbers, from 20,571 records |
| Duplicates in the real data | **20 exact** (skipped) and **8 conflicting** across batch files, plus the injected 1 exact + 1 conflicting |
| Data gaps (kept, not dropped) | classification missing for **37.9%** of patents; filing date missing for **100%** of the large set (its file names carry no date); 219 patents `incomplete`, 20,320 `partial` |
| Parse + validate + chunk | **~19 patents/s per worker**. Of 1,087 s of this work, JSON parsing was 12 s and adaptive chunking 1,075 s |
| Measured sizes | **~96 chunks and ~19.4k tokens per patent** (model tokenizer) |
| Embedding on CPU | ~655 tokens/s per worker (2 threads, 4 workers sharing the CPU) |

**Extrapolation to 10M patents** (used in §1, §4.1 and §5):
- parse + validate + chunk: 10M ÷ 19 patents/s ≈ **147 worker-hours** (≈ 2.3 h on 64 vCPUs);
- embedding: 10M × 19.4k tokens ≈ 194B tokens → **~1,075 GPU-hours** at an *assumed* 50k tokens/s per L4. On CPU at the measured rate it would be ~82,000 worker-hours, which is why §2 uses GPU workers;
- adaptive chunking, not parsing, is the CPU cost to optimise first (for example, cache the per-patent TF-IDF or move it to a compiled tokenizer).

**What it does not prove:** coordination across machines (SQLite is single-host), OpenSearch query
performance at 10M, GPU embedding throughput (assumed, not measured), or cost. No Docker files are
included: the workers run as plain local processes.
