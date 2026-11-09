# Patent Claim Search: System Design

This document describes how the patent search engine works and why it is built this way.

| Document | Contents |
|---|---|
| **DESIGN.md** (this file) | the search engine (Part 1) and its two enhancements, evaluation & training and two-phase search (Part 3) |
| [SCALE.md](SCALE.md) | how the same engine would run over 10 million patents (Part 2), and a working proof-of-concept |
| [README.md](README.md) | how to run everything, and all measured results |

---

## 1. The problem

A patent examiner or attorney has an idea, a claim, or an existing application. They need to find
**which other vehicle patent applications cover overlapping subject matter, and exactly where**:
which claim, which claim element, which description passage. They also narrow results the way
examiners work: by classification code (e.g. only `B60B`, vehicle wheels), by keywords that must
appear in the title or abstract, or by a specific title.

Novelty and overlap are judged **element by element**. A claim is new over an earlier document only if at
least one of its elements is not disclosed there. So the engine must account for every element of
a claim, and must tell "not disclosed" apart from "could not be checked because the source data
is incomplete".

| Input | Output |
|---|---|
| A natural-language query ("tire with noise-damping foam inside the cavity") | Ranked **patents**, each with its **best-matching passage**, why it matched, its data status, and metadata |
| A claim (pasted text) | The same, plus **element coverage**: each element of the claim → the best evidence found, or "not found", or "undetermined (data incomplete)" |
| The document number of a patent in the dataset | A **claim map**: each independent claim of that patent → the closest claims in other patents, with element coverage |
| Any of the above + filters (classification prefix, title/abstract keywords, title, filing-week range) | The same results, restricted to the filters, plus counts by classification and by week |

---

## 2. What the data looks like

The dataset is 64 weekly JSON files (`patents_ipa{YYMMDD}.json`) of vehicle patent applications.

| Fact | Value | Consequence for the design |
|---|---|---|
| Files | 64 weekly files, 2024-02-15 → 2025-05-01 | The week is read from the file name into `filing_week` |
| Patents | 640, every `doc_number` unique | `doc_number` is the primary key |
| Fields | `title, doc_number, filename, abstract, detailed_description[], claims[], bibtex, classification` | Every key is present, but several contain **defective content** (rows below) |
| Classification | One code per patent, e.g. `B60B2700FI`; 145 distinct; B60C 318, B60B 298, B60D 20, others 1 each | Codes are filtered by **prefix**. Section, class and subclass (`B`, `B60`, `B60B`) are unambiguous; below that the missing slash makes the split ambiguous (`B60C1103` = `11/03` or `1/103`), so finer drill-down uses raw prefixes |
| Claims | 10,578 claim items, median 19 per patent; 167 are `"(canceled)"`, 71 of them ranges (`"1 - 5 . (canceled)"`) | Canceled claims are kept as status rows (shown in claim maps) but not searched |
| Claim numbers | 3,221 claim items carry no number | The number is inferred from position and flagged |
| Lost claim openings | Claim 1 starts mid-sentence in **501 of 640 patents** (`"an axle body, having …"`); a later claim does in 511 | Flagged as data issues; the title is added to each claim's text as context |
| Claim complexity | Dependency chains up to 19 claims deep; independent claims p50 57 / p90 178 / max 812 words; a claim plus all its ancestor claims p50 118 / p99 513 / max 950 words | Drives **adaptive chunking** (§3.3) and the choice of a long-context embedding model |
| Description | 42,040 paragraphs, of which 20,819 are empty; non-empty p50 70 / max 4,045 words; 817 under 15 words (mostly headings) | Empty paragraphs are blank lines; headings become section labels; long paragraphs are split |
| Thin description | 127 patents have under 50 words of description | Flagged; element checks that would need the description report "undetermined" |
| Text artifacts | `inFIG.1`, `tire1of`, `claim 6 ,` | Spacing is repaired before indexing; wording is never changed |
| Titles | `TIRE` ×47, `PNEUMATIC TIRE` ×33 | A title search returns **all** matches; a title is never used as an identifier |
| Volume | ≈ 3.1M words of text | Fits in memory; embedding runs on a laptop CPU |

---

## 3. Ingestion

```
JSON files ─► load ─► detect data issues ─► normalize ─► parse claims ─► adaptive chunks ─► embed ─► SQLite + vectors
```

### 3.1 Normalization and claim parsing

- **Spacing only.** Whitespace is collapsed and glued text is separated (`inFIG.1` → `in FIG. 1`, `tire1of` → `tire 1 of`, `claim 6 ,` → `claim 6,`). Chemical formulas (`SiO2`), ordinals (`2nd`) and decimals (`1.5 mm`) are left alone.
- **Claim numbers** come from the inline prefix (`7 . …`), including canceled ranges (`1 - 5 . (canceled)` → claims 1–5). A claim without a prefix gets the next number by position and is marked `number_inferred`.
- **Dependencies** are read from phrases such as `according to claim N`, `of claim N` and `any one of claims N to M`. A claim can depend on several claims.
- **Elements**: an independent claim's body is split into its limitations on `;`, after the transitional phrase (`comprising:`), and before `wherein` clauses. These elements are the units of element coverage (§5.4).

### 3.2 Data gaps: recorded and shown, never dropped

Nothing is skipped silently. Every defect becomes a row in the `data_issues` table, every patent gets
a `data_status`, and every output that relies on an affected patent says so (§6.3).

| Issue type | Count in the dataset | Severity | Effect on outputs |
|---|---|---|---|
| `missing_field` | a field absent or empty (none in this dataset; the rule still applies) | high for abstract, claims, description; medium otherwise | the field is shown as missing, pending data research |
| `no_identifier` | a record without `doc_number` | blocking | kept in quarantine with its file and position; not searchable until researched |
| `claim1_preamble_missing` | 501 patents | high | notice on results; matches on claim 1 are provisional |
| `claim_opening_missing` | 511 patents | high | a later claim starts mid-sentence, so its "according to claim N" may be lost; it is treated as independent until researched |
| `parent_claim_unresolved` | 36 patents | high | a dependent claim cites a claim that is not in the data; it is indexed without that parent's text |
| `description_thin` | 127 patents | high | element checks that need the description report "undetermined", never "not found" |
| `claim_number_inferred` | 613 patents | medium | claim numbers are shown as inferred |
| `claim_canceled` | 89 patents (1,031 claims) | info | shown in claim maps as canceled |
| `classification_ambiguous` | all 640 codes | info | affects drill-down below subclass only, so no per-result notice |

Each issue has a status: `open` → `researching` → `resolved` (data filled in) or `confirmed_absent`
(the source itself lacks it). The source of a fix is recorded.

**Data research.** `python -m patent_search data-issues` lists open issues by severity. Each record's
`filename` (e.g. `US20240051333A1-20240215.XML`) names the USPTO full-text XML that can fill the gap.
When a record is fixed, only that patent is re-processed, its `record_version` increases, its issues
are closed, and any saved result that used the old version is marked **"record updated — re-examine"**.

**Data status** per patent: `complete` (no open issue above `info`), `partial` (open medium or high
issues), `incomplete` (a field needed for the assessment is missing, or a blocking issue is open).
In this dataset: 622 partial, 18 complete, 0 incomplete. Incomplete data does not lower a patent's
rank (missing data is not evidence of irrelevance), but the output always says what is missing.

### 3.3 Adaptive chunking

A fixed-size window treats a one-line claim and a 20-element apparatus claim the same way. Instead,
each claim and passage gets **as much context as it needs to be understood on its own**.

**Claims are sorted into three tiers** using: dependency depth, number of elements, length, and
whether the claim depends on several claims.

| Tier | Rule | Chunks produced |
|---|---|---|
| **Simple** | independent, ≤ 60 words, ≤ 2 elements | one chunk: title + claim |
| **Moderate** | independent with 3–10 elements or 61–250 words, or a dependent claim one level deep | the claim with its context (title + parent claim + claim), and for independent claims one chunk per element |
| **Complex** | two or more levels deep, depends on several claims, ≥ 11 elements, or > 250 words | the claim with its **full ancestor chain** (root claim + each intermediate claim + this claim), and one chunk per element |

A dependent claim adds little text (median 31 words), so a chain is mostly "root claim + small
additions". If a chain exceeds 1,536 tokens, the intermediate claims are cut to their `wherein`
clause, oldest first. This affects about 3% of chains.

**Description passages are sized by structure and topic:**
- Empty paragraphs are dropped. Short heading-like paragraphs ("BACKGROUND") become the section label of the passages below them.
- Neighbouring paragraphs are **merged** while they stay on one topic (word-overlap similarity, TF-IDF cosine ≥ 0.10 within the patent) and the result stays under the target size.
- Long paragraphs are **split** at sentence boundaries, and early where the topic shifts.
- Dense passages (≥ 5% of words are reference numerals like "spoke 12") aim for ~150 words; narrative passages for up to ~350. Standard boilerplate ("presented for purpose of illustration…") is detected and counts at half weight.

**Small chunks point to big ones.** Every element chunk points to its claim chunk. A search can
match a precise element and still show the whole claim; an element is never shown without its claim.

**Result:** 33,173 chunks: 10,451 claims (1,549 simple, 6,479 moderate, 2,423 complex), 6,932
elements, 15,150 description passages and 640 abstracts. None exceeds the 2,048-token input limit used.

**Embedding model.** Complex chains need more than 512 tokens, and everything runs on a laptop, so
the model must be long-context **and** small: `Alibaba-NLP/gte-modernbert-base` (149M parameters,
768 dimensions, 8,192-token context), loaded through `sentence-transformers`.

### 3.4 Local, reproducible embedding

Embedding runs inside the program on the local machine: no external service or API key. The same
encoder embeds chunks when the index is built and queries at search time.

| Rule | How |
|---|---|
| Same libraries | exact versions pinned in `requirements.lock` |
| Same model | model name **and** revision hash pinned in `config.py` |
| Same order | files read in sorted order; chunk ids assigned deterministically |
| Same numbers | CPU, **float32** (the model's float16 default is ~13× slower on CPU). A build without cache is deterministic; re-encoding a sample matches stored vectors to ~1e-7 |
| Checkable | `index/manifest.json` records the model and revision, device, library versions, a SHA-256 of every data file, the configuration, a digest of every chunk's text and a SHA-256 of the vectors. `verify-index` re-runs ingest and chunking, re-encodes a sample, and compares |
| No repeated work | an embedding cache keyed by model revision + text; each distinct text is encoded once |

---

## 4. Storage

One SQLite file (`index/patents.db`) and one vector file (`index/vectors.npy`, row *i* = chunk *i*).
SQLite provides B-tree indexes for the structured filters and **FTS5** full-text search (BM25
ranking, phrase and boolean queries) without extra infrastructure.

```sql
CREATE TABLE patents (doc_number TEXT PRIMARY KEY, title TEXT, title_norm TEXT, abstract TEXT,
  classification TEXT, cpc_subclass TEXT, filing_week TEXT, source_file TEXT, source_xml TEXT,
  bibtex TEXT, data_status TEXT, record_version INTEGER, n_claims_active INTEGER,
  n_claims_canceled INTEGER, description_words INTEGER);
CREATE INDEX ix_pat_cpc ON patents(classification);      -- prefix filter = range scan on this index
CREATE INDEX ix_pat_week ON patents(filing_week);
CREATE INDEX ix_pat_title ON patents(title_norm);         -- exact title match

CREATE TABLE claims (doc_number TEXT, claim_number INTEGER, number_inferred INTEGER, status TEXT,
  depends_on TEXT, unresolved_refs TEXT, depth INTEGER, n_elements INTEGER, tier TEXT,
  preamble TEXT, elements TEXT, text TEXT, PRIMARY KEY (doc_number, claim_number));

CREATE TABLE chunks (id INTEGER PRIMARY KEY, chunk_key TEXT UNIQUE, doc_number TEXT, kind TEXT,
  claim_number INTEGER, element_idx INTEGER, section TEXT, para_span TEXT, parent_chunk_id INTEGER,
  tier TEXT, n_tokens INTEGER, context_claims TEXT, embed_text TEXT, display_text TEXT, weight REAL);

CREATE TABLE data_issues (id INTEGER PRIMARY KEY, doc_number TEXT, location TEXT, issue_type TEXT,
  severity TEXT, status TEXT, detail TEXT, detected_at TEXT, resolved_at TEXT, resolution_source TEXT);
CREATE TABLE quarantine (id INTEGER PRIMARY KEY, file TEXT, position INTEGER, reason TEXT, detail TEXT);

CREATE VIRTUAL TABLE patents_fts USING fts5(title, abstract, content='patents', tokenize='porter unicode61');
CREATE VIRTUAL TABLE chunks_fts USING fts5(display_text, content='chunks', content_rowid='id',
  tokenize='porter unicode61');
```

---

## 5. Search

```
request
  ├─1─ Filters (cheap, exact)       classification prefix, week range → B-tree; keywords, title → full-text index
  │                                 ⇒ the set of allowed patents
  ├─2─ Query analysis               the query is tiered like a claim; a multi-element claim is also split into elements
  ├─3─ Retrieval, two ways          dense: vector similarity over allowed chunks only
  │                                 lexical: BM25 over allowed chunks only
  ├─4─ Fusion                       Reciprocal Rank Fusion (k = 60) × chunk-kind weight × chunk weight;
  │                                 element hits roll up to their claim, claims to their patent
  ├─5─ Phase 2 (optional)           cross-encoder reranks the top 100 patents, returns the top 30 (Part 3B)
  └─6─ Explain, check, flag         best evidence per patent, element coverage, data-gap notices, counts
```

### 5.1 Filtering first makes hybrid search faster

If vector search ran over everything and filters were applied afterwards, a selective filter (B60D is
20 of 640 patents; at 10M patents a classification group can be 0.01%) would discard almost every hit.
So the filters run first, on indexes built for them, and produce the set of allowed patents. Vector
scoring then touches only those patents' chunks:

| Allowed set | Strategy |
|---|---|
| ≤ 50,000 chunks | gather those rows of the vector matrix and score them exactly |
| larger, or no filter | score the full matrix with the disallowed rows masked out |

Both paths are exact here (all 33k chunks fit in memory, ~10 ms). At 10M patents the second path
becomes approximate search with over-fetching (SCALE.md §4.6). BM25 is restricted the same way, by
joining the full-text index with a temporary table of allowed chunks.

### 5.2 Adaptive retrieval for the query

- **A simple query** (a phrase, a fact, a one-element claim) runs one search.
- **A multi-element claim** runs the whole claim **and** each element as its own sub-query. Whole-claim matches find documents that look similar overall; element matches find documents that disclose one specific limitation. Both feed fusion and element coverage.

### 5.3 Query modes

| Mode | What it does | Chunk-kind weights (abstract / claim / element / description) |
|---|---|---|
| `text` | the pipeline above | 1.0 / 1.0 / 1.0 / 1.0 |
| `claim` | the query is a claim; element coverage is computed | **1.0 / 0.6 / 0.3 / 0.6**, tuned on the dev split (Part 3A.5) |
| `patent` | each independent claim of a patent in the dataset runs as a `claim` query; the patent itself is excluded; the output is a claim map | as `claim` |

The `claim` weights matter a great deal. Claim and element chunks from *other* patents are written in
the same style as the query claim. At full weight, thousands of them crowd out the abstract and
description passages that actually disclose the claimed subject matter. Lowering their weight raised
test MRR@10 from 0.557 (initial weights: abstract 0.8, claim 1.0, element 1.0, description 0.6) to
0.825 (README §5.3).

### 5.4 Element coverage

For each element of the query claim and each top patent, the best-matching chunk of that patent is
found by cosine similarity:

| Status | Meaning |
|---|---|
| `found` | similarity ≥ the `found` threshold; evidence and score attached |
| `weak` | between the `weak` and `found` thresholds; worth checking (marked provisional if the patent has data gaps) |
| `not_found` | below `weak`, and the patent's data is complete for the parts that could disclose it |
| `undetermined` | below `weak`, but the patent has open data gaps in the parts that could disclose it (thin description, lost claim opening, missing parent claim). Reported as pending data research, **never** as `not_found` |

Thresholds are calibrated per index without labels (`scripts/calibrate_coverage.py`), because each
embedding model has its own similarity scale. `found` is the 97.5th percentile of similarity between a
claim element and *other* patents in the same subclass; `weak` is the 99th percentile for patents in
*other* subclasses. Base model: `found` 0.802, `weak` 0.744 (an element's own patent reaches `found`
96.4% of the time, a same-subclass patent 2.5%). Fine-tuned model: 0.638 and 0.572.

A patent where every element is `found` is a potential anticipating reference. Any `undetermined`
element makes that conclusion provisional, and the output says so.

### 5.5 Filters

All filters are optional and combine with AND.

| Filter | Meaning | Implementation |
|---|---|---|
| `cpc_prefix: ["B60B", "B60C11"]` | classification starts with any listed prefix | index range scan (`code >= prefix AND code < prefix + max char`) |
| `cpc_exclude: [...]` | classification starts with none of them (patents without a code are kept) | negated range scan |
| `keywords: {terms, field, op}` | terms must appear in the title, the abstract or either; all / any / exact phrase | FTS5, e.g. `{title abstract} : ("spoke" AND "carbon")` |
| `title: {value, mode}` | exact title (case-insensitive) or a phrase within the title | `title_norm = ?` / FTS5 phrase |
| `week_from`, `week_to` | filing-week range | `filing_week BETWEEN …` |

A request with filters and no query returns the filtered patents, newest first. When patents lack a
classification or a filing date, a filtered response says how many could not be matched and why.

---

## 6. Outputs

### 6.1 One format: JSON

Every search produces one **typed response object** (Pydantic models in `patent_search/schemas.py`),
serialized as JSON. Every other output reads that object; nothing parses text.

| Consumer | Receives |
|---|---|
| HTTP API (`POST /search`) | the JSON response; an interactive page is at `/docs` |
| Command line (`search …`) | a readable text view by default, or the same JSON with `--format json` |
| Claim chart export | a Markdown or CSV table: query element × reference patent × evidence × status |
| Evaluation (Part 3A) | JSONL run files, one line per query, with the ranking and per-stage timings |

`schema_version` (currently 1.1) lets new optional fields be added without breaking consumers.

### 6.2 Response shape

```jsonc
{
  "schema_version": "1.1",
  "request_id": "q_3f9c…",
  "took_ms": 66,
  "pipeline": ["dense:full_scan", "bm25", "rrf", "coverage"],    // + "rerank:<model>@<rev>" when reranked
  "query": { "mode": "claim", "tier": "moderate", "elements": ["a tread portion …", "a pair of bead portions …"] },
  "filters_applied": { "cpc_prefix": ["B60C"] },
  "total_matched": 118,
  "data_notice": "3 of 30 results have incomplete or partial source data. Their overlap/novelty assessment is provisional …",
  "filter_notices": [],
  "results": [{
    "rank": 1, "doc_number": "20250121635", "title": "PNEUMATIC TIRE", "classification": "B60C1900FI",
    "filing_week": "2025-04-17", "record_version": 1, "score": 0.0375,
    "signals": { "dense_rank": 1, "bm25_rank": 2, "rrf": 0.031, "phase1_rank": 1, "rerank": null, "coverage": 0.4 },
    "best_match": { "kind": "claim", "claim_number": 2, "context_claims": [1], "display_text": "…", "highlights": [[12, 17]] },
    "coverage": [
      { "element": 0, "element_text": "a tread portion", "status": "found", "score": 0.83, "evidence": { "kind": "element", "claim_number": 1, "text": "…" } },
      { "element": 3, "element_text": "a sound absorbing member …", "status": "undetermined", "reason": "… data gaps open (claim1_preamble_missing), pending data research", "issue_ids": [412] }
    ],
    "data_status": "partial",
    "data_issues": [{ "id": 412, "type": "claim1_preamble_missing", "severity": "high", "status": "open", "notice": "…" }],
    "matched_filters": ["cpc_prefix:B60C"]
  }],
  "facets": { "cpc_subclass": { "B60C": 118 }, "cpc_code": { "B60C1900FI": 40 }, "filing_week": { "2025-04-17": 3 },
              "data_status": { "partial": 115, "complete": 3 } },
  "claim_map": null,                     // filled in patent mode
  "stage_ms": { "filter": 0, "phase1": 60, "coverage": 3 },
  "rerank": null,                        // model, device, depth, pairs, alpha, beta when reranked
  "pipeline_notices": []                 // e.g. when a rerank time budget ran out
}
```

Other endpoints: `GET /patents/{doc_number}` (full record with its data issues), `GET /data-issues`,
`GET /health`.

### 6.3 How data gaps appear in outputs

- A result with an open issue above `info` carries its `data_status` and a notice per issue; the response carries a `data_notice` whenever any result does.
- Every notice says what is missing, that the assessment is **provisional**, that it **will be updated once data research is complete**, and to **re-examine after the update**. No notice says a record or field is ignored.
- The text view shows the `data_notice` once at the top and one short line per affected result; claim charts mark affected columns with a footnote.
- Element coverage never reports `not_found` where the data is incomplete; it reports `undetermined` with the issue ids.

---

## 7. Code layout

```
patent_search/
  config.py          all settings (model + pinned revision, chunking thresholds, search weights, rerank settings)
  schemas.py         the JSON contract (Pydantic request/response models)
  notices.py         fixed wording of data-gap notices
  manifest.py        index manifest (what was built, from what, with what)
  ingest/            load.py, normalize.py, claims.py, validate.py, chunk.py
  embed/             encoder.py (local model, float32; hashed bag-of-words stand-in for tests), cache.py
  index/             db.py (schema), build.py (build-index), verify.py (verify-index)
  search/            filters.py, planner.py, query.py, hybrid.py, coverage.py, rerank.py, engine.py
  render/            text.py (terminal view), chart.py (Markdown / CSV claim charts)
  eval/              build_sets.py, runner.py, metrics.py, report.py      (Part 3A)
  train/             finetune.py                                         (Part 3A)
  api/app.py         FastAPI app
  cli.py             build-index, verify-index, search, interactive, show, data-issues, eval, train
poc/                 the Part 2 proof-of-concept (SCALE.md §10)
scripts/             calibration, tuning, test-run, latency and table-generation scripts
tests/               34 tests; a stand-in encoder and reranker keep them fast and offline
```

---

## 8. How correctness is checked

- **Unit and end-to-end tests** (`pytest`): claim parsing (numbers, ranges, dependencies, inferred numbers), tiers, ancestor chains and the token budget, description merging and splitting, data-issue detection, filter queries, the planner (both paths give identical results), coverage never reporting `not_found` on incomplete data, two-phase search only reordering phase-1 candidates, evaluation metrics, family-preserving splits, and the PoC queue.
- **`verify-index`** confirms that the index on disk is exactly what the code, the data and the pinned model produce.
- **Filter correctness:** 7,510 results over 7 filter types, with and without reranking, gave 0 violations (Part 3A.3, T5).

---

## Part 2: running at 10 million patents

The design for the full patent database is in **[SCALE.md](SCALE.md)**: components, pipelines,
cost, error handling, status tracking and known challenges. It includes a proof-of-concept that ran
on real files from the large dataset, and its sizing and cost use the numbers that run measured.

---

## Part 3: evaluation & training, and two-phase search

The two enhancements are **evaluation & training** and **two-phase search**. They need no data beyond
what is available, run locally, and support each other: the evaluation is what shows whether
two-phase search, the fine-tuned model and the choices made in the engine actually help.

### How Part 3 connects to the engine

Part 3 is part of the same system, not a separate tool:

| Engine component | What Part 3 adds |
|---|---|
| `Engine._search` | phase 2 (`search/rerank.py`) runs on the phase-1 candidates; filters are already applied, so it can only reorder |
| Request / response | `rerank`, `rerank_model`, `rerank_depth`, `retrieval`, `chunk_kinds`, `hide_claims_of`; `signals.phase1_rank`, `signals.rerank`, `stage_ms`, `rerank`, `pipeline_notices` |
| `build-index` | `--chunking fixed` (a comparison baseline) and `--model <dir>` (the fine-tuned model). The engine reads the model from the index manifest, so `--index index_ft` uses the fine-tuned model |
| `config.py` | the claim-mode weights and rerank α/β are the dev-tuned values |
| Element coverage | thresholds calibrated per index |
| Command line and API | `--rerank`, `--rerank-model`, `--rerank-depth`, `--retrieval`; the text view shows `(was N)`, each result's rank before reranking; `eval` and `train` commands |
| Evaluation runner | calls the same `Engine.search` that users call, so it measures the real system |

### 3A. Evaluation and training

#### 3A.1 Labels from document structure

There are no human relevance judgments and no examiner citations in this dataset, so the labels are
**built from document structure** ("silver" labels): a claim and its own patent's abstract and
description describe the same invention.

- **Positive pairs:** (independent claim, its patent's abstract), used for evaluation and training. Training also uses (claim, the most similar passage of its own description) and (one claim element, the most similar passage of its own description).
- **Hard negatives** look related but are not the target: other patents in the **same classification subclass** that the engine ranks 5th to 50th for the claim, excluding the same patent family and any patent whose abstract is near-identical to the positive (cosine > 0.95). Up to 3 per claim. For training triplets, negatives are drawn **only from training patents**, so no dev or test patent's text is seen during training.
- **Patents with data gaps stay in.** Every query records its patent's `data_status`, and results are broken down by it.

#### 3A.2 Splits

- **By patent family, never by pair**, so near-duplicates cannot leak from training into testing. A family is approximated by the same title filed less than 8 weeks apart, or near-identical abstracts (word 5-gram overlap ≥ 0.8). The 640 patents form 518 families.
- **60 / 20 / 20** split by family with a fixed seed (13): 384 train, 128 dev, 128 test patents.
- All tuning and model selection use **dev**. **Test** is run once per final configuration.

#### 3A.3 Evaluation tasks

| Task | Query | Correct answer | Metrics |
|---|---|---|---|
| **T1** claim → patent | an independent claim (up to 2 per patent, ≥ 15 words) | its own patent, with **that patent's own claim and element chunks hidden**, so it must be found through its abstract or description | Recall@1/10/100, MRR@10, nDCG@10 |
| **T2** partial claim → patent | 2–3 randomly chosen elements of a claim with ≥ 4 elements | its own patent, same masking | as T1 |
| **T4** positives vs negatives | a claim | its own abstract must score above each hard negative | triplet accuracy, ROC-AUC, mean similarity of positives and negatives, margin |
| **T5** filter correctness | claims with 7 kinds of filter, with and without reranking | every result satisfies the filter | violations (must be 0) |

Dev has 234 queries (150 T1, 84 T2); test has 306 (224 T1, 82 T2). Every metric has a 95% bootstrap
confidence interval (1,000 resamples), and comparisons on the same queries use a paired bootstrap.
Results are also broken down by claim tier and by data status.

#### 3A.4 Experiments

Each experiment changes one thing:

| ID | Bi-encoder | Chunking | Retrieval | Rerank | Question it answers |
|---|---|---|---|---|---|
| E0 | — | adaptive | BM25 only | — | how far does keyword search alone get? |
| E1 | `gte-modernbert-base` | fixed 200-word windows | dense only | — | baseline for chunking |
| E2 | `gte-modernbert-base` | adaptive | dense only | — | does adaptive chunking help? (vs E1) |
| E3 | `gte-modernbert-base` | adaptive | hybrid<br>dense + BM25, fused by RRF | — | does fusing dense and BM25 help? (vs E0, E2) |
| E4 | `gte-modernbert-base`<br>fine-tuned (`gte-patent-ft`) | adaptive | hybrid<br>dense + BM25, fused by RRF | — | does fine-tuning help? (vs E3) |
| E5 | `gte-modernbert-base` | adaptive | hybrid<br>dense + BM25, fused by RRF | `bge-reranker-base`<br>top 100 → 30 | does reranking help the base model? (vs E3) |
| E6 | `gte-modernbert-base`<br>fine-tuned (`gte-patent-ft`) | adaptive | hybrid<br>dense + BM25, fused by RRF | `bge-reranker-base`<br>top 100 → 30 | the full pipeline (vs E4, E3) |

Reference rows: E3 with the original, untuned claim-mode weights; E1h (fixed chunks, hybrid); and
E1d / E2d (fixed vs adaptive, searching only abstracts and descriptions), which separates the
quality of the passages from the effect of claim chunks competing for the same queries.

#### 3A.5 Tuning on dev

- **Claim-mode chunk-kind weights:** a 12-point grid (claim 1.0 / 0.6 / 0.3 × element 1.0 / 0.3 × description 0.6 / 1.0, abstract 1.0). The chosen weights (claim 0.6, element 0.3, description 0.6, abstract 1.0) scored 0.907 MRR@10 on dev, against 0.644 for the initial weights (abstract 0.8, claim 1.0, element 1.0, description 0.6). Claim 0.3 scored 0.914, a difference within noise; the milder setting was kept because real use also values claim-to-claim matches, which these labels cannot reward.
- **Rerank blend:** α ∈ {0.2, 0.3, 0.5} × β ∈ {0, 0.1, 0.2} on the full pipeline; α 0.2, β 0.2 was best (Part 3B.3).

#### 3A.6 Training

- **Model:** `gte-modernbert-base`, fine-tuned in a plain PyTorch loop over the model's own forward pass (which reproduces its `encode()` output exactly).
- **Loss:** InfoNCE (multiple-negatives ranking). Each claim must score its own positive above every other positive in the batch and above its mined hard negative. Scale 20.
- **Data:** 2,649 triplets mined from the training split only. At most 6 per patent are used, so long patents don't dominate: 1,570 triplets, 1,292 with a hard negative.
- **Settings:** 1 epoch (99 steps), batch 16, learning rate 2e-5 with 10% warm-up then linear decay, maximum length 256 tokens, float32 on CPU, seed 13. Training took 24 minutes; the loss fell from 0.83 to 0.23. Details are in `models/gte-patent-ft/training_card.json`.
- **Selection rule:** keep the fine-tuned model if dev MRR@10 does not drop. It rose (+0.044), so it is kept as the `index_ft` index. The default index remains the base model: on test the two perform the same, and the base model needs no training to reproduce.

#### 3A.7 Weaknesses of this evaluation

- **The labels reward self-similarity.** Finding a claim's own patent is easier than finding *prior art* written by someone else in different words, and the labels cannot reward finding other patents that genuinely overlap, which is the real goal. Human judgments or examiner citations would be needed for that.
- **Some hard negatives are genuinely relevant**: two tire patents from different applicants can overlap. That makes scores look slightly worse than they are.
- **The test set is small** (306 queries over 128 patents), so confidence intervals are wide. Only 7 dev/test queries come from patents with complete data, so the complete-vs-partial breakdown is reported but too thin to conclude from.

### 3B. Two-phase search

#### 3B.1 Phases

| | Phase 1: recall | Phase 2: precision |
|---|---|---|
| Goal | don't miss the right patents | put the best ones on top |
| Method | the engine's hybrid search (§5) | a cross-encoder reads the query and a passage **together** and scores the pair |
| Output | the **top 100 patents**, each with its 2 best chunks | the **top 30 patents**, reordered |
| Runs on | CPU | the Apple GPU (`mps`) when available, otherwise CPU |

**Filters apply only in phase 1.** Phase 2 can reorder candidates but never add any, so a patent
outside the user's filters can never appear (T5: 0 violations). Phase 1 puts the correct patent in its
top 100 for 96.7% of test queries, which is the ceiling for phase 2.

#### 3B.2 Reranker model

| Model | Size | Input limit | Role |
|---|---|---|---|
| `BAAI/bge-reranker-base` | 278M parameters | 512 tokens per (query, passage) pair | the reranker behind `--rerank`, pinned to a fixed revision; runs on the Apple GPU when available |

#### 3B.3 Scoring

1. **Pairs.** For each of the 100 candidates, its 2 best phase-1 chunks are paired with the query, about 200 pairs. An element hit is shown as its whole claim, and every passage is prefixed with the patent title. Pairs longer than 512 tokens are truncated.
2. **Patent score** = the best of its pair scores.
3. **Final score** = α · z(rerank score) + (1 − α) · z(phase-1 score), plus in claim mode β × the share of query elements `found`. z-scores are taken within the 100 candidates. Dev tuning chose **α 0.2, β 0.2**: giving the cross-encoder more weight lowered accuracy, because on these patents it is weaker than the tuned phase 1.
4. **Data gaps stay visible.** Reranking never changes coverage statuses or data notices.

#### 3B.4 Cost, limits and caching

- **Cost:** measured with a cold cache on the Mac GPU, reranking adds about 7 s per query (7.3 s p50 and 10.5 s p95 in total, against 153 ms for phase 1 alone).
- **Opt-in:** reranking is turned on with `--rerank` (or `"rerank": true`), not by default, because its measured gain is small (README §5.3) and its cost is high.
- **Time budget:** 30 s in serving, well above the measured cold-cache p95 (10.5 s) plus first-query warm-up. If it runs out, the scored candidates are reranked, the rest keep their phase-1 order, and the response says so in `pipeline_notices`. Evaluation runs without a budget, so results never depend on how busy the machine is.
- **Score cache** on disk, keyed by model revision, device and the exact query and passage text. A pair is scored once, even across indexes that hold the same passages.

### 3C. Not included

- **T3, element → passage evaluation:** there is no ground truth for which passage supports an element. Labels picked by the model being evaluated would make the test circular.
- **A hand-labelled query set** and **examiner-citation labels:** they need human time or outside data.
- **Extra training data from the large dataset** (its ~2% vehicle patents), and **Matryoshka training** for smaller vectors.
- **Fine-tuning the cross-encoder** on the same triplets: the most promising next step, since the off-the-shelf rerankers are not trained on patents.
- **Other embedding models** (e.g. `bge-m3`, `PatentSBERTa`) as comparisons.
- **Patents outside the dataset:** there is no fetcher. One that reads the USPTO XML named in each record would also serve as the data-research backfill.

---

## Known limitations

- **Lost claim openings** (501 claim-1s and 511 later claims): the title is added as context, and some dependent claims are probably treated as independent. Fixing this needs the source XML; it is data research, not code.
- **Element splitting is rule-based** (`;`, the transitional phrase, `wherein`), so very short elements ("a rim") match weakly by construction.
- **Element coverage is cosine similarity** with thresholds calibrated without human labels. It indicates similar evidence, not a legal finding of disclosure.
- **Classification drill-down below subclass** uses raw prefixes, because the codes lack the slash.
- **Patent mode** grows slower with the number of independent claims (up to 10 are used), and does not rerank.
