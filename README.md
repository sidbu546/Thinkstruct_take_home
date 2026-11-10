# Patent Claim Search

A local, reproducible hybrid search engine for vehicle patent applications (2024–2025) that finds
**which other applications overlap with an idea, a claim, or a whole patent, and exactly where**:
which claim, which claim element, which description passage.

| Document | What's in it |
|---|---|
| **README.md** (this file) | problem statement, how to run, results |
| [DESIGN.md](DESIGN.md) | how the engine (Part 1) and its two enhancements (Part 3) work, and why |
| [SCALE.md](SCALE.md) | Part 2: running this at 10M patents (components, pipelines, cost, errors) and the proof-of-concept |

Status: **Parts 1, 2 and 3 are complete.** Part 1 (§2–§4) and Part 3 (§5) are implemented and measured.
Part 2 is designed in SCALE.md, and its proof-of-concept is built and measured (§6).

---

## 1. Problem statement

Examiners and attorneys judge novelty and overlap **element by element**: a claim is anticipated only
if every one of its elements is disclosed in a reference. So the tool must:

1. take a natural-language query, a claim, or a patent in the corpus;
2. return ranked patents **with the exact passage that matched**, and for claims, **each element's
   evidence** (found / weak / not found / undetermined);
3. let the user constrain results the way examiners do: by **classification code prefix**, by
   **keywords in the title or abstract**, by a **specific title**, by filing week;
4. **never hide a data gap**. If a patent's source data is defective (e.g. its claim 1 lost its
   opening words), every output that relies on it says so and marks the assessment provisional,
   pending data research. A gap is never reported as "not disclosed".

### 1.1 Inputs

**To build the index**, the system takes the dataset itself: the 64 weekly JSON files of patent
applications (title, document number, abstract, description paragraphs, claims, classification).

**To search**, a user gives one of three kinds of query, optionally with filters:

| Input | Example | How to give it |
|---|---|---|
| A description in plain English | "tire with sound absorbing foam in the cavity" | `search "<text>"` |
| A claim (pasted text) | "A wheel comprising: a rim; a disk with spokes; and a wall portion …" | `search --mode claim "<claim>"` |
| A patent in the dataset, by document number | `20240165989` | `search --mode patent --doc <number>` |
| **Filters** (with any query above, or on their own) | classification starts with `B60C`; abstract contains "tread"; title is "SPOKE"; filed June–December 2024 | `--cpc B60C`, `--cpc-exclude …`, `--kw tread --kw-field abstract` (`--kw-op all / any / phrase`), `--title SPOKE` (`--title-mode exact / phrase`), `--week-from 2024-06-01 --week-to 2024-12-31` |

Options: `--rerank` turns on two-phase search (phase 1's top 100 are reranked by a cross-encoder and
the top 30 are returned); `--index index_ft`, placed before `search`, uses the fine-tuned model; `--top-k N` sets how many
results to show; `--format json` prints the raw JSON; `--export chart.md` (or `.csv`) saves a claim chart.

### 1.2 Outputs

Every search returns a **ranked list of patents**. For each patent:

| Output | What it tells the examiner |
|---|---|
| Document number, title, classification, filing week, score | which patent it is and how strongly it matched |
| **Best match** | the exact passage that matched: a claim (a dependent claim is shown with the claims it depends on), a claim element, a description paragraph, or the abstract, with the matched words highlighted |
| **Element coverage** (claim and patent queries) | for each element of the query claim: `found`, `weak`, `not found` or `undetermined`, with the evidence passage and its similarity score. This is the element-by-element overlap check |
| **Data status and notices** | whether this patent's source data has gaps (e.g. claim 1 lost its opening words), and that its assessment is provisional until the data is researched |
| Phase-1 rank (with `--rerank`) | where the patent ranked before reranking, shown as `(was N)` |

For the search as a whole: how many patents matched, counts by classification, filing week and data
status, the pipeline steps used, and timings. A **patent** query also returns a **claim map**: each
independent claim of that patent next to the closest claims in other patents.

The same result is available as a terminal view (default), as JSON (`--format json`, or the HTTP API),
and as a Markdown or CSV claim chart (`--export`). A real example, a claim query restricted to
vehicle wheels (`--cpc B60B`), first result shown:

```
Query: claim mode · tier: moderate · 3 elements · filters: cpc_prefix=['B60B']
Pipeline: filter → dense:exact_subset → bm25 → rrf → coverage   (115 matched · 188 ms)

  1. 20240165989  VEHICLE WHEEL                                         B60B304FI   2024-05-23  score 0.0264  [2/3 found]
     Best match: claim 5 (number inferred)
       "a cylindrical rim; a disk including a central connection portion connected to a vehicle
       body, a ring-shaped connection portion connected to the rim, and a plurality of spokes
       arranged to connect the central connection po…"
       element 1 "a rim": weak — provisional: open data gaps (claim1_preamble_missing,
         claim_opening_missing)
     ! Data partial: claim 1 lost its opening words; other claims start mid-sentence; claim numbers
       inferred. Provisional, pending data research; re-examine after the update.
```

## 2. How the code addresses it

```
JSON files → load (quarantine, dedupe) → validate (data issues) → parse claims → adaptive chunks
          → local embedding (pinned model, CPU) → SQLite (+FTS5) + vectors.npy + manifest.json

query → filters first (B-tree / FTS5) → allowed set → dense (exact over allowed rows) + BM25
      → reciprocal rank fusion → roll up to patents → element coverage → data notices → JSON
```

- **Adaptive chunking.** Each claim gets as much context as it needs. Simple claims become one small
  chunk. Dependent claims carry their ancestor chain, so "wherein the content is 75 mol %" is embedded
  with the claims it refines. Multi-element claims also get one chunk per element. Description paragraphs
  are merged while on-topic and split where the topic shifts, sized by density. See DESIGN.md §3.3.
- **Hybrid search that gets faster with filters.** Filters run first on indexes built for them, and
  vector scoring only touches the allowed patents' chunks (DESIGN.md §5.1).
- **Three query modes:** `text`; `claim` (element coverage on, and chunk weights tuned so that other
  patents' claim-style text doesn't crowd out the abstracts and descriptions that disclose the claim);
  `patent` (each independent claim of a patent in the dataset becomes a query, the patent itself is
  excluded, and the output is a claim map).
- **One output contract:** a typed JSON response (`patent_search/schemas.py`). The CLI text view, the
  Markdown/CSV claim charts and the API all render that same object.

### Missing and defective data: the choice made

**Nothing is excluded.** Every record with an identifier is indexed with whatever it has. Records
without an identifier are quarantined and listed. Every defect becomes a row in `data_issues` with a
severity and a research status. Each patent gets a `data_status` (`complete` / `partial` /
`incomplete`), and every search output that relies on an affected record carries a notice that the
assessment is provisional, will be updated once data research is complete, and should be re-examined
then. Element coverage reports `undetermined` (never `not_found`) when the data that could disclose
the element is missing.

What the ingest found in the small dataset is in [§4](#4-results-part-1).

## 3. How to run (Part 1)

Requirements: Python 3.12, about 1.5 GB free disk (model + index). No GPU, no external service, no API key.
Everything, including embedding, runs locally. All commands are run from the repository folder.

**1. Get the code and install the exact library versions** (`requirements.lock` pins every package):

```bash
git clone https://github.com/sidbu546/Thinkstruct_take_home.git && cd Thinkstruct_take_home
```

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.lock
```

(or, with `uv`: `uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.lock`)

**2. Add the dataset.** Copy `UTF-8patent_data_small.zip` from the assignment into the repository
folder and unzip it there (the data is not in the repository):

```bash
unzip -q UTF-8patent_data_small.zip          # creates data/patent_data_small/
```

**3. Build the index and check it:**

```bash
.venv/bin/python -m patent_search build-index --data data/patent_data_small
```

The first build downloads the pinned model (`Alibaba-NLP/gte-modernbert-base` @ `e7f32e3`) and encodes
every chunk on CPU; see §4 for the measured time. Re-runs reuse the embedding cache.

```bash
.venv/bin/python -m patent_search verify-index
```

Try it interactively (the model loads once, then type one query after another; `help` shows examples, `quit` leaves):

```bash
.venv/bin/python -m patent_search interactive
```

One-off searches:

```bash
.venv/bin/python -m patent_search search "tire with sound absorbing foam in the cavity" --cpc B60C
.venv/bin/python -m patent_search search --mode claim "a rim; a disk with spokes; and a wall portion partially closing the air holes between the spokes"
.venv/bin/python -m patent_search search --mode patent --doc 20240165989 --export claim_chart.md
.venv/bin/python -m patent_search search "spoke" --kw carbon --kw-field abstract --cpc B60B --format json
.venv/bin/python -m patent_search search --title "SPOKE"            # filter-only: exact title
.venv/bin/python -m patent_search show 20240051333
.venv/bin/python -m patent_search data-issues --severity high
```

API (same JSON):

```bash
.venv/bin/uvicorn patent_search.api.app:app --port 8000      # then open http://localhost:8000/docs
```

Tests (use a fake hashed bag-of-words encoder on 4 weekly files, so they run in seconds without the model):

```bash
.venv/bin/python -m pytest -q
```

## 4. Results (Part 1)

These are the Part 1 measurements, taken with Part 1's original claim-mode weights. Part 3 (§5)
re-tuned those weights on the dev split.

All numbers were measured on the machine this was built on: Apple M4 (10 cores), 24 GB RAM, **CPU only**,
Python 3.12, versions in `requirements.lock`. The source for each is `index/manifest.json`,
`index/ingest_report.json` or `reports/`.

**Reproduce the Part 1 results** (after the §3 setup and `build-index`):

| Result | Command | Reproduces |
|---|---|---|
| Ingest, chunk and data-issue counts (§4.1, §4.2) | `build-index` prints them; `.venv/bin/python scripts/part1_checks.py` prints them again from `index/manifest.json` | exactly |
| Build reproducibility (§4.3) | `.venv/bin/python -m patent_search verify-index` | exactly (PASS) |
| Claim self-retrieval and latency (§4.4) | `.venv/bin/python scripts/part1_checks.py` (Part 1's original weights; add `--weights final` for the tuned ones) | self-retrieval exactly; latency depends on the machine |
| Element-coverage thresholds (§4.5) | `.venv/bin/python scripts/calibrate_coverage.py --index index` | exactly |

### 4.1 What ingest found (small dataset)

| | Count | Handling |
|---|---|---|
| Patents loaded | 640 (64 weekly files) | all indexed; 0 quarantined, 0 duplicates |
| Active claims | 10,451 | parsed: number, dependencies, depth, elements, tier |
| Canceled claims | 1,031 (from 167 items, 71 of them ranges like `1 - 5 . (canceled)`) | kept as status rows, shown in claim maps, not searched |
| Empty description paragraphs | 20,819 | structural blank lines in the source, dropped from chunking |

**Data issues** (all open, pending data research; `python -m patent_search data-issues`):

| Issue | Patents | Severity | Effect on outputs |
|---|---|---|---|
| `claim1_preamble_missing`: claim 1 starts mid-sentence | 501 | high | notice on results; claim-1 matches provisional |
| `claim_opening_missing`: a later claim starts mid-sentence (its "according to claim N" may be lost) | 511 | high | notice; such claims are treated as independent until researched |
| `description_thin`: < 50 words of description | 127 | high | coverage reports `undetermined`, not `not_found` |
| `parent_claim_unresolved`: a dependent claim cites a claim that isn't there | 36 | high | that claim is indexed without its parent's text; notice |
| `claim_number_inferred`: claim number not in source | 613 | medium | shown as "number inferred" |
| `claim_canceled` / `classification_ambiguous` | 89 / 640 | info | listed; no per-result notice |

`data_status`: **622 partial, 18 complete.** Nearly every patent has at least one
extraction defect, mostly the lost claim-1 opening, so nearly every result carries a notice. That is
intended. The defects are real and affect claim matching, so they're shown rather than hidden.

### 4.2 Adaptive chunks

| Kind | Chunks | Notes |
|---|---|---|
| claim | 10,451 | tiers: 1,549 simple · 6,479 moderate · 2,423 complex; dependents carry their ancestor chain |
| element | 6,932 | one per limitation of moderate/complex independent claims |
| description | 15,150 | merged/split by topic and density; median 125 words |
| abstract | 640 | |
| **total** | **33,173** | 5.31M tokens; 0 over the 2,048-token limit; 31,110 distinct texts |

### 4.3 Build and reproducibility

- **Embedding** runs on CPU at 9–17 chunks/s, depending on chunk length. A from-scratch build of the
  31,110 distinct texts takes **≈ 40 minutes** (estimated from those rates; this index was built across
  three resumable runs that reused the cache). Re-running with an unchanged cache takes **15 s**.
- Loading the checkpoint in float16 (the library default for this model) gave 1.1 chunks/s on CPU. The
  encoder forces **float32**, which is about 13× faster and deterministic.
- `verify-index`: **PASS**. Data hashes, chunk digest and vectors hash all match. Re-encoding a
  200-chunk sample reproduces stored vectors to max |diff| 3.3e-7 (cosine 1.000000). A build without the
  cache (`--no-cache`) is deterministic. Builds that reuse a cache written by earlier runs can differ from
  it by ~1e-7 per value because batch padding differs; that is within verify's 1e-4 tolerance.

### 4.4 Search quality and speed

**Sanity check, not an evaluation.** Pasting one of 100 random independent claims back in as a `claim`
query returns its own patent **first 95%** of the time and in the **top 5 99%** of the time. This is
easy, because the claim's own chunk is in the index. Part 3 builds the real evaluation, with the
patent's own claims masked out.

**Latency** (warm, model loaded, CPU; `scripts/part1_checks.py`, saved in `reports/part1_checks_part1.json`):

| Request | p50 | p95 |
|---|---|---|
| text query | 42 ms | 49 ms |
| text query + CPC + keyword filters | 50 ms | 54 ms |
| claim query (with element sub-queries and coverage) | 101 ms | 860 ms |
| claim query + CPC filter | 106 ms | 832 ms |
| patent mode (each independent claim as a query + claim map) | 1,205 ms | 2,041 ms |
| filters only (no query) | 1 ms | 1 ms |

Filtering first makes the vector step cheaper (exact scoring over only the allowed rows). One fix
mattered a lot: restricting FTS5 BM25 with `rowid IN (subquery)` took ~2 s per filtered query;
a `JOIN` on the same temp table gives identical results in ~3 ms.

### 4.5 Element-coverage thresholds (calibrated, no labels)

`scripts/calibrate_coverage.py` → `reports/coverage_calibration.json` (300 random claims, seed 13).
For each claim element, it takes the best cosine against:

| Compared with | Median | Share ≥ `weak` (0.744) | Share ≥ `found` (0.802) |
|---|---|---|---|
| its own patent's abstract + description (positives) | 0.897 | 99.2% | 96.4% |
| other patents, same subclass (some genuinely overlap) | 0.672 | 13.0% | 2.5% |
| patents in other subclasses (negatives) | 0.622 | 1.0% | 0.05% |

`found` = 97.5th percentile of same-subclass similarity; `weak` = 99th percentile of
other-subclass similarity. The first choice (99th percentile of other-subclass for `found`) let 13% of
same-subclass look-alikes count as "found", which is too generous for a novelty signal, so it was
tightened. These are similarity cut-offs, not legal judgments. They get re-tuned against labels in Part 3.

## 5. Part 3: evaluation & training, and two-phase search

**Enhancements picked:** (A) **evaluation and training** and (B) **two-phase search**. Why these two:
they need no extra data (the non-English sample for translation was never provided), they run
locally, and they reinforce each other. (A) is the measurement that says whether (B), the fine-tuned
model, and the Part 1 choices (adaptive chunking, hybrid fusion, chunk weights) actually help.

### 5.1 How Part 3 connects to Part 1 (one system, not a side tool)

```
query ─► filters ─► PHASE 1: hybrid retrieval (Part 1) ─► top 100 ─► PHASE 2: cross-encoder ─► top 30 ─► JSON
                    bi-encoder = base or FINE-TUNED (A)              bge-reranker-base, Mac GPU (opt-in: --rerank)
evaluation (A) ─► calls the same Engine.search ─► runs/*.jsonl ─► reports/metrics.json ─► tuned defaults in config.py
```

| Part 1 piece | What Part 3 adds there |
|---|---|
| `Engine._search` | phase 2 (`search/rerank.py`) reranks the phase-1 **top 100** and returns the **top 30**; filters are already applied, so it can only reorder (0 filter violations in 7,510 checked results) |
| `SearchRequest/Response` | schema 1.1 (additive): `rerank`, `rerank_model`, `rerank_depth`, `retrieval`, `chunk_kinds`; `signals.phase1_rank`, `signals.rerank`, `stage_ms`, `pipeline_notices` |
| `build-index` | `--chunking fixed` (baseline) and `--model models/gte-patent-ft` (fine-tuned); the engine reads the model from the index manifest |
| `config.py` | claim-mode chunk weights and rerank α/β are the **dev-tuned** values |
| coverage | thresholds calibrated per index (`<index>/coverage_calibration.json`) |
| CLI / API / text view | `--rerank`, `--rerank-model bge`, `--rerank-depth`, `--retrieval`; `(was N)` shows each result's phase-1 rank |

### 5.2 Evaluation data (silver labels, from document structure)

There are no human relevance judgments, so labels come from structure (`eval build-sets`):

- **T1, claim → patent:** an independent claim is the query; the relevant patent is its own, with **its own claim and element chunks hidden**, so it must be found through its abstract or description.
- **T2, partial claim → patent:** 2–3 elements of a claim, which is the usual partial-overlap situation.
- **T4, pairs:** (claim, own abstract) vs (claim, **hard negatives**). Hard negatives are other patents in the same CPC subclass ranked 5–50 by the engine, excluding the same family and near-duplicates.
- **T5, filters:** no result may violate a filter, before or after reranking.
- **Splits by patent family** (same title filed < 8 weeks apart, or near-identical abstracts): 384 / 128 / 128 patents, 518 families, seed 13. Dev: 234 queries; test: 306. Training: 2,649 triplets from the train split only.
- Patents with data gaps are **kept**. Every query records its source patent's `data_status`. Only 7 dev/test queries come from patents with complete data, so the complete-vs-partial breakdown is reported but too thin to conclude from.

All choices (weights, α/β, model selection) were made on **dev**. **Test** was run once per final configuration.

### 5.3 Results (test split, 306 queries; every choice made on dev)

All rows use the dev-tuned claim-mode chunk weights, except the "Part 1 weights" reference row.
Rerank rows: phase 1 hands its top 100 patents to `bge-reranker-base` (Mac GPU), which returns the
top 30 (α 0.2, β 0.2, tuned on dev).

**How to read the metrics.** Each query has one correct patent (the claim's own patent, §5.2), so
every metric asks *where in the ranking that patent appears*.

| Metric | Full form | What it means here | Range |
|---|---|---|---|
| **R@k** (R@1, R@10, R@100) | Recall at k | share of queries whose correct patent is in the top *k* results. R@1 = ranked first; R@100 = within the 100 candidates handed to reranking | 0–1, higher is better |
| **MRR@10** | Mean Reciprocal Rank at 10 | average of 1 ÷ (rank of the correct patent), counting 0 if it is below 10th. 1st → 1.0, 2nd → 0.5, 3rd → 0.33. The main metric: it rewards putting the right patent at the very top | 0–1, higher is better |
| **nDCG@10** | normalized Discounted Cumulative Gain at 10 | like MRR, but the credit for rank *r* is 1 ÷ log₂(r + 1), so it falls off more gently (3rd → 0.5) | 0–1, higher is better |
| **95% CI** | 95% confidence interval | the range the true value most likely lies in, from 1,000 bootstrap resamples of the queries. Wide intervals mean the test set is small | — |
| **Δ (paired)** | paired difference | the change in MRR@10 between two configurations *on the same queries*, with its 95% CI. If the interval includes 0, the difference is not clearly real | — |
| **Triplet accuracy** | — | share of (claim, own abstract, hard negative) triples where the claim is more similar to its own abstract than to the look-alike patent | 0–1, higher is better |
| **ROC-AUC** | Area Under the Receiver Operating Characteristic curve | probability that a randomly chosen correct pair scores higher than a randomly chosen look-alike pair; 0.5 = chance | 0.5–1, higher is better |
| **Margin** | — | mean similarity of correct pairs minus mean similarity of look-alike pairs | higher is better |
| **p50 / p95** | 50th / 95th percentile | the median query time, and the time 95% of queries finish within | lower is better |

Other abbreviations: **RRF** = Reciprocal Rank Fusion (each ranked list adds 1 ÷ (60 + rank) to an
item's score, so items high in both the dense and the keyword list win); **BM25** = Best Matching 25,
the standard keyword-ranking formula; **CPC** = Cooperative Patent Classification (codes like `B60C`);
**T1–T5** = the evaluation tasks in §5.2; **α, β** = the rerank blend weights (DESIGN.md §3B.3).

**Experiment matrix (test, 306 queries)**

| Exp | Bi-encoder | Chunking | Retrieval | Rerank | MRR@10 [95% CI] | R@1 | R@10 | R@100 |
|---|---|---|---|---|---|---|---|---|
| E0 | — | adaptive | BM25 only | — | 0.811 [0.775, 0.846] | 0.729 | 0.941 | 0.961 |
| E1 | `gte-modernbert-base` | fixed 200-word windows | dense only | — | 0.741 [0.696, 0.785] | 0.657 | 0.892 | 0.938 |
| E2 | `gte-modernbert-base` | adaptive | dense only | — | 0.744 [0.702, 0.785] | 0.667 | 0.869 | 0.918 |
| E3 | `gte-modernbert-base` | adaptive | hybrid<br>dense + BM25, fused by RRF | — | 0.825 [0.790, 0.858] | 0.752 | 0.938 | 0.967 |
| E4 | `gte-modernbert-base`<br>fine-tuned (`gte-patent-ft`) | adaptive | hybrid<br>dense + BM25, fused by RRF | — | 0.836 [0.801, 0.870] | 0.768 | 0.938 | 0.967 |
| E5 | `gte-modernbert-base` | adaptive | hybrid<br>dense + BM25, fused by RRF | `bge-reranker-base`<br>top 100 → 30 | 0.829 [0.795, 0.862] | 0.755 | 0.941 | 0.967 |
| **E6** | `gte-modernbert-base`<br>fine-tuned (`gte-patent-ft`) | adaptive | hybrid<br>dense + BM25, fused by RRF | `bge-reranker-base`<br>top 100 → 30 | 0.843 [0.809, 0.877] | 0.778 | 0.941 | 0.967 |

**Ablations and reference rows (test)**

| Exp | Bi-encoder | Chunking | Retrieval | Rerank | MRR@10 [95% CI] | R@1 | R@10 | R@100 |
|---|---|---|---|---|---|---|---|---|
| E3 · Part 1 weights | `gte-modernbert-base` | adaptive | hybrid<br>dense + BM25, fused by RRF<br>original hand-set claim-mode weights | — | 0.557 [0.515, 0.600] | 0.395 | 0.882 | 0.964 |
| E1h | `gte-modernbert-base` | fixed 200-word windows | hybrid<br>dense + BM25, fused by RRF | — | 0.815 [0.779, 0.851] | 0.739 | 0.938 | 0.964 |
| E1d | `gte-modernbert-base` | fixed 200-word windows<br>abstract + description only | dense only | — | 0.758 [0.712, 0.797] | 0.683 | 0.899 | 0.951 |
| E2d | `gte-modernbert-base` | adaptive<br>abstract + description only | dense only | — | 0.760 [0.719, 0.801] | 0.683 | 0.899 | 0.948 |

**Paired differences on the same test queries** (ΔMRR@10, paired bootstrap 95% CI)

| Change | Δ | 95% CI |
|---|---|---|
| dev-tuned chunk weights vs Part 1 weights | +0.268 | [+0.234, +0.303] |
| adaptive vs fixed chunks (dense, E2 vs E1) | +0.003 | [−0.017, +0.022] |
| adaptive vs fixed, abstract + description only | +0.002 | [−0.013, +0.016] |
| hybrid vs BM25 only (E3 vs E0) | +0.015 | [−0.013, +0.041] |
| fine-tuned vs base bi-encoder (E4 vs E3) | +0.011 | [−0.009, +0.029] |
| rerank on the base model (E5 vs E3) | +0.004 | [−0.010, +0.017] |
| rerank on the fine-tuned model (E6 vs E4) | +0.007 | [−0.005, +0.019] |
| full Part 3 pipeline vs tuned Part 1 (E6 vs E3) | +0.018 | [−0.002, +0.037] |

**Reading the tables:**
- **The dev-tuned chunk weights are the one clearly real gain** (+0.268).
- **Everything else is within the noise on test:** adaptive vs fixed chunks, hybrid vs BM25, fine-tuning (which did win on dev: +0.044 [+0.024, +0.065]), and reranking.
- **E6, the full pipeline, has the highest MRR@10 (0.843)**, but its gain over E3 (+0.018 [−0.002, +0.037]) is not quite significant.
- **Reranking on the base model (E5) barely helps** (+0.004): the right patent moved up in 21 queries and down in 11.
- **Reranking on the fine-tuned model (E6) helps slightly** (+0.007): up in 23 queries, down in 6.
- **Phase 1 already puts the right patent in the top 100 for 96.7% of queries,** which is the ceiling for phase 2.

**T4, positives vs hard negatives** (bi-encoder only, test, 300 anchors, 900 hard negatives):

| Model | triplet accuracy | ROC-AUC | mean cos pos / neg | margin |
|---|---|---|---|---|
| base gte-modernbert | 0.924 | 0.859 | 0.797 / 0.681 | 0.116 |
| fine-tuned | **0.930** | **0.910** | 0.624 / 0.352 | **0.272** |

**Training** (`train`): 1,570 triplets after a 6-per-patent cap (1,292 with a mined hard negative); 99 steps; max length 256 in training; one
epoch, batch 16, lr 2e-5, InfoNCE with in-batch plus hard negatives; CPU, seed 13; loss 0.83 → 0.23 in
24 minutes. Details are in `models/gte-patent-ft/training_card.json`.

**Phase-2 tuning (dev):**
- **Reranker used: `BAAI/bge-reranker-base`**, on the phase-1 top 100 (2 best chunks each, about 200 pairs), returning the top 30, on the Mac GPU (`mps`). On dev, α 0.2 / β 0.2 was best: MRR@10 0.952 against 0.950 for phase 1 (paired Δ +0.002, CI [−0.006, +0.011]). Giving the reranker more weight hurt (α 0.5: 0.940–0.944).

**Serving latency** (fine-tuned index, cold cache, 20 dev claim queries, `scripts/latency_bench.py`):

| Pipeline | p50 | p95 |
|---|---|---|
| phase 1 only (default; CPU) | 153 ms | 810 ms |
| phase 1 top 100 → bge rerank → top 30 (Mac GPU) | 7.3 s | 10.5 s |

### 5.4 What the results say, and what I did about them

1. **Chunk weights mattered far more than models.** The Part 1 claim-mode weights favoured
   claim/element chunks. Thousands of short, claim-shaped chunks from *other* patents then outranked
   the abstract and description that actually disclose a claim's subject matter. Tuning the weights on
   dev raised test MRR@10 from 0.557 to 0.825. Those weights are now the default.
2. **Adaptive chunking: no measurable retrieval gain on this task.** Dense adaptive first looked much
   worse on dev (0.290 vs 0.536). The ablation that searches only abstracts and descriptions showed
   the passages themselves were as good (0.823 vs 0.807 dev). The gap came from claim-chunk crowding,
   which the weight tuning fixed. Adaptive chunks are kept for what this metric can't see: claim maps,
   element coverage, and evidence that shows a dependent claim with its parents.
3. **Fine-tuning:** it beat the base model on dev (+0.044) and on every pair metric on test (triplet
   accuracy 0.924 → 0.930, ROC-AUC 0.859 → 0.910, margin 0.116 → 0.272), but its retrieval gain on
   test is inside the noise, and its element-coverage separation is a little worse (own evidence
   reaches `found` 93.6% vs 96.4%). By the rule fixed in the design (keep it if dev doesn't
   drop), it is kept as `index_ft`. The **default index stays the base model**: it performs the same on
   test, and reproducing it needs no training.
4. **Reranking** (top 100 → top 30) is wired in, tuned and tested, but its test gain is within the noise
   (+0.007 on the fine-tuned model, +0.004 on the base model) and it costs about 7 s per query even on
   the Mac GPU, so it is **opt-in** (`--rerank`). Off-the-shelf
   cross-encoders don't know patent claims; fine-tuning the cross-encoder on these triplets is the
   obvious next step (designed as E7, not built).
5. **Honest limits of the labels.** "Find the claim's own patent" rewards self-similarity. It can't
   reward finding *other* patents that genuinely overlap, which is the real goal. Human or
   examiner-citation labels are needed to measure that.

### 5.5 How to run Part 3 and reproduce the results

**Prerequisite:** the Part 1 setup and index from §3 (steps 1–3).

**Use the full two-phase pipeline.** Phase 1 retrieves; the cross-encoder reranks the top 100 on the
Apple GPU (or the CPU if there is none); the reranked top 30 are shown. Add `--index index_ft` to use
the fine-tuned model (E6); that index exists after step 4 below.

```bash
.venv/bin/python -m patent_search search --mode claim "A pneumatic tire comprising: a tread portion; a pair of bead portions; and a sound absorbing member fixed to the inner surface of the tread portion" --rerank
```

```bash
.venv/bin/python -m patent_search --index index_ft search --mode claim "A pneumatic tire comprising: a tread portion; a pair of bead portions; and a sound absorbing member fixed to the inner surface of the tread portion" --rerank
```

**Check the reported numbers (seconds).** The repository includes the evaluation sets
(`eval_data/`), the per-query rankings of every final test run (`runs/*_test.jsonl`) and all metrics
(`reports/metrics.json`). This regenerates every table in §5.3 from them, and the eval report:

```bash
.venv/bin/python scripts/readme_tables.py
```

```bash
.venv/bin/python -m patent_search eval report --split test
```

**Reproduce every number from scratch (several hours, unattended).** One script runs all steps in
order:

```bash
scripts/reproduce_part3.sh
```

| # | Step | Produces | Time (Apple M4) |
|---|---|---|---|
| 1 | build the evaluation sets | `eval_data/`: family splits, dev/test queries and answers, T4 pairs, training triplets | ~10 min |
| 2 | build the fixed-chunk baseline index | `index_fixed/` for E1, E1h, E1d | ~30 min |
| 3 | dev runs, including Part 1's original weights, and the chunk-weight grid | dev results and `reports/kind_weight_grid_dev.log` | ~25 min |
| 4 | fine-tune the bi-encoder, then index with it | `models/gte-patent-ft/`, `index_ft/` | ~25–45 min + ~45 min |
| 5 | calibrate element-coverage thresholds | `reports/coverage_calibration*.json` | ~2 min |
| 6 | dev selection check, T4 pairs on dev, rerank α/β grid on dev | `reports/rerank_tuning_final_dev.json` | ~45 min |
| 7 | all phase-1 test runs (E0–E4, ablations), T4 on test, T5 filters | `runs/E*_test.jsonl`, `reports/metrics.json` | ~25 min |
| 8 | reranked test runs E6 and E5 | `runs/E5_test.jsonl`, `runs/E6_test.jsonl` | ~1–2.5 h |
| 9 | serving latency, tables, report | `reports/latency_bench_index_ft.json`, §5.3 tables, `reports/eval_report_test.md` | ~10 min |

The chosen settings (claim-mode chunk weights, rerank α and β) are already in
`patent_search/config.py`, so steps 7–8 use them directly. Steps 3 and 6 show how they were chosen on dev.

**How close a rerun gets.** Phase 1 on CPU is deterministic: `verify-index` checks this, and the base-model
results (E0–E3) reproduce exactly. Training and GPU reranking can differ in the last decimal places
from machine to machine, so E4–E6 may move slightly, well within their confidence intervals.

**Outputs:** `eval_data/` (sets), `runs/*.jsonl` (one line per query), `reports/metrics.json` (every
metric, with breakdowns by task, tier and data status, and CIs), `reports/eval_report_test.md` (tables)
and `reports/*tuning*` (grids).

## 6. Part 2: implementation at scale, and the proof-of-concept

The design for 10M patents is in **[SCALE.md](SCALE.md)**: components, pipelines (backfill, weekly,
data research, re-embedding, query), cost, error handling, status tracking, challenges and an
implementation plan. Its sizing and cost use numbers measured by the proof-of-concept below.

**Proof-of-concept** (`poc/`; details and full results in SCALE.md §10): a local **job queue**
(SQLite, WAL) with **leases, retries with backoff and a dead state**, **parallel worker processes**
running the Part 1 ingest code on **real batch files from the large dataset zip** (read without
extracting), with corrupted-file salvage, quarantine, dedupe, data-issue tracking, fault injection and
a **live status dashboard**.

### How to run the proof-of-concept

It uses **two terminals**: one runs the dashboard (a small web server that keeps running), the other
runs the workers. A run over 20 batch files takes about 35 minutes on the M4, most of it the
embedding sample (`--embed-sample 0` skips it and takes about 15 minutes).

**Terminal 1: start the dashboard and leave it running.**

```bash
cd ~/Desktop/thinkstruct_take_home
```

```bash
.venv/bin/python -m poc.dashboard
```

It prints `Dashboard running at http://localhost:8001 …` and then waits; it does not finish on its
own. Open http://localhost:8001 in a browser; the page refreshes every 2 seconds. If it says the port
is already in use, a dashboard is already running: just open the address.

**Terminal 2: queue the work, then start the workers.** Open a new terminal tab (Cmd+T) and go to the
project folder:

```bash
cd ~/Desktop/thinkstruct_take_home
```

Start from an empty queue. A previous run's jobs are already finished, so without this the workers
would exit immediately with nothing to do. The dashboard does not need restarting; it reads the queue
fresh on every refresh.

```bash
rm -rf poc_data
```

Queue 20 batch files from the large dataset (one job per file). Running it twice adds nothing the
second time:

```bash
.venv/bin/python -m poc.enqueue --zip coding_challenge_large_data.zip --jobs job_000 --limit 20
```

Add three bad inputs to test the error handling: a file cut at 60% of its bytes, a file with duplicate
and malformed records, and a file that does not exist:

```bash
.venv/bin/python -m poc.faults --zip coding_challenge_large_data.zip --out poc_data/faults
```

Start 4 workers. `--fail-rate 0.05` makes 5% of job attempts fail on purpose, `--embed-sample 3`
embeds 3 patents per job to measure embedding speed, and `--kill-one-after 60` kills worker 0 after 60
seconds to show that its job is picked up by another worker:

```bash
.venv/bin/python -m poc.worker --workers 4 --fail-rate 0.05 --embed-sample 3 --kill-one-after 60
```

**What to watch on the dashboard:** jobs move from queued to running to succeeded; the injected
failure is retried; worker 0 turns "unresponsive" after it is killed, and its job is re-claimed once
its lease expires (30 s); the truncated file ends "partial" with its unreadable part in quarantine; the
missing file ends "dead" after 3 attempts; the data-issues and quarantine tables fill in; the
extrapolation box shows the measured throughput scaled to 10M patents.

When the workers finish, print the same summary in the terminal, then press Ctrl+C in terminal 1 to
stop the dashboard:

```bash
.venv/bin/python -m poc.status
```

### Running more workers

Choose the number with `--workers` at the start:

```bash
.venv/bin/python -m poc.worker --workers 8 --fail-rate 0.05 --embed-sample 0
```

Or add workers to a run already in progress, from a third terminal. They join the same queue straight
away, and because every job is claimed with a lease, no two workers ever process the same job:

```bash
.venv/bin/python -m poc.worker --workers 2 --embed-sample 0
```

Workers exit as soon as the queue is empty, so to watch many of them busy for longer, queue more
files (files already queued are skipped):

```bash
.venv/bin/python -m poc.enqueue --zip coding_challenge_large_data.zip --jobs job_000,job_001
```

How many to run on a 10-core, 24 GB laptop:

| Setting | Sensible number of workers | Why |
|---|---|---|
| `--embed-sample 0` | up to about 8 | each worker uses about one core for parsing and chunking |
| `--embed-sample 3` | about 4 | each worker also loads the embedding model (~600 MB) and uses 2 threads |

Beyond that, workers mostly compete for the same cores and total throughput stops improving.
`--kill-one-after` only ever kills worker 0 of the command it is given to; leave it off when adding
workers.

### Results

| Result (20 real batch files + 3 fault inputs, 4 workers) | |
|---|---|
| Jobs | 20 succeeded, 2 partial (salvaged), 1 dead (missing file, after 3 attempts); 4 retries |
| Worker killed mid-job | its job was re-claimed after the lease expired and succeeded; no duplicate rows (20,539 rows = 20,539 doc numbers) |
| Truncated file | 567 records salvaged, the rest quarantined with its byte offset |
| Real-data duplicates | 20 exact (skipped) and 8 conflicting (kept as versions, `duplicate_conflict` issue) |
| Data gaps | classification missing for 37.9%, filing date missing for 100% of the large set; flagged, not dropped |
| Throughput | ~19 patents/s per worker for parse + validate + adaptive chunking (chunking is ~99% of it); ~96 chunks and ~19.4k tokens per patent |
| 10M extrapolation | ~147 worker-hours to parse and chunk; ~1,075 GPU-hours to embed at an assumed 50k tokens/s per L4 |

## 7. Known limitations

- **Lost claim openings** (501 claim-1s, 511 later claims): the title is prepended as a stopgap, and
  some dependent claims are probably treated as independent. Fixing this needs the source XML
  (named in each record's `filename`): data research, not code.
- **Element splitting is rule-based** (`;`, transitional phrase, `wherein`), so very short elements
  ("a rim") match weakly by construction.
- **Coverage is cosine similarity**, calibrated per index without human labels; see §4.5.
- **Silver labels only** (§5.4, point 5). The off-the-shelf rerankers are not patent-trained.
- **Patent-mode latency** grows with the number of independent claims (up to 10 are used); patent mode
  doesn't rerank.
- **Not supported yet:** patents outside the corpus (no fetcher).
- **The proof-of-concept is single-host** (SQLite); it shows the job semantics and measures throughput,
  not distributed coordination, OpenSearch at 10M or GPU embedding speed.

## 8. Repository layout

```
patent_search/   ingest/ (load, normalize, claims, validate, chunk) · embed/ (encoder, cache)
                 index/ (db, build, verify) · search/ (filters, planner, query, hybrid, rerank, coverage, engine)
                 eval/ (build_sets, runner, metrics, report) · train/ (finetune)
                 render/ (text, chart) · api/app.py · cli.py · schemas.py · notices.py · config.py · manifest.py
poc/             Part 2 proof-of-concept: db, queue, enqueue, process, worker, faults, status, dashboard
scripts/         part1_checks.py · calibrate_coverage.py · tune_kind_weights.sh · tune_rerank_final.py
                 run_test_phase1.sh · run_test_rerank.sh · latency_bench.py · readme_tables.py · reproduce_part3.sh
tests/           34 tests (fake encoder and reranker; ~3 s)
eval_data/       evaluation splits, queries, answer files, pairs and training triplets (in git)
runs/            per-query rankings of the final test runs E0–E6 and ablations (in git)
reports/         metrics, evaluation report, calibration and tuning results (in git)
data/ index/ index_fixed/ index_ft/ models/ cache/ poc_data/   generated locally by the commands above (not in git)
DESIGN.md        engine and enhancements design                   SCALE.md   10M-patent design + PoC results
```
