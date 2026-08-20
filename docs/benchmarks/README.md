# AidLens retrieval benchmarks

AidLens uses an offline, human-judged benchmark for comparing lexical, semantic,
and hybrid retrieval before AidRanker is trained.

Benchmark working files live under `apps/api/benchmarks`. That directory is
visible inside the API container as `/app/benchmarks` because the development
Compose stack bind-mounts `apps/api` to `/app`. Local files ending in
`.local.json` or `.local.jsonl` are ignored by Git.

## Ground truth: evaluation, section, and optional passage anchor

Chunk UUIDs change whenever a report is re-chunked. Benchmark labels therefore
reference the stable AidData evaluation ID plus an optional normalized report
section. A judgment without a section treats any passage from that evaluation as
relevant. A section-specific judgment requires a passage from that section.

V0.5 adds an optional `anchor_text` field for cases where an evaluation/section
contains both relevant and irrelevant chunks. The anchor is normalized for
whitespace and case, then matched as a substring of the retrieved passage. Use a
short, distinctive phrase copied from the relevant evidence rather than an
entire chunk.

Example:

```json
{"evaluation_id":"PA0218BQ","section":"findings","anchor_text":"instrumental in supporting transitory food insecure households","relevance":3}
```

Relevance is graded:

- `3`: directly answers the evidence question
- `2`: clearly relevant supporting evidence
- `1`: marginally relevant/contextual
- `0`: irrelevant; preserve as a hard negative when it was highly ranked

Only human-reviewed judgments should be treated as benchmark ground truth.
Retrieved candidates are not automatically positive examples.

## 1. Scale the corpus carefully

Use resumable ingestion for new archive pages:

```bash
docker compose run --rm api \
  python -m app.cli ingest --pages 10 --start-page 2 --skip-existing
```

Before embedding a newly expanded corpus, inspect both aggregate quality and
record-level anomalies:

```bash
docker compose run --rm api python -m app.cli corpus-report
docker compose run --rm api python -m app.cli corpus-audit
```

The audit names future publication years, metadata-only records without text,
and duplicate-title groups. Publication-year parsing prefers a trailing title
year when available and ignores future year-like program horizons such as
"Vision 2050".

Refresh a single problematic record without re-fetching an entire page range:

```bash
docker compose run --rm api \
  python -m app.cli refresh-evaluation AIDDATA_EXTERNAL_ID
```

Refreshing a text-backed evaluation recreates its chunks and therefore removes
that evaluation's old embeddings. Re-embed missing chunks afterward.

Live V0.6 scale validation reached 110 evaluations and 18,750 v2 chunks. The
initial audit identified `PA00ZSBS` as incorrectly parsed with publication year
2050 from its Mongolia Vision 2050 title. A targeted refresh corrected the
corpus maximum publication year to 2024 and cleared the future-year quality
flag. `PA0213MZ` remains a metadata-only Ghana T2E+ record because the AidData
entry has no plaintext source URL. Two duplicate-title groups remain retained;
equal titles with distinct archive IDs are not treated as sufficient evidence
for deletion without content-level confirmation.

## 2. Generate and label the V0.6 candidate pool

V0.6 includes `apps/api/benchmarks/queries.v1.jsonl`: 30 questions balanced
across six evidence intents:

- intervention outcomes
- sustainability risks
- recommendations
- evaluation methods
- implementation factors
- transferability context

The first 30-query candidate set was generated from hybrid retrieval:

```bash
docker compose run --rm api \
  python -m app.cli export-ranking-candidates \
  benchmarks/queries.v1.jsonl \
  --output benchmarks/candidates-v1.local.jsonl \
  --mode hybrid \
  --top-k 20 \
  --max-per-evaluation 3
```

Review every candidate and replace `relevance: null` with a value from 0 to 3.
Compilation is intentionally strict: a partially labeled query is rejected, and
a query with no positive evidence is not silently admitted to the benchmark.

Compile both evaluation truth and future AidRanker supervision from the same
labels:

```bash
docker compose run --rm api \
  python -m app.cli compile-labels \
  benchmarks/candidates-v1.local.jsonl \
  --judgments-output benchmarks/judgments-v1.local.jsonl \
  --ranker-output benchmarks/ranker-v1.local.jsonl
```

Positive passages become anchor-aware benchmark judgments. Relevance-0 passages
remain in the ranker dataset as hard negatives.

## 3. V0.6 benchmark result and selection-bias caveat

The first 30-query diversified benchmark with `BAAI/bge-base-en-v1.5`,
`top_k=10`, and `max_per_evaluation=3` produced:

| Mode | Recall@10 | MRR | nDCG@10 |
| --- | ---: | ---: | ---: |
| Lexical | 0.092174 | 0.286111 | 0.117938 |
| Semantic | 0.579470 | 0.944444 | 0.613957 |
| Hybrid | 0.584814 | 0.961111 | 0.614439 |

These numbers are useful as a retrieval smoke test, but they are not yet a fair
model-selection benchmark. The human judgments were compiled from candidates
surfaced by hybrid retrieval only. That means the positive set is conditioned on
one of the systems being evaluated, which can undercount relevant lexical-only
or semantic-only passages and structurally advantage the source system.

Do not use the V0.6 numbers alone to select fusion weights or claim that hybrid
is superior to semantic retrieval.

## 4. V0.7 pooled judging

V0.7 removes the hybrid-only candidate-pool dependency. Build a judging pool from
independent lexical, semantic, and hybrid ranked lists:

```bash
docker compose run --rm api \
  python -m app.cli export-pooled-candidates \
  benchmarks/queries.v1.jsonl \
  --output benchmarks/candidates-v2-pooled.local.jsonl \
  --modes lexical,semantic,hybrid \
  --per-mode-k 20 \
  --max-per-evaluation 5
```

The pooled export:

- retrieves each mode independently using the same production search engine
- deduplicates candidates by chunk UUID
- rotates mode traversal by rank depth so one configured retriever does not win
  every tie
- applies the annotation-only per-evaluation cap after cross-mode pooling
- records `retrieval_modes` and `mode_ranks` for every pooled passage

The benchmark-pool cap defaults to five passages per evaluation, intentionally
higher than the product-facing three-passage diversity cap. Product presentation
and ground-truth discovery are different concerns.

Reuse labels from the already reviewed V0.6 candidate pool before doing new
human work:

```bash
docker compose run --rm api \
  python -m app.cli carry-forward-labels \
  benchmarks/candidates-v2-pooled.local.jsonl \
  --previous benchmarks/candidates-v1.local.jsonl \
  --output benchmarks/candidates-v2-pooled-seeded.local.jsonl
```

The command reports how many query/chunk labels were copied and how many newly
surfaced passages still require review. Review only the remaining
`relevance: null` candidates, then compile the pooled benchmark:

```bash
docker compose run --rm api \
  python -m app.cli compile-labels \
  benchmarks/candidates-v2-pooled-seeded.local.jsonl \
  --judgments-output benchmarks/judgments-v2-pooled.local.jsonl \
  --ranker-output benchmarks/ranker-v2-pooled.local.jsonl
```

Now rerun the fair comparison:

```bash
docker compose run --rm api \
  python -m app.cli benchmark \
  benchmarks/judgments-v2-pooled.local.jsonl \
  --modes lexical,semantic,hybrid \
  --top-k 10 \
  --max-per-evaluation 3 \
  --output benchmarks/report-v2-pooled-diverse3.local.json
```

Reports contain overall and per-query-family:

- Recall@K
- Mean Reciprocal Rank (MRR)
- nDCG@K
- unique evaluations at K
- duplicate share at K

## 5. Prepare AidRanker only after pooled benchmark validation

The compiled ranker file is the starting point for a cross-encoder/reranker.
Do not train or select AidRanker against the hybrid-only V0.6 judging pool.
First finish pooled judgments, rerun lexical/semantic/hybrid against the pooled
truth, and then split queries by family into development and held-out test sets.
Retain high-ranked relevance-0 candidates from all retrieval modes as hard
negatives.
