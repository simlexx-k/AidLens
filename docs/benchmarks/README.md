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

## 2. Generate the V0.6 candidate pool

V0.6 includes `apps/api/benchmarks/queries.v1.jsonl`: 30 questions balanced
across six evidence intents:

- intervention outcomes
- sustainability risks
- recommendations
- evaluation methods
- implementation factors
- transferability context

Generate a diversified human-review pool:

```bash
docker compose run --rm api \
  python -m app.cli export-ranking-candidates \
  benchmarks/queries.v1.jsonl \
  --output benchmarks/candidates-v1.local.jsonl \
  --mode hybrid \
  --top-k 20 \
  --max-per-evaluation 3
```

Candidate export oversamples production retrieval and caps the annotation pool
at three passages per evaluation. Each candidate retains its original
`retrieval_rank`, query family, evaluation ID, section, passage text, lexical
score, semantic score, fused score, and a nullable `relevance` field.

## 3. Label every candidate 0-3

Review every candidate and replace `relevance: null` with a value from 0 to 3.
V0.6 compilation is intentionally strict: a partially labeled query is rejected,
and a query with no positive evidence is not silently admitted to the benchmark.

Once the pool is fully reviewed, compile both evaluation truth and future
AidRanker supervision from the same labels:

```bash
docker compose run --rm api \
  python -m app.cli compile-labels \
  benchmarks/candidates-v1.local.jsonl \
  --judgments-output benchmarks/judgments-v1.local.jsonl \
  --ranker-output benchmarks/ranker-v1.local.jsonl
```

Positive passages become anchor-aware benchmark judgments. Relevance-0 passages
remain in the ranker dataset as hard negatives.

## 4. Compare retrieval modes

Raw ranking:

```bash
docker compose run --rm api \
  python -m app.cli benchmark \
  benchmarks/judgments-v1.local.jsonl \
  --modes lexical,semantic,hybrid \
  --top-k 10 \
  --output benchmarks/report-v1-raw.local.json
```

Product-facing diversified ranking:

```bash
docker compose run --rm api \
  python -m app.cli benchmark \
  benchmarks/judgments-v1.local.jsonl \
  --modes lexical,semantic,hybrid \
  --top-k 10 \
  --max-per-evaluation 3 \
  --output benchmarks/report-v1-diverse3.local.json
```

Reports contain overall and per-query-family:

- Recall@K
- Mean Reciprocal Rank (MRR)
- nDCG@K
- unique evaluations at K
- duplicate share at K

The first live four-query baseline with `BAAI/bge-base-en-v1.5` was:

| Mode | Recall@10 | MRR | nDCG@10 |
| --- | ---: | ---: | ---: |
| Lexical | 0.125 | 0.375 | 0.075385 |
| Semantic | 0.750 | 0.791667 | 0.615354 |
| Hybrid | 0.791667 | 0.875 | 0.631609 |

With a three-passage-per-evaluation cap, the four-query hybrid smoke test
improved to Recall@10 `0.916667`, MRR `0.875`, and nDCG@10 `0.682006`, while
mean unique evaluations rose from `3.25` to `4.5` and duplicate share fell from
`0.675` to `0.475`.

V0.6 deepens the retrieval pool before applying this cap so heavily concentrated
queries are less likely to return fewer than the requested `top_k`.

## 5. Prepare AidRanker only after benchmark expansion

The compiled ranker file is the starting point for a cross-encoder/reranker. Do
not train AidRanker on the original four-query smoke test. Expand human judgments,
split queries into development and held-out test sets, and retain high-ranked
relevance-0 lexical/semantic/hybrid candidates as hard negatives.
