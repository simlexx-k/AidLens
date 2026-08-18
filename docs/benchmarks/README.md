# AidLens retrieval benchmarks

V0.4 introduces an offline, human-judged benchmark for comparing lexical,
semantic, and hybrid retrieval before AidRanker is trained.

Benchmark working files live under `apps/api/benchmarks`. That directory is
visible inside the API container as `/app/benchmarks` because the development
Compose stack bind-mounts `apps/api` to `/app`. Local files ending in
`.local.json` or `.local.jsonl` are ignored by Git.

## Why judgments use evaluation ID + section

Chunk UUIDs change whenever a report is re-chunked. Benchmark labels therefore
reference the stable AidData evaluation ID plus an optional normalized report
section. A judgment without a section treats any passage from that evaluation as
relevant. A section-specific judgment requires a passage from that section.

Relevance is graded:

- `3`: directly answers the evidence question
- `2`: clearly relevant supporting evidence
- `1`: marginally relevant/contextual

Only human-reviewed judgments should be treated as benchmark ground truth.
Retrieved candidates are not automatically positive examples.

## 1. Generate candidate pools

Start with `apps/api/benchmarks/queries.example.jsonl`, then run:

```bash
docker compose run --rm api \
  python -m app.cli export-ranking-candidates \
  benchmarks/queries.example.jsonl \
  --output benchmarks/candidates.local.jsonl \
  --mode hybrid \
  --top-k 20
```

The output persists on the host at
`apps/api/benchmarks/candidates.local.jsonl`. Each candidate includes the
evaluation ID, chunk ID, section, passage text, lexical score, semantic score,
and fused rank. Review these passages and assign relevance labels manually.

## 2. Build a judgment dataset

Copy `apps/api/benchmarks/judgments.template.jsonl` to a local judgment file and
replace the placeholder evaluation ID with human-reviewed evidence targets.
Create one JSON object per query:

```json
{"query_id":"q001","query":"What interventions improved household resilience to food insecurity?","judgments":[{"evaluation_id":"REAL_AIDDATA_ID","section":"findings","relevance":3}]}
```

Keep at least one relevant judgment per query. Add multiple judgments when more
than one evaluation or section answers the question.

## 3. Compare retrieval modes

```bash
docker compose run --rm api \
  python -m app.cli benchmark \
  benchmarks/judgments.local.jsonl \
  --modes lexical,semantic,hybrid \
  --top-k 10 \
  --output benchmarks/report.local.json
```

The report contains per-query and mean:

- Recall@K
- Mean Reciprocal Rank (MRR)
- nDCG@K

Do not tune fusion weights or train AidRanker against the same queries used for
final evaluation. As the benchmark grows, split judgments into development and
held-out test sets.

## 4. Prepare AidRanker data

The candidate export is the starting pool for cross-encoder/reranker training.
After human labeling, convert each query-candidate pair into a supervised record
with its graded relevance. Retain hard negatives from high-ranked but irrelevant
lexical and semantic candidates; these are especially valuable for reranker
training.

## Scaling ingestion

When moving through new archive pages, avoid re-fetching records already in the
local corpus:

```bash
docker compose run --rm api \
  python -m app.cli ingest --pages 10 --start-page 2 --skip-existing
```

Omit `--skip-existing` when an existing evaluation intentionally needs metadata
or chunker refresh.
