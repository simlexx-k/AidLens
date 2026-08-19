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
  --top-k 20 \
  --max-per-evaluation 3
```

The output persists on the host at
`apps/api/benchmarks/candidates.local.jsonl`.

Candidate export intentionally oversamples the production retrieval result and
then caps the annotation pool at three passages per evaluation by default. This
prevents one long or strongly matched report from crowding the entire human
labeling pool. It does **not** change production search ranking or benchmark
ranking.

Each candidate includes both:

- `rank`: its position in the diversified annotation pool
- `retrieval_rank`: its original position in the unmodified production result

It also includes the evaluation ID, chunk ID, section, passage text, lexical
score, semantic score, and fused score. Review these passages and assign
relevance labels manually.

## 2. Build a judgment dataset

Copy `apps/api/benchmarks/judgments.template.jsonl` to a local judgment file and
replace the placeholder evaluation ID with human-reviewed evidence targets.
Create one JSON object per query:

```json
{"query_id":"q001","query":"What interventions improved household resilience to food insecurity?","judgments":[{"evaluation_id":"REAL_AIDDATA_ID","section":"findings","relevance":3}]}
```

Keep at least one relevant judgment per query. Add multiple judgments when more
than one evaluation or section answers the question. Add `anchor_text` when a
section-level label would mark unrelated chunks as relevant.

## 3. Compare retrieval modes

Raw production ranking:

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
- unique evaluations at K
- duplicate share at K

The first live four-query baseline with `BAAI/bge-base-en-v1.5` was:

| Mode | Recall@10 | MRR | nDCG@10 |
| --- | ---: | ---: | ---: |
| Lexical | 0.125 | 0.375 | 0.075385 |
| Semantic | 0.750 | 0.791667 | 0.615354 |
| Hybrid | 0.791667 | 0.875 | 0.631609 |

This establishes hybrid as the current baseline while also showing that semantic
retrieval is responsible for most of the improvement over lexical search.

## 4. Run a diversity experiment

V0.5 can cap repeated passages from one evaluation without changing the default
search behavior. For example, compare the same benchmark with at most three
passages from one evaluation:

```bash
docker compose run --rm api \
  python -m app.cli benchmark \
  benchmarks/judgments.local.jsonl \
  --modes lexical,semantic,hybrid \
  --top-k 10 \
  --max-per-evaluation 3 \
  --output benchmarks/report-diverse.local.json
```

Compare relevance metrics together with `mean_unique_evaluations_at_k` and
`mean_duplicate_share_at_k`. Do not adopt a diversity cap as the default solely
because it increases variety; it should preserve or improve relevance quality.

The web search UI exposes the same optional diversity control for manual review.

## 5. Prepare AidRanker data

The candidate export is the starting pool for cross-encoder/reranker training.
After human labeling, convert each query-candidate pair into a supervised record
with its graded relevance. Retain hard negatives from high-ranked but irrelevant
lexical and semantic candidates; these are especially valuable for reranker
training.

Do not tune fusion weights or train AidRanker against the same queries used for
final evaluation. The four-query benchmark is a smoke-test baseline, not a
training set. Expand it and split judgments into development and held-out test
sets before supervised ranking work.

## Scaling ingestion

When moving through new archive pages, avoid re-fetching records already in the
local corpus:

```bash
docker compose run --rm api \
  python -m app.cli ingest --pages 10 --start-page 2 --skip-existing
```

Omit `--skip-existing` when an existing evaluation intentionally needs metadata
or chunker refresh.
