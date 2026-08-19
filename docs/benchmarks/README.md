# AidLens retrieval benchmarks

AidLens uses an offline, human-judged benchmark for comparing lexical, semantic,
and hybrid retrieval before AidRanker is trained.

Benchmark working files live under `apps/api/benchmarks`. That directory is
visible inside the API container as `/app/benchmarks` because the development
Compose stack bind-mounts `apps/api` to `/app`. Local files ending in
`.local.json` or `.local.jsonl` are ignored by Git.

## Ground truth: evaluation, section, and passage anchor

Chunk UUIDs change whenever a report is re-chunked. Benchmark labels therefore
reference the stable AidData evaluation ID plus an optional normalized report
section. An optional `anchor_text` makes a judgment passage-specific when one
large section contains both relevant and irrelevant chunks.

Example:

```json
{"evaluation_id":"PA0218BQ","section":"findings","anchor_text":"instrumental in supporting transitory food insecure households","relevance":3}
```

Relevance is graded:

- `3`: directly answers the evidence question
- `2`: clearly relevant supporting evidence
- `1`: marginally relevant/contextual
- `0`: irrelevant hard negative in candidate-labeling files only

Only human-reviewed labels become benchmark truth. Retrieved candidates are not
automatically positive examples.

## V0.5 product baseline

The first four-query benchmark with `BAAI/bge-base-en-v1.5` established hybrid
retrieval as the strongest uncapped baseline:

| Mode | Recall@10 | MRR | nDCG@10 | Unique evaluations@10 | Duplicate share@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lexical | 0.125 | 0.375 | 0.075385 | 0.5 | 0.291667 |
| Semantic | 0.750 | 0.791667 | 0.615354 | 3.0 | 0.700 |
| Hybrid | 0.791667 | 0.875 | 0.631609 | 3.25 | 0.675 |

With `max_per_evaluation=3`, hybrid improved to Recall@10 `0.916667`, MRR
`0.875`, nDCG@10 `0.682006`, mean unique evaluations@10 `4.5`, and duplicate
share@10 `0.475`. The user-facing search UI therefore defaults to a maximum of
three passages per evaluation while the API still permits uncapped experiments.

The sample is intentionally small. Treat these numbers as an engineering
baseline, not a final model-selection result.

## 1. Expand the corpus first

The initial benchmark was produced from a ten-evaluation corpus. Before labeling
the V0.6 seed set, ingest a broader slice of the archive and embed new chunks:

```bash
docker compose run --rm api \
  python -m app.cli ingest --pages 10 --start-page 2 --skip-existing

docker compose run --rm api \
  python -m app.cli embed --batch-size 16
```

Use `corpus-report` after each batch to monitor section coverage, stale chunker
versions, and embedding coverage.

## 2. Generate the V0.6 candidate pool

`apps/api/benchmarks/queries.v1.jsonl` contains 30 seed questions across six
families:

- `intervention_outcomes`
- `sustainability_risks`
- `recommendations`
- `evaluation_methods`
- `implementation_factors`
- `transferability_context`

Generate a diversified hybrid candidate pool:

```bash
docker compose run --rm api \
  python -m app.cli export-ranking-candidates \
  benchmarks/queries.v1.jsonl \
  --output benchmarks/candidates-v1.local.jsonl \
  --mode hybrid \
  --top-k 20 \
  --max-per-evaluation 3
```

Each candidate preserves its original production `retrieval_rank` as well as its
annotation-pool rank. Query families are preserved through the labeling and
benchmark pipeline.

## 3. Label every candidate 0-3

Human reviewers should set every candidate's `relevance` field to `0`, `1`, `2`,
or `3`. Do not leave partially reviewed candidate pools as benchmark input.

A query with no genuine positive evidence should not be forced into the benchmark.
Either broaden the corpus and regenerate its pool or remove that query from the
judged set.

## 4. Compile benchmark truth and AidRanker records

V0.6 compiles one fully reviewed candidate pool into two outputs:

```bash
docker compose run --rm api \
  python -m app.cli compile-labels \
  benchmarks/candidates-v1.local.jsonl \
  --judgments-output benchmarks/judgments-v1.local.jsonl \
  --ranker-output benchmarks/ranker-v1.local.jsonl
```

The compiler is deliberately strict:

- every candidate must have a `0-3` label
- every retained benchmark query must have at least one positive
- positive candidates become anchor-aware stable benchmark judgments
- all labeled candidates, including `0` labels, become AidRanker training records
- high-ranked irrelevant passages are preserved as hard negatives

## 5. Benchmark by mode and query family

```bash
docker compose run --rm api \
  python -m app.cli benchmark \
  benchmarks/judgments-v1.local.jsonl \
  --modes lexical,semantic,hybrid \
  --top-k 10 \
  --max-per-evaluation 3 \
  --output benchmarks/report-v1.local.json
```

Reports contain overall mode summaries, per-query results, and V0.6 summaries by
query family. This lets AidLens distinguish, for example, strong method retrieval
from weak transferability retrieval instead of hiding both inside one mean score.

## 6. Split before training AidRanker

Do not train and evaluate a reranker on the same queries. Once enough questions
are labeled, split by `query_id` rather than by individual passages so passages
from one question cannot leak across train and test sets.

A practical next threshold is at least 30 reviewed queries for pipeline testing,
then a larger corpus of judgments before treating AidRanker results as robust.
The V0.6 ranker JSONL is preparation data, not authorization to train a model on
an undersized benchmark.
