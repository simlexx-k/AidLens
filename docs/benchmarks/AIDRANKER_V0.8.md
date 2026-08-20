# AidLens V0.8 — AidRanker V1

V0.8 introduces the first supervised reranking experiment after V0.7 established a system-neutral pooled benchmark.

## Baseline decision

The fair pooled 30-query benchmark (`top_k=10`, `max_per_evaluation=3`) established semantic retrieval as the primary candidate generator:

- lexical: Recall@10 0.074667, MRR 0.286111, nDCG@10 0.117487
- semantic: Recall@10 0.547609, MRR 0.944444, nDCG@10 0.620252
- hybrid: Recall@10 0.539805, MRR 0.961111, nDCG@10 0.609704

Hybrid remains a comparison arm because it occasionally improves first-hit ranking, but semantic has the best overall recall and nDCG.

## AidRanker V1 scope

1. Split ranker supervision by query, never by individual passage.
2. Preserve all six evidence families in train/dev/test.
3. Use 18 train, 6 dev, and 6 held-out test queries (3/1/1 per family).
4. Train a local SentenceTransformers CrossEncoder; no paid API dependency.
5. Treat relevance 0–3 as graded supervision, with relevance 0 retained as hard negatives.
6. Calibrate model/fusion choices on dev only.
7. Evaluate the frozen configuration once on held-out test queries.
8. Do not integrate the reranker into production search until held-out ranking quality improves without losing strong-evidence recall.

## Leakage controls

- Split unit is `query_id`.
- A query may occur in exactly one split.
- Split generation is deterministic from a supplied seed.
- Family balance is checked and reported.
- The held-out test set is not used for threshold, epoch, fusion-weight, or model selection.

## Initial model

Default model: `cross-encoder/ms-marco-MiniLM-L6-v2`.

This is intentionally small enough for local experimentation while providing a strong pretrained passage-ranking initialization. V0.8 is an experiment and data pipeline first; production inference integration is deferred until benchmark validation.

## Recompile pooled ranker records

V0.8 preserves `retrieval_modes` and `mode_ranks` in the ranker JSONL so reranking can be evaluated against the exact semantic first-stage candidate set.

**Important:** compile from the fully reviewed pooled file, not the raw pooled export. The required input is `candidates-v2-pooled-labeled.local.jsonl`; every candidate must already have a `relevance` value from 0 to 3.

Optional preflight check:

```bash
python3 - <<'PY'
import json
from pathlib import Path

path = Path('apps/api/benchmarks/candidates-v2-pooled-labeled.local.jsonl')
items = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
records = [c for item in items for c in item['candidates']]
unlabeled = [c for c in records if c.get('relevance') is None]
print(f'queries={len(items)} records={len(records)} unlabeled={len(unlabeled)}')
assert not unlabeled, 'Use the reviewed labeled pooled candidate file before compile-labels.'
PY
```

Expected for V0.7 pooled labels:

```text
queries=30 records=638 unlabeled=0
```

Compile:

```bash
docker compose run --rm api \
  python -m app.cli compile-labels \
  benchmarks/candidates-v2-pooled-labeled.local.jsonl \
  --judgments-output benchmarks/judgments-v2-pooled.local.jsonl \
  --ranker-output benchmarks/ranker-v2-pooled.local.jsonl
```

## Split

```bash
docker compose run --rm api \
  python -m app.ranker_cli split \
  benchmarks/ranker-v2-pooled.local.jsonl \
  --output-dir benchmarks/ranker-v2-split.local \
  --seed 42
```

For the current 638-record dataset, seed 42 yields:

- train: 18 queries / 406 records
- dev: 6 queries / 113 records
- test: 6 queries / 119 records

Each of the six families contributes 3 train, 1 dev, and 1 test query.

## Train

The ML-enabled API image is required. The `ml` extra uses `sentence-transformers[train]`, which installs the supported SentenceTransformers training stack including Hugging Face datasets/accelerate.

```bash
docker compose run --rm api \
  python -m app.ranker_cli train \
  benchmarks/ranker-v2-split.local/train.jsonl \
  --output-dir models/aidranker-v1.local \
  --epochs 1 \
  --batch-size 8
```

The 0–3 relevance labels are scaled to 0–1 soft targets. One epoch is the initial default because CrossEncoders can overfit quickly on small datasets.

## Evaluate on dev first

```bash
docker compose run --rm api \
  python -m app.ranker_cli evaluate \
  benchmarks/ranker-v2-split.local/dev.jsonl \
  --model-path models/aidranker-v1.local \
  --candidate-mode semantic \
  --top-k 10 \
  --output benchmarks/aidranker-v1-dev.local.json
```

The evaluator reports both backward-compatible any-positive recall and stronger evidence-aware metrics:

- `recall_any_at_k`: relevance >= 1
- `recall_supporting_at_k`: relevance >= 2
- `recall_direct_at_k`: relevance = 3
- `graded_recall_at_k`: fraction of total 0–3 relevance mass recovered in top K
- MRR, nDCG@K, unique evaluations, duplicate share
- any/supporting/direct/graded candidate-recall ceilings

## Dev-only fusion calibration

The initial dev run improved MRR/nDCG but reduced any-positive Recall@10. Before touching the held-out test set, sweep one global semantic/AidRanker weight on dev:

```bash
docker compose run --rm api \
  python -m app.ranker_cli sweep-fusion \
  benchmarks/ranker-v2-split.local/dev.jsonl \
  --model-path models/aidranker-v1.local \
  --candidate-mode semantic \
  --alphas 0,0.25,0.5,0.75,1 \
  --top-k 10 \
  --output benchmarks/aidranker-v1-dev-fusion.local.json
```

Scores are min-max normalized within each query before fusion. `alpha=0` is semantic-only and `alpha=1` is AidRanker-only. The selector maximizes dev nDCG while requiring:

- supporting-recall >= semantic baseline
- direct-recall >= semantic baseline
- MRR >= semantic baseline
- mean duplicate share no more than 0.05 above semantic baseline

This is one global alpha only. Do not fit family-specific weights on the six-query dev split.

## Held-out test

Only after the model and fusion alpha are frozen should the held-out test be evaluated. Do not use test results to adjust epochs, learning rate, loss, or alpha.

The pure-reranker diagnostic remains:

```bash
docker compose run --rm api \
  python -m app.ranker_cli evaluate \
  benchmarks/ranker-v2-split.local/test.jsonl \
  --model-path models/aidranker-v1.local \
  --candidate-mode semantic \
  --top-k 10 \
  --output benchmarks/aidranker-v1-test.local.json
```

A frozen fused test evaluation should use the alpha selected on dev; production integration remains deferred until that held-out result is reviewed.
