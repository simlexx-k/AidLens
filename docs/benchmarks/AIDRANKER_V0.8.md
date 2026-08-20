# AidLens V0.8 — AidRanker V1

V0.8 introduces the first supervised reranking experiment after V0.7 established a system-neutral pooled benchmark.

## Baseline decision

The fair pooled 30-query benchmark (`top_k=10`, `max_per_evaluation=3`) established semantic retrieval as the primary candidate generator:

- lexical: Recall@10 0.074667, MRR 0.286111, nDCG@10 0.117487
- semantic: Recall@10 0.547609, MRR 0.944444, nDCG@10 0.620252
- hybrid: Recall@10 0.539805, MRR 0.961111, nDCG@10 0.609704

Hybrid remains a comparison arm because it occasionally improves first-hit ranking, but semantic has the best overall recall and nDCG.

## Experiment design

- query-grouped 18 train / 6 dev / 6 held-out test split
- every evidence family contributes 3 train / 1 dev / 1 test query
- relevance 0–3 is retained as graded supervision
- relevance-0 passages remain hard negatives
- default model: `cross-encoder/ms-marco-MiniLM-L6-v2`
- one epoch is the initial AidRanker V1 training configuration
- production serving remains unchanged during V0.8

## Leakage controls

- split unit is `query_id`, never an individual passage
- the held-out test set is not used for epoch, loss, learning-rate, model, or fusion-weight selection
- dev uses one global fusion alpha only; no family-specific calibration
- held-out test evaluation uses a fixed-alpha command that performs no sweep or model selection

## Recompile pooled ranker records

Compile from the fully reviewed pooled candidate file:

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

The 638-record dataset yields 18 train / 6 dev / 6 test queries.

## Train

The `ml` extra uses `sentence-transformers[train]` so CrossEncoder training dependencies such as Hugging Face datasets/accelerate are installed.

```bash
docker compose run --rm api \
  python -m app.ranker_cli train \
  benchmarks/ranker-v2-split.local/train.jsonl \
  --output-dir models/aidranker-v1.local \
  --epochs 1 \
  --batch-size 8
```

## Initial dev result

Pure AidRanker reranking versus semantic baseline on the six dev queries:

- semantic: any-positive Recall@10 0.607664, MRR 0.916667, nDCG@10 0.613312
- AidRanker-only: any-positive Recall@10 0.574627, MRR 1.000000, nDCG@10 0.693929
- semantic candidate any-positive recall ceiling: 0.980392

Pure reranking materially improves first-hit and graded ranking quality but reduces broad any-positive recall, motivating fusion calibration.

## Evidence-aware metrics

The V0.8 evaluator reports:

- `recall_any_at_k`: relevance >= 1
- `recall_supporting_at_k`: relevance >= 2
- `recall_direct_at_k`: relevance = 3
- `graded_recall_at_k`: fraction of total 0–3 relevance mass recovered in top K
- MRR and nDCG@K
- unique evaluations and duplicate share
- tiered candidate-recall ceilings

## Dev fusion calibration

The dev-only command is:

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

Semantic and AidRanker scores are min-max normalized within each query before fusion. `alpha=0` is semantic-only and `alpha=1` is AidRanker-only.

Selection requires supporting recall >= baseline, direct recall >= baseline, MRR >= baseline, and mean duplicate share <= baseline + 0.05; among feasible weights, dev nDCG is maximized.

### Frozen V0.8 configuration

The six-query dev calibration selected **alpha = 0.50**.

Compared with semantic-only on dev:

- any-positive Recall@10: 0.607664 -> 0.649876
- supporting Recall@10: 0.637210 -> 0.681655
- direct-answer Recall@10: 0.481902 -> 0.567761
- graded Recall@10: 0.611341 -> 0.665029
- MRR: 0.916667 -> 0.916667
- nDCG@10: 0.613312 -> 0.676883
- mean unique evaluations@10: 6.666667 -> 7.000000
- mean duplicate share@10: 0.333333 -> 0.300000

This configuration satisfies every dev selection constraint. Alpha 0.75 and 1.0 are rejected because they lose supporting-evidence recall relative to baseline.

The configuration is now frozen as:

```text
candidate generator: semantic (BAAI/bge-base-en-v1.5)
reranker: models/aidranker-v1.local
fusion alpha: 0.50
top_k: 10
```

## One-shot held-out evaluation

Do not call `sweep-fusion` on test data. Use the fixed-alpha evaluator:

```bash
docker compose run --rm api \
  python -m app.ranker_cli evaluate-fusion \
  benchmarks/ranker-v2-split.local/test.jsonl \
  --model-path models/aidranker-v1.local \
  --candidate-mode semantic \
  --alpha 0.5 \
  --top-k 10 \
  --output benchmarks/aidranker-v1-test-fusion.local.json
```

The command evaluates exactly one already-selected alpha and emits baseline versus fused metrics without any test-time selection object.

## Production gate

Do not integrate AidRanker into serving until the one-shot held-out test is reviewed. The held-out result is for final validation, not another tuning cycle.
