# AidLens V0.9 — AidRanker production serving

V0.9 integrates the V0.8-validated semantic + AidRanker pipeline into evidence search without changing the frozen ranking configuration.

## Frozen ranking contract

The held-out V0.8 test validated:

```text
candidate generator: semantic (BAAI/bge-base-en-v1.5)
reranker: AidRanker V1
fusion alpha: 0.50
top_k: 10
```

The serving alpha is a code constant (`FROZEN_AIDRANKER_ALPHA = 0.50`), not an environment variable. V0.9 must not retune it.

## Serving behavior

- `mode=auto`, `rerank=auto`, and AidRanker enabled: semantic retrieval -> AidRanker fusion -> evidence-spread cap.
- explicit `mode=semantic`, `rerank=auto`: the same reranked path when AidRanker is enabled.
- `rerank=disabled`: first-stage retrieval only.
- `rerank=aidranker`: require AidRanker; return 503 rather than silently falling back.
- explicit lexical or hybrid retrieval is not reranked.
- Auto reranking is fail-open by default: inference/model-load failures return semantic ranking and expose `reranker_fallback_reason=aidranker_unavailable`.

The API response reports whether reranking actually ran, the frozen alpha, model reference, fallback state, raw reranker score, and final fusion score.

## Model artifact

The fine-tuned model is a deployment artifact and is intentionally not committed to Git.

The V0.8 Docker training command writes the local model under the API working directory, normally:

```text
apps/api/models/aidranker-v1.local
```

The development Docker Compose bind mount (`./apps/api:/app`) makes that model available inside the API container as:

```text
/app/models/aidranker-v1.local
```

The default setting `AIDLENS_AIDRANKER_MODEL=models/aidranker-v1.local` therefore resolves correctly from the `/app` working directory. Production may instead provide an absolute mounted path or a model-registry/Hugging Face identifier.

## Enable locally

The API must include ML dependencies:

```env
AIDLENS_API_EXTRAS=ml
AIDLENS_EMBEDDING_PROVIDER=sentence-transformers
AIDLENS_EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
AIDLENS_AIDRANKER_PROVIDER=sentence-transformers
AIDLENS_AIDRANKER_MODEL=models/aidranker-v1.local
AIDLENS_AIDRANKER_CANDIDATE_K=40
AIDLENS_AIDRANKER_FAIL_OPEN=true
```

Rebuild after changing `AIDLENS_API_EXTRAS`:

```bash
docker compose build api
```

Start the stack:

```bash
docker compose up -d
```

## Smoke test

```bash
curl -sS http://localhost:8000/api/v1/search/evidence \
  -H 'content-type: application/json' \
  -d '{
    "query":"What implementation factors improved program outcomes?",
    "mode":"auto",
    "rerank":"aidranker",
    "top_k":10,
    "max_per_evaluation":3
  }' | python -m json.tool
```

Expected metadata:

```json
{
  "mode": "semantic",
  "reranker_applied": true,
  "reranker": "aidranker-v1",
  "reranker_alpha": 0.5,
  "reranker_fallback_reason": null
}
```

Each reranked hit should include `semantic_score`, `reranker_score`, `fusion_score`, and `aidranker` in `retrieval_sources`.

## Fail-open verification

For normal product requests (`rerank=auto`), a missing/unavailable model must not take evidence search down. The API returns semantic ranking and:

```json
{
  "mode": "semantic",
  "reranker_applied": false,
  "reranker_fallback_reason": "aidranker_unavailable"
}
```

Operational verification should use `rerank=aidranker`, which fails closed with HTTP 503 if the configured model cannot be served.

## Candidate pool

The production reranker scores a bounded first-stage semantic pool before diversity filtering. Default candidate pool is 40 and scales up to four times requested `top_k`, capped at 100. This bounds CPU inference latency while leaving enough headroom for the default max-3-per-report diversity rule.

## Production gate

Before enabling AidRanker by default in a deployed environment:

1. provision the validated V1 model artifact;
2. build the API with ML extras;
3. verify embedding coverage is healthy;
4. run the explicit `rerank=aidranker` smoke test;
5. confirm response alpha is exactly `0.5`;
6. confirm Auto search reports `reranker_applied=true`;
7. exercise fail-open behavior separately;
8. monitor search latency and reranker fallback frequency.
