# Semantic and hybrid retrieval

AidLens V0.2 keeps PostgreSQL full-text search as the baseline and adds optional SentenceTransformers embeddings stored in pgvector.

## Enable the ML runtime

The normal API image deliberately excludes PyTorch and model weights. Set these values in `.env`:

```bash
AIDLENS_API_EXTRAS=ml
AIDLENS_EMBEDDING_PROVIDER=sentence-transformers
AIDLENS_EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
```

Then rebuild and restart:

```bash
docker compose build api
docker compose up -d
```

## Embed the corpus

Start with a small batch:

```bash
docker compose run --rm api python -m app.cli embed --limit 100 --batch-size 16
```

Inspect coverage and data-quality statistics:

```bash
docker compose run --rm api python -m app.cli corpus-report
```

Then embed all remaining chunks:

```bash
docker compose run --rm api python -m app.cli embed --batch-size 32
```

## Retrieval modes

`POST /api/v1/search/evidence` accepts `lexical`, `semantic`, `hybrid`, or `auto`. Auto uses hybrid when semantic retrieval is enabled and lexical otherwise. Hybrid uses reciprocal-rank fusion and preserves lexical/semantic component scores for inspection.

The `0002_semantic_retrieval` migration adds the embedding-model identifier and an HNSW cosine index so vector provenance and retrieval performance remain auditable.
