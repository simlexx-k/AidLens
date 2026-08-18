# AidLens

**Development Evidence Intelligence**

AidLens turns development-program evaluation reports into a structured, searchable evidence layer: **intervention → context → outcome → evidence**.

The first data source is AidData's preserved archive of USAID Development Experience Clearinghouse evaluations. The product is intentionally designed to grow beyond a document chatbot into a reproducible ML system with domain classifiers, evidence extraction, learned retrieval, and eventually a development-evidence graph.

## V0.1 foundation

This first implementation provides:

- FastAPI API with typed OpenAPI contracts
- PostgreSQL + pgvector persistence
- evaluation and evidence-chunk schema with source provenance
- polite AidData archive ingestion adapter
- report text chunking with basic section detection
- PostgreSQL full-text evidence search as the retrieval baseline
- vector-ready chunk storage for the upcoming embedding model
- Next.js evidence-search interface
- Docker Compose local environment
- Alembic migrations, API tests, linting, and GitHub Actions CI

The lexical search baseline is deliberate: when semantic retrieval is introduced, we will have a conventional IR baseline against which to measure Recall@K, MRR, and nDCG rather than claiming improvement without comparison.

## Architecture

```text
AidData archive
      │
      ▼
archive adapter ──► ingestion/chunking ──► PostgreSQL + pgvector
                                               │
                                  ┌────────────┴────────────┐
                                  ▼                         ▼
                            lexical baseline          ML retrieval
                                  │                   (next phase)
                                  └────────────┬────────────┘
                                               ▼
                                            FastAPI
                                               │
                                               ▼
                                            Next.js
```

Training is intentionally separated from API serving. The planned ML stack is PyTorch + Hugging Face Transformers + SentenceTransformers + scikit-learn, with PyTorch Geometric added when the evidence graph is mature enough to justify graph learning.

## Repository layout

```text
apps/
  api/          FastAPI, ingestion, persistence, retrieval, migrations
  web/          Next.js product interface
.github/        CI
```

## Run locally

1. Create the environment file:

```bash
cp .env.example .env
```

2. Start the stack:

```bash
docker compose up --build
```

The web app runs at `http://localhost:3000`, the API at `http://localhost:8000`, and OpenAPI docs at `http://localhost:8000/docs`.

3. Ingest the first archive page (10 evaluations):

```bash
docker compose run --rm api python -m app.cli ingest --pages 1
```

For a larger batch:

```bash
docker compose run --rm api python -m app.cli ingest --pages 10 --concurrency 4
```

The client intentionally throttles source requests. Do not increase concurrency aggressively against the public archive.

## API

```text
GET  /api/v1/health
GET  /api/v1/ready
GET  /api/v1/evaluations
GET  /api/v1/evaluations/{external_id}
POST /api/v1/search/evidence
```

Example evidence search:

```bash
curl -X POST http://localhost:8000/api/v1/search/evidence \
  -H 'Content-Type: application/json' \
  -d '{"query":"smallholder farmer income","top_k":10}'
```

## ML roadmap

### Phase 1 — retrieval baseline

- ingest and normalize the corpus
- evaluate chunking quality
- create frozen retrieval test queries
- benchmark PostgreSQL full-text retrieval

### Phase 2 — AidEncoder / AidRanker

- generate embeddings with a pretrained SentenceTransformer
- evaluate semantic retrieval against the lexical baseline
- annotate query/passage relevance
- fine-tune a domain retrieval model and CrossEncoder reranker

### Phase 3 — EvalClassifier / EvidenceExtractor

- build human-reviewed training labels
- multi-label evaluation/sector/method classification
- intervention, outcome, population, factor, and effect-direction extraction
- persist extracted evidence with passage-level provenance

### Phase 4 — AidGraph

- canonical entity resolution
- evidence graph construction
- GraphSAGE/GAT experiments for similarity and link prediction

## Data-source note

AidLens does not treat the AidData archive as a complete historical record of USAID DEC. The source describes itself as a curated subset. AidLens preserves source URLs and raw metadata so derived evidence remains auditable and corpus limitations can be shown to users.
