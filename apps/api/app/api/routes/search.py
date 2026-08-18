import asyncio
from functools import lru_cache

from fastapi import APIRouter, HTTPException

from app.api.dependencies import DbSession
from app.core.config import get_settings
from app.schemas.evaluation import EvidenceSearchRequest, EvidenceSearchResponse
from app.services.embeddings.sentence_transformer import SentenceTransformerEncoder
from app.services.search.engine import execute_search

router = APIRouter(prefix="/search", tags=["search"])


@lru_cache(maxsize=4)
def _encoder(model_name: str) -> SentenceTransformerEncoder:
    return SentenceTransformerEncoder(model_name)


@router.post("/evidence", response_model=EvidenceSearchResponse)
async def search_evidence(
    payload: EvidenceSearchRequest,
    db: DbSession,
) -> EvidenceSearchResponse:
    settings = get_settings()
    needs_vector = payload.mode in {"auto", "semantic", "hybrid"}
    semantic_enabled = settings.embedding_provider == "sentence-transformers"

    query_vector: list[float] | None = None
    embedding_model: str | None = None
    if needs_vector and semantic_enabled:
        encoder = _encoder(settings.embedding_model)
        query_vector = await asyncio.to_thread(encoder.encode_query, payload.query)
        embedding_model = settings.embedding_model
    elif payload.mode in {"semantic", "hybrid"}:
        raise HTTPException(
            status_code=503,
            detail=(
                "Semantic retrieval is disabled. Set "
                "AIDLENS_EMBEDDING_PROVIDER=sentence-transformers and rebuild "
                "the API with AIDLENS_API_EXTRAS=ml."
            ),
        )

    try:
        return await execute_search(
            db,
            payload,
            query_vector=query_vector,
            embedding_model=embedding_model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
