import asyncio
from functools import lru_cache

from fastapi import APIRouter, HTTPException

from app.api.dependencies import DbSession
from app.core.config import get_settings
from app.schemas.evaluation import EvidenceSearchRequest, EvidenceSearchResponse
from app.services.embeddings.sentence_transformer import SentenceTransformerEncoder
from app.services.ranker.provider import get_aidranker_service
from app.services.ranker.serving import AidRankerService
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
    aidranker_enabled = settings.aidranker_provider == "sentence-transformers"

    if payload.rerank == "aidranker" and not aidranker_enabled:
        raise HTTPException(
            status_code=503,
            detail=(
                "AidRanker is disabled. Set "
                "AIDLENS_AIDRANKER_PROVIDER=sentence-transformers and provide the "
                "validated AidRanker V1 model artifact."
            ),
        )
    if payload.rerank == "aidranker" and not semantic_enabled:
        raise HTTPException(
            status_code=503,
            detail="AidRanker requires semantic retrieval to be enabled.",
        )

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

    reranker: AidRankerService | None = None
    if payload.rerank != "disabled" and aidranker_enabled and query_vector is not None:
        reranker = get_aidranker_service(
            settings.aidranker_model,
            settings.aidranker_candidate_k,
            settings.aidranker_batch_size,
            settings.aidranker_device,
            settings.aidranker_fail_open,
        )

    try:
        return await execute_search(
            db,
            payload,
            query_vector=query_vector,
            embedding_model=embedding_model,
            reranker=reranker,
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
