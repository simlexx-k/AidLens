import asyncio
from time import perf_counter

from fastapi import APIRouter, HTTPException

from app.api.dependencies import DbSession
from app.core.config import get_settings
from app.schemas.evaluation import EvidenceSearchRequest, EvidenceSearchResponse
from app.services.embeddings.provider import get_embedding_encoder
from app.services.ranker.provider import get_aidranker_service
from app.services.ranker.serving import AidRankerService
from app.services.search.engine import execute_search

router = APIRouter(prefix="/search", tags=["search"])


@router.post("/evidence", response_model=EvidenceSearchResponse)
async def search_evidence(
    payload: EvidenceSearchRequest,
    db: DbSession,
) -> EvidenceSearchResponse:
    request_started = perf_counter()
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
    query_encoding_latency_ms: float | None = None
    if needs_vector and semantic_enabled:
        encoding_started = perf_counter()
        encoder = get_embedding_encoder(settings.embedding_model)
        query_vector = await asyncio.to_thread(encoder.encode_query, payload.query)
        query_encoding_latency_ms = _elapsed_ms(encoding_started)
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
        response = await execute_search(
            db,
            payload,
            query_vector=query_vector,
            embedding_model=embedding_model,
            reranker=reranker,
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return response.model_copy(
        update={
            "query_encoding_latency_ms": query_encoding_latency_ms,
            "request_latency_ms": _elapsed_ms(request_started),
        }
    )


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000.0, 3)
