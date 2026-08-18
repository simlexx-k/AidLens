import asyncio
from functools import lru_cache

from fastapi import APIRouter, HTTPException

from app.api.dependencies import DbSession
from app.core.config import get_settings
from app.schemas.evaluation import EvidenceSearchRequest, EvidenceSearchResponse
from app.services.embeddings.sentence_transformer import SentenceTransformerEncoder
from app.services.search.hybrid import reciprocal_rank_fusion
from app.services.search.lexical import lexical_search
from app.services.search.semantic import semantic_search

router = APIRouter(prefix="/search", tags=["search"])


@lru_cache(maxsize=4)
def _encoder(model_name: str) -> SentenceTransformerEncoder:
    return SentenceTransformerEncoder(model_name)


@router.post("/evidence", response_model=EvidenceSearchResponse)
async def search_evidence(payload: EvidenceSearchRequest, db: DbSession) -> EvidenceSearchResponse:
    settings = get_settings()
    mode = payload.mode
    if mode == "auto":
        mode = "hybrid" if settings.embedding_provider == "sentence-transformers" else "lexical"
    if mode == "lexical":
        hits = await lexical_search(db, payload)
        return EvidenceSearchResponse(query=payload.query, mode="lexical", hits=hits)
    if settings.embedding_provider != "sentence-transformers":
        raise HTTPException(
            status_code=503,
            detail=(
                "Semantic retrieval is disabled. Set "
                "AIDLENS_EMBEDDING_PROVIDER=sentence-transformers and rebuild "
                "the API with AIDLENS_API_EXTRAS=ml."
            ),
        )
    encoder = _encoder(settings.embedding_model)
    query_vector = await asyncio.to_thread(encoder.encode_query, payload.query)
    if mode == "semantic":
        hits = await semantic_search(db, payload, query_vector)
    else:
        candidate_k = min(max(payload.top_k * 4, 20), 100)
        candidate_payload = payload.model_copy(update={"top_k": candidate_k})
        lexical_hits, semantic_hits = await asyncio.gather(
            lexical_search(db, candidate_payload),
            semantic_search(db, candidate_payload, query_vector),
        )
        hits = reciprocal_rank_fusion(lexical_hits, semantic_hits, top_k=payload.top_k)
    return EvidenceSearchResponse(
        query=payload.query,
        mode=mode,
        embedding_model=settings.embedding_model,
        hits=hits,
    )
