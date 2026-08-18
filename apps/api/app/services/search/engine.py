import asyncio

from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.evaluation import EvidenceSearchRequest, EvidenceSearchResponse
from app.services.search.hybrid import reciprocal_rank_fusion
from app.services.search.lexical import lexical_search
from app.services.search.semantic import semantic_search


async def execute_search(
    session: AsyncSession,
    payload: EvidenceSearchRequest,
    *,
    query_vector: list[float] | None = None,
    embedding_model: str | None = None,
) -> EvidenceSearchResponse:
    mode = payload.mode
    if mode == "auto":
        mode = "hybrid" if query_vector is not None else "lexical"

    if mode == "lexical":
        hits = await lexical_search(session, payload)
        return EvidenceSearchResponse(query=payload.query, mode=mode, hits=hits)

    if query_vector is None:
        raise ValueError("Semantic and hybrid retrieval require a query vector.")

    if mode == "semantic":
        hits = await semantic_search(session, payload, query_vector)
    else:
        candidate_k = min(max(payload.top_k * 4, 20), 100)
        candidate_payload = payload.model_copy(update={"top_k": candidate_k})
        lexical_hits, semantic_hits = await asyncio.gather(
            lexical_search(session, candidate_payload),
            semantic_search(session, candidate_payload, query_vector),
        )
        hits = reciprocal_rank_fusion(
            lexical_hits,
            semantic_hits,
            top_k=payload.top_k,
        )

    return EvidenceSearchResponse(
        query=payload.query,
        mode=mode,
        embedding_model=embedding_model,
        hits=hits,
    )
