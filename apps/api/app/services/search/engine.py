from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.evaluation import (
    EvidenceSearchHit,
    EvidenceSearchRequest,
    EvidenceSearchResponse,
)
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

    pool_k = _diversity_pool_k(payload)
    retrieval_payload = payload.model_copy(update={"top_k": pool_k})

    if mode == "lexical":
        hits = await lexical_search(session, retrieval_payload)
        hits = _finalize_hits(hits, payload)
        return EvidenceSearchResponse(
            query=payload.query,
            mode=mode,
            max_per_evaluation=payload.max_per_evaluation,
            hits=hits,
        )

    if query_vector is None:
        raise ValueError("Semantic and hybrid retrieval require a query vector.")

    if mode == "semantic":
        hits = await semantic_search(session, retrieval_payload, query_vector)
        hits = _finalize_hits(hits, payload)
    else:
        candidate_k = min(max(pool_k * 4, 20), 100)
        candidate_payload = payload.model_copy(update={"top_k": candidate_k})
        # AsyncSession is stateful and does not support concurrent database
        # operations. Run both retrieval queries sequentially on this session;
        # RRF still fuses the independently ranked candidate lists afterward.
        lexical_hits = await lexical_search(session, candidate_payload)
        semantic_hits = await semantic_search(
            session,
            candidate_payload,
            query_vector,
        )
        hits = reciprocal_rank_fusion(
            lexical_hits,
            semantic_hits,
            top_k=pool_k,
        )
        hits = _finalize_hits(hits, payload)

    return EvidenceSearchResponse(
        query=payload.query,
        mode=mode,
        embedding_model=embedding_model,
        max_per_evaluation=payload.max_per_evaluation,
        hits=hits,
    )


def _diversity_pool_k(payload: EvidenceSearchRequest) -> int:
    if payload.max_per_evaluation is None:
        return payload.top_k
    return min(max(payload.top_k * 4, 20), 100)


def _finalize_hits(
    hits: list[EvidenceSearchHit],
    payload: EvidenceSearchRequest,
) -> list[EvidenceSearchHit]:
    if payload.max_per_evaluation is None:
        return hits[: payload.top_k]

    selected: list[EvidenceSearchHit] = []
    counts: dict[str, int] = {}
    for hit in hits:
        count = counts.get(hit.evaluation_id, 0)
        if count >= payload.max_per_evaluation:
            continue
        selected.append(hit)
        counts[hit.evaluation_id] = count + 1
        if len(selected) >= payload.top_k:
            break
    return selected
