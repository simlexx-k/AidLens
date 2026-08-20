import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.evaluation import (
    EvidenceSearchHit,
    EvidenceSearchRequest,
    EvidenceSearchResponse,
)
from app.services.ranker.serving import AidRankerService
from app.services.search.hybrid import reciprocal_rank_fusion
from app.services.search.lexical import lexical_search
from app.services.search.semantic import semantic_search

logger = logging.getLogger(__name__)


async def execute_search(
    session: AsyncSession,
    payload: EvidenceSearchRequest,
    *,
    query_vector: list[float] | None = None,
    embedding_model: str | None = None,
    reranker: AidRankerService | None = None,
) -> EvidenceSearchResponse:
    reranker_requested = payload.rerank != "disabled"
    if payload.rerank == "aidranker" and reranker is None:
        raise ValueError("AidRanker reranking is not enabled on this API instance.")
    if payload.rerank == "aidranker" and payload.mode in {"lexical", "hybrid"}:
        raise ValueError("AidRanker reranking requires semantic or auto retrieval mode.")

    mode = payload.mode
    if mode == "auto":
        if reranker_requested and reranker is not None and query_vector is not None:
            mode = "semantic"
        else:
            mode = "hybrid" if query_vector is not None else "lexical"

    if mode == "lexical":
        retrieval_payload = payload.model_copy(
            update={"top_k": _diversity_pool_k(payload)}
        )
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
        return await _semantic_response(
            session,
            payload,
            query_vector=query_vector,
            embedding_model=embedding_model,
            reranker=reranker if reranker_requested else None,
        )

    pool_k = _diversity_pool_k(payload)
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


async def _semantic_response(
    session: AsyncSession,
    payload: EvidenceSearchRequest,
    *,
    query_vector: list[float],
    embedding_model: str | None,
    reranker: AidRankerService | None,
) -> EvidenceSearchResponse:
    pool_k = (
        _reranker_pool_k(payload, reranker.candidate_k)
        if reranker is not None
        else _diversity_pool_k(payload)
    )
    retrieval_payload = payload.model_copy(update={"top_k": pool_k})
    hits = await semantic_search(session, retrieval_payload, query_vector)

    reranker_applied = False
    fallback_reason: str | None = None
    if reranker is not None:
        try:
            hits = await reranker.rerank(payload.query, hits)
            reranker_applied = True
        except Exception as exc:  # pragma: no cover - defensive serving fallback
            logger.exception("AidRanker serving failed; returning semantic ranking.")
            if payload.rerank == "aidranker" or not reranker.fail_open:
                raise ValueError("AidRanker reranking is temporarily unavailable.") from exc
            fallback_reason = "aidranker_unavailable"

    hits = _finalize_hits(hits, payload)
    return EvidenceSearchResponse(
        query=payload.query,
        mode="semantic",
        embedding_model=embedding_model,
        max_per_evaluation=payload.max_per_evaluation,
        reranker_applied=reranker_applied,
        reranker=reranker.name if reranker_applied and reranker is not None else None,
        reranker_model=(
            reranker.model_name_or_path
            if reranker_applied and reranker is not None
            else None
        ),
        reranker_alpha=reranker.alpha if reranker_applied and reranker is not None else None,
        reranker_fallback_reason=fallback_reason,
        hits=hits,
    )


def _reranker_pool_k(payload: EvidenceSearchRequest, configured_candidate_k: int) -> int:
    return min(max(configured_candidate_k, payload.top_k * 4, 40), 100)


def _diversity_pool_k(payload: EvidenceSearchRequest) -> int:
    if payload.max_per_evaluation is None:
        return payload.top_k

    # A diversity cap can discard many otherwise high-ranked passages from a
    # dominant report. Search deeply enough to refill the requested top_k from
    # other evaluations instead of returning a short result list prematurely.
    multiplier = max(payload.max_per_evaluation * 3, 8)
    return min(max(payload.top_k * multiplier, 40), 100)


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
