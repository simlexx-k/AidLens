import logging
from time import perf_counter

from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.evaluation import (
    EvidenceSearchHit,
    EvidenceSearchRequest,
    EvidenceSearchResponse,
)
from app.services.ranker.serving import AidRankerService
from app.services.search.hybrid import reciprocal_rank_fusion
from app.services.search.intelligence import group_evidence_hits, synthesize_evidence_groups
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
    started = perf_counter()
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
        retrieval_started = perf_counter()
        retrieval_payload = payload.model_copy(
            update={"top_k": _diversity_pool_k(payload)}
        )
        hits = await lexical_search(session, retrieval_payload)
        first_stage_ms = _elapsed_ms(retrieval_started)
        hits = _finalize_hits(hits, payload)
        return _build_response(
            payload,
            mode=mode,
            hits=hits,
            started=started,
            first_stage_latency_ms=first_stage_ms,
            ranking_pipeline=_pipeline(["lexical"], payload),
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
            started=started,
        )

    retrieval_started = perf_counter()
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
    first_stage_ms = _elapsed_ms(retrieval_started)
    hits = _finalize_hits(hits, payload)

    return _build_response(
        payload,
        mode=mode,
        hits=hits,
        started=started,
        embedding_model=embedding_model,
        first_stage_latency_ms=first_stage_ms,
        ranking_pipeline=_pipeline(["lexical", "semantic", "rrf"], payload),
    )


async def _semantic_response(
    session: AsyncSession,
    payload: EvidenceSearchRequest,
    *,
    query_vector: list[float],
    embedding_model: str | None,
    reranker: AidRankerService | None,
    started: float,
) -> EvidenceSearchResponse:
    pool_k = (
        _reranker_pool_k(payload, reranker.candidate_k)
        if reranker is not None
        else _diversity_pool_k(payload)
    )
    retrieval_payload = payload.model_copy(update={"top_k": pool_k})
    retrieval_started = perf_counter()
    hits = await semantic_search(session, retrieval_payload, query_vector)
    first_stage_ms = _elapsed_ms(retrieval_started)

    reranker_applied = False
    reranker_ms: float | None = None
    fallback_reason: str | None = None
    if reranker is not None:
        rerank_started = perf_counter()
        try:
            hits = await reranker.rerank(payload.query, hits)
            reranker_applied = True
        except Exception as exc:  # pragma: no cover - defensive serving fallback
            logger.exception("AidRanker serving failed; returning semantic ranking.")
            if payload.rerank == "aidranker" or not reranker.fail_open:
                raise ValueError("AidRanker reranking is temporarily unavailable.") from exc
            fallback_reason = "aidranker_unavailable"
        finally:
            reranker_ms = _elapsed_ms(rerank_started)

    hits = _finalize_hits(hits, payload)
    pipeline = ["semantic"]
    if reranker_applied and reranker is not None:
        pipeline.extend([reranker.name, f"fusion:{reranker.alpha:.2f}"])
    return _build_response(
        payload,
        mode="semantic",
        hits=hits,
        started=started,
        embedding_model=embedding_model,
        first_stage_latency_ms=first_stage_ms,
        reranker_latency_ms=reranker_ms,
        reranker_applied=reranker_applied,
        reranker=reranker.name if reranker_applied and reranker is not None else None,
        reranker_model=(
            reranker.model_name_or_path
            if reranker_applied and reranker is not None
            else None
        ),
        reranker_model_fingerprint=(
            reranker.artifact_fingerprint
            if reranker_applied and reranker is not None
            else None
        ),
        reranker_alpha=reranker.alpha if reranker_applied and reranker is not None else None,
        reranker_backend=reranker.backend if reranker_applied and reranker is not None else None,
        reranker_batch_size=(
            reranker.batch_size if reranker_applied and reranker is not None else None
        ),
        reranker_device=reranker.device if reranker_applied and reranker is not None else None,
        reranker_model_load_latency_ms=(
            reranker.model_load_latency_ms
            if reranker_applied and reranker is not None
            else None
        ),
        reranker_fallback_reason=fallback_reason,
        ranking_pipeline=_pipeline(pipeline, payload),
    )


def _build_response(
    payload: EvidenceSearchRequest,
    *,
    mode: str,
    hits: list[EvidenceSearchHit],
    started: float,
    ranking_pipeline: list[str],
    embedding_model: str | None = None,
    first_stage_latency_ms: float | None = None,
    reranker_latency_ms: float | None = None,
    reranker_applied: bool = False,
    reranker: str | None = None,
    reranker_model: str | None = None,
    reranker_model_fingerprint: str | None = None,
    reranker_alpha: float | None = None,
    reranker_backend: str | None = None,
    reranker_batch_size: int | None = None,
    reranker_device: str | None = None,
    reranker_model_load_latency_ms: float | None = None,
    reranker_fallback_reason: str | None = None,
) -> EvidenceSearchResponse:
    groups = group_evidence_hits(hits)
    return EvidenceSearchResponse(
        query=payload.query,
        mode=mode,
        embedding_model=embedding_model,
        max_per_evaluation=payload.max_per_evaluation,
        reranker_applied=reranker_applied,
        reranker=reranker,
        reranker_model=reranker_model,
        reranker_model_fingerprint=reranker_model_fingerprint,
        reranker_alpha=reranker_alpha,
        reranker_backend=reranker_backend,
        reranker_batch_size=reranker_batch_size,
        reranker_device=reranker_device,
        reranker_model_load_latency_ms=reranker_model_load_latency_ms,
        reranker_fallback_reason=reranker_fallback_reason,
        ranking_pipeline=ranking_pipeline,
        first_stage_latency_ms=first_stage_latency_ms,
        reranker_latency_ms=reranker_latency_ms,
        total_search_latency_ms=_elapsed_ms(started),
        synthesis=synthesize_evidence_groups(groups),
        groups=groups,
        hits=hits,
    )


def _pipeline(stages: list[str], payload: EvidenceSearchRequest) -> list[str]:
    if payload.max_per_evaluation is not None:
        return [*stages, f"diversity:max-{payload.max_per_evaluation}-per-report"]
    return stages


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000.0, 3)


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
