import uuid

from app.schemas.evaluation import EvidenceSearchHit


def reciprocal_rank_fusion(
    lexical_hits: list[EvidenceSearchHit],
    semantic_hits: list[EvidenceSearchHit],
    *,
    top_k: int,
    rrf_k: int = 60,
) -> list[EvidenceSearchHit]:
    scores: dict[uuid.UUID, float] = {}
    items: dict[uuid.UUID, EvidenceSearchHit] = {}
    lexical_scores: dict[uuid.UUID, float] = {}
    semantic_scores: dict[uuid.UUID, float] = {}
    sources: dict[uuid.UUID, set[str]] = {}
    for name, hits in (("lexical", lexical_hits), ("semantic", semantic_hits)):
        for rank, hit in enumerate(hits, start=1):
            chunk_id = hit.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            items.setdefault(chunk_id, hit)
            sources.setdefault(chunk_id, set()).add(name)
            if hit.lexical_score is not None:
                lexical_scores[chunk_id] = hit.lexical_score
            if hit.semantic_score is not None:
                semantic_scores[chunk_id] = hit.semantic_score
    ordered_ids = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
    return [
        items[chunk_id].model_copy(
            update={
                "score": scores[chunk_id],
                "lexical_score": lexical_scores.get(chunk_id),
                "semantic_score": semantic_scores.get(chunk_id),
                "retrieval_sources": sorted(sources[chunk_id]),
            }
        )
        for chunk_id in ordered_ids
    ]
