import asyncio
import math
from functools import cached_property

from app.schemas.evaluation import EvidenceSearchHit

FROZEN_AIDRANKER_ALPHA = 0.50


class AidRankerService:
    """Lazy CrossEncoder serving adapter for the validated AidRanker pipeline."""

    name = "aidranker-v1"
    alpha = FROZEN_AIDRANKER_ALPHA

    def __init__(
        self,
        model_name_or_path: str,
        *,
        candidate_k: int = 40,
        fail_open: bool = True,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.candidate_k = candidate_k
        self.fail_open = fail_open

    @cached_property
    def _model(self):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "AidRanker serving requires the API ML extras. "
                "Install with `pip install -e '.[ml]'` or rebuild with "
                "AIDLENS_API_EXTRAS=ml."
            ) from exc
        return CrossEncoder(self.model_name_or_path)

    async def rerank(
        self,
        query: str,
        hits: list[EvidenceSearchHit],
    ) -> list[EvidenceSearchHit]:
        if not hits:
            return []
        return await asyncio.to_thread(self._rerank_sync, query, hits)

    def _rerank_sync(
        self,
        query: str,
        hits: list[EvidenceSearchHit],
    ) -> list[EvidenceSearchHit]:
        pairs = [(query, hit.text) for hit in hits]
        predictions = self._model.predict(
            pairs,
            batch_size=32,
            show_progress_bar=False,
        )
        reranker_scores = [float(score) for score in predictions]
        return fuse_semantic_and_aidranker(
            hits,
            reranker_scores,
            alpha=self.alpha,
        )


def fuse_semantic_and_aidranker(
    hits: list[EvidenceSearchHit],
    reranker_scores: list[float],
    *,
    alpha: float = FROZEN_AIDRANKER_ALPHA,
) -> list[EvidenceSearchHit]:
    """Fuse semantic and AidRanker scores using the frozen validated alpha."""

    if len(hits) != len(reranker_scores):
        raise ValueError("AidRanker score count must match the semantic candidate count.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("AidRanker fusion alpha must be between 0 and 1.")
    if not hits:
        return []

    semantic_scores = [
        float(hit.semantic_score if hit.semantic_score is not None else hit.score)
        for hit in hits
    ]
    semantic_normalized = _minmax(semantic_scores)
    ranker_normalized = _minmax(reranker_scores)

    fused_hits: list[EvidenceSearchHit] = []
    for hit, raw_ranker, semantic_score, ranker_score in zip(
        hits,
        reranker_scores,
        semantic_normalized,
        ranker_normalized,
        strict=True,
    ):
        fusion_score = ((1.0 - alpha) * semantic_score) + (alpha * ranker_score)
        sources = list(hit.retrieval_sources)
        if "aidranker" not in sources:
            sources.append("aidranker")
        fused_hits.append(
            hit.model_copy(
                update={
                    "score": fusion_score,
                    "reranker_score": raw_ranker,
                    "fusion_score": fusion_score,
                    "retrieval_sources": sources,
                }
            )
        )

    return sorted(
        fused_hits,
        key=lambda hit: (
            hit.fusion_score if hit.fusion_score is not None else hit.score,
            hit.semantic_score if hit.semantic_score is not None else hit.score,
        ),
        reverse=True,
    )


def _minmax(values: list[float]) -> list[float]:
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return [0.5 for _ in values]
    span = high - low
    return [(value - low) / span for value in values]
