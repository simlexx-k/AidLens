import asyncio
import hashlib
import math
from functools import cached_property
from pathlib import Path
from time import perf_counter

from app.schemas.evaluation import EvidenceSearchHit

FROZEN_AIDRANKER_ALPHA = 0.50


class AidRankerService:
    """CrossEncoder serving adapter for the validated AidRanker pipeline."""

    name = "aidranker-v1"
    alpha = FROZEN_AIDRANKER_ALPHA
    backend = "torch"

    def __init__(
        self,
        model_name_or_path: str,
        *,
        candidate_k: int = 40,
        batch_size: int = 32,
        device: str = "auto",
        fail_open: bool = True,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.candidate_k = candidate_k
        self.batch_size = batch_size
        self.device = device
        self.fail_open = fail_open
        self._model_load_latency_ms: float | None = None

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

        started = perf_counter()
        model = CrossEncoder(
            self.model_name_or_path,
            device=None if self.device == "auto" else self.device,
            backend=self.backend,
        )
        self._model_load_latency_ms = round((perf_counter() - started) * 1000.0, 3)
        return model

    @property
    def model_loaded(self) -> bool:
        return "_model" in self.__dict__

    @property
    def model_load_latency_ms(self) -> float | None:
        return self._model_load_latency_ms

    async def warmup(self) -> None:
        """Load model weights and execute one tiny prediction before serving traffic."""

        await asyncio.to_thread(self._warmup_sync)

    def _warmup_sync(self) -> None:
        self._model.predict(
            [("AidLens warmup", "Development evidence reranking warmup passage.")],
            batch_size=1,
            show_progress_bar=False,
        )

    @cached_property
    def artifact_fingerprint(self) -> str | None:
        """Fingerprint local model weights/config so ranked responses are reproducible."""

        root = Path(self.model_name_or_path)
        if not root.is_dir():
            return None
        candidates = [root / "config.json", root / "model.safetensors"]
        files = [path for path in candidates if path.is_file()]
        if not files:
            return None
        digest = hashlib.sha256()
        for path in files:
            digest.update(path.name.encode("utf-8"))
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"

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
            batch_size=self.batch_size,
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
