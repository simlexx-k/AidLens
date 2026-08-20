import uuid

import pytest

from app.schemas.evaluation import EvidenceSearchHit
from app.services.ranker.serving import (
    FROZEN_AIDRANKER_ALPHA,
    AidRankerService,
    fuse_semantic_and_aidranker,
)


def _hit(evaluation_id: str, semantic_score: float) -> EvidenceSearchHit:
    return EvidenceSearchHit(
        chunk_id=uuid.uuid4(),
        evaluation_id=evaluation_id,
        title=f"Evaluation {evaluation_id}",
        text="Evidence passage.",
        score=semantic_score,
        semantic_score=semantic_score,
        retrieval_sources=["semantic"],
        source_url="https://example.test/evaluation",
    )


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[tuple[list[tuple[str, str]], int]] = []

    def predict(self, pairs, *, batch_size, show_progress_bar):
        self.calls.append((list(pairs), batch_size))
        return [float(index) for index, _ in enumerate(pairs)]


def test_frozen_aidranker_fusion_reorders_semantic_candidates() -> None:
    hits = [
        _hit("A", 0.9),
        _hit("B", 0.8),
        _hit("C", 0.1),
    ]

    fused = fuse_semantic_and_aidranker(
        hits,
        [0.0, 2.0, 1.0],
    )

    assert FROZEN_AIDRANKER_ALPHA == 0.5
    assert [hit.evaluation_id for hit in fused] == ["B", "A", "C"]
    assert fused[0].fusion_score == fused[0].score
    assert fused[0].reranker_score == 2.0
    assert fused[0].retrieval_sources == ["semantic", "aidranker"]


def test_fusion_rejects_mismatched_score_count() -> None:
    hits = [_hit("A", 0.9)]

    try:
        fuse_semantic_and_aidranker(hits, [])
    except ValueError as exc:
        assert "score count" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("Expected a mismatched score count to fail.")


@pytest.mark.asyncio
async def test_warmup_loads_model_and_executes_tiny_prediction() -> None:
    service = AidRankerService("fake", batch_size=16)
    fake_model = _FakeModel()
    service.__dict__["_model"] = fake_model

    assert service.model_loaded is True
    await service.warmup()

    assert len(fake_model.calls) == 1
    assert fake_model.calls[0][1] == 1


def test_rerank_uses_configured_serving_batch_size() -> None:
    service = AidRankerService("fake", batch_size=16)
    fake_model = _FakeModel()
    service.__dict__["_model"] = fake_model

    service._rerank_sync(
        "food security",
        [_hit("A", 0.9), _hit("B", 0.8)],
    )

    assert fake_model.calls[-1][1] == 16
    assert service.backend == "torch"
    assert service.device == "auto"
