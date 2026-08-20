import uuid

from app.schemas.evaluation import EvidenceSearchHit
from app.services.ranker.serving import (
    FROZEN_AIDRANKER_ALPHA,
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
