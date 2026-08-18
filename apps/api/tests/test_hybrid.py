import uuid

from app.schemas.evaluation import EvidenceSearchHit
from app.services.search.hybrid import reciprocal_rank_fusion


def _hit(chunk_id: uuid.UUID, *, lexical: float | None = None, semantic: float | None = None):
    return EvidenceSearchHit(
        chunk_id=chunk_id,
        evaluation_id="TEST",
        title="Test evaluation",
        text="Evidence passage",
        score=lexical or semantic or 0.0,
        lexical_score=lexical,
        semantic_score=semantic,
        retrieval_sources=["lexical" if lexical is not None else "semantic"],
        source_url="https://example.com",
    )


def test_rrf_rewards_results_present_in_both_rankings() -> None:
    shared = uuid.uuid4()
    lexical_only = uuid.uuid4()
    semantic_only = uuid.uuid4()
    fused = reciprocal_rank_fusion(
        [_hit(shared, lexical=1.0), _hit(lexical_only, lexical=0.8)],
        [_hit(shared, semantic=0.9), _hit(semantic_only, semantic=0.7)],
        top_k=3,
    )
    assert fused[0].chunk_id == shared
    assert fused[0].retrieval_sources == ["lexical", "semantic"]
    assert fused[0].lexical_score == 1.0
    assert fused[0].semantic_score == 0.9
