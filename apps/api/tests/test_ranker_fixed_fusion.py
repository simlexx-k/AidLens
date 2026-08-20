import uuid

from app.schemas.benchmark import RankerTrainingRecord
from app.services.ranker.fixed_fusion import evaluate_fixed_fusion


def _record(
    *,
    evaluation_id: str,
    relevance: int,
    semantic_score: float,
    semantic_rank: int,
) -> RankerTrainingRecord:
    return RankerTrainingRecord(
        query_id="q1",
        family="outcomes",
        query="What worked?",
        chunk_id=uuid.uuid4(),
        evaluation_id=evaluation_id,
        title="Evaluation",
        section="findings",
        text=f"Evidence {evaluation_id}",
        relevance=relevance,
        retrieval_rank=semantic_rank,
        score=semantic_score,
        semantic_score=semantic_score,
        lexical_score=None,
        retrieval_modes=["semantic"],
        mode_ranks={"semantic": semantic_rank},
    )


def test_fixed_fusion_evaluates_one_frozen_alpha_without_selection() -> None:
    weak = _record(
        evaluation_id="WEAK",
        relevance=1,
        semantic_score=0.9,
        semantic_rank=1,
    )
    strong = _record(
        evaluation_id="STRONG",
        relevance=3,
        semantic_score=0.5,
        semantic_rank=2,
    )
    negative = _record(
        evaluation_id="NEG",
        relevance=0,
        semantic_score=0.4,
        semantic_rank=3,
    )
    records = [weak, strong, negative]
    reranker_scores = {
        str(weak.chunk_id): 0.2,
        str(strong.chunk_id): 0.9,
        str(negative.chunk_id): 0.1,
    }

    report = evaluate_fixed_fusion(
        records,
        reranker_scores,
        alpha=0.5,
        candidate_mode="semantic",
        top_k=2,
        model_path="aidranker-test",
    )

    assert report["alpha"] == 0.5
    assert report["query_count"] == 1
    assert "selection" not in report
    assert "sweep" not in report
    assert report["fused"]["mean_recall_direct_at_k"] == 1.0
    assert report["fused"]["mean_ndcg_at_k"] >= report["baseline"]["mean_ndcg_at_k"]
