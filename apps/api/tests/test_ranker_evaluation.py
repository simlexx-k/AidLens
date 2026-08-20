import uuid

from app.schemas.benchmark import RankerTrainingRecord
from app.services.ranker.evaluation import evaluate_reranking


def _record(
    *,
    evaluation_id: str,
    relevance: int,
    semantic_score: float | None,
    semantic_rank: int | None,
) -> RankerTrainingRecord:
    modes = ["semantic"] if semantic_rank is not None else ["lexical"]
    ranks = {"semantic": semantic_rank} if semantic_rank is not None else {"lexical": 1}
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
        retrieval_rank=semantic_rank or 1,
        score=semantic_score or 0.1,
        semantic_score=semantic_score,
        lexical_score=None if semantic_rank is not None else 0.2,
        retrieval_modes=modes,
        mode_ranks=ranks,
    )


def test_aidranker_can_improve_order_without_changing_candidate_set() -> None:
    negative = _record(
        evaluation_id="NEG",
        relevance=0,
        semantic_score=0.9,
        semantic_rank=1,
    )
    medium = _record(
        evaluation_id="MED",
        relevance=2,
        semantic_score=0.8,
        semantic_rank=2,
    )
    strong = _record(
        evaluation_id="STRONG",
        relevance=3,
        semantic_score=0.4,
        semantic_rank=3,
    )
    lexical_only = _record(
        evaluation_id="LEXICAL",
        relevance=3,
        semantic_score=None,
        semantic_rank=None,
    )
    records = [negative, medium, strong, lexical_only]
    scores = {
        str(negative.chunk_id): 0.1,
        str(medium.chunk_id): 0.8,
        str(strong.chunk_id): 0.9,
    }

    report = evaluate_reranking(
        records,
        scores,
        candidate_mode="semantic",
        top_k=2,
        model_path="aidranker-test",
    )

    assert report["query_count"] == 1
    assert report["mean_candidate_recall_ceiling"] == 0.666667
    assert report["baseline"]["mean_recall_at_k"] == 0.333333
    assert report["reranked"]["mean_recall_at_k"] == 0.666667
    assert report["baseline"]["mean_reciprocal_rank"] == 0.5
    assert report["reranked"]["mean_reciprocal_rank"] == 1.0
    assert report["reranked"]["mean_ndcg_at_k"] > report["baseline"]["mean_ndcg_at_k"]
