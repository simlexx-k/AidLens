import uuid

from app.schemas.benchmark import RankerTrainingRecord
from app.services.ranker.evaluation import evaluate_fusion_sweep, evaluate_reranking


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
    assert report["mean_candidate_recall_supporting_ceiling"] == 0.666667
    assert report["mean_candidate_recall_direct_ceiling"] == 0.5
    assert report["baseline"]["mean_recall_at_k"] == 0.333333
    assert report["baseline"]["mean_recall_supporting_at_k"] == 0.333333
    assert report["baseline"]["mean_recall_direct_at_k"] == 0.0
    assert report["reranked"]["mean_recall_at_k"] == 0.666667
    assert report["reranked"]["mean_recall_supporting_at_k"] == 0.666667
    assert report["reranked"]["mean_recall_direct_at_k"] == 0.5
    assert report["reranked"]["mean_graded_recall_at_k"] > report["baseline"][
        "mean_graded_recall_at_k"
    ]
    assert report["baseline"]["mean_reciprocal_rank"] == 0.5
    assert report["reranked"]["mean_reciprocal_rank"] == 1.0
    assert report["reranked"]["mean_ndcg_at_k"] > report["baseline"]["mean_ndcg_at_k"]


def test_fusion_sweep_can_select_middle_alpha_without_losing_strong_recall() -> None:
    supporting = _record(
        evaluation_id="SUPPORTING",
        relevance=2,
        semantic_score=0.9,
        semantic_rank=1,
    )
    contextual = _record(
        evaluation_id="CONTEXTUAL",
        relevance=1,
        semantic_score=0.8,
        semantic_rank=2,
    )
    direct = _record(
        evaluation_id="DIRECT",
        relevance=3,
        semantic_score=0.7,
        semantic_rank=3,
    )
    negative = _record(
        evaluation_id="NEGATIVE",
        relevance=0,
        semantic_score=0.6,
        semantic_rank=4,
    )
    records = [supporting, contextual, direct, negative]
    scores = {
        str(supporting.chunk_id): 0.1,
        str(contextual.chunk_id): 0.5,
        str(direct.chunk_id): 0.9,
        str(negative.chunk_id): 0.7,
    }

    report = evaluate_fusion_sweep(
        records,
        scores,
        candidate_mode="semantic",
        top_k=3,
        model_path="aidranker-test",
        alphas=[0.0, 0.5, 1.0],
    )

    assert report["selection"]["status"] == "selected"
    assert report["selection"]["alpha"] == 0.5
    selected = report["selection"]["metrics"]
    baseline = report["baseline"]
    assert selected["mean_ndcg_at_k"] > baseline["mean_ndcg_at_k"]
    assert selected["mean_recall_supporting_at_k"] >= baseline[
        "mean_recall_supporting_at_k"
    ]
    assert selected["mean_recall_direct_at_k"] >= baseline["mean_recall_direct_at_k"]

    pure_ranker = next(item for item in report["sweep"] if item["alpha"] == 1.0)
    assert pure_ranker["feasible"] is False
