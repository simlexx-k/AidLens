from app.schemas.benchmark import BenchmarkQueryResult, RetrievalMetrics
from app.services.evaluation.benchmark import _summarize_family


def _result(*, recall: float, ndcg: float) -> BenchmarkQueryResult:
    return BenchmarkQueryResult(
        query_id="q1",
        query="What worked?",
        family="intervention_outcomes",
        mode="hybrid",
        metrics=RetrievalMetrics(
            recall_at_k=recall,
            reciprocal_rank=1.0,
            ndcg_at_k=ndcg,
            relevant_count=1,
            retrieved_relevant_count=1,
            unique_evaluations_at_k=5,
            duplicate_share_at_k=0.5,
        ),
        top_evaluation_ids=["A"],
        top_sections=["findings"],
    )


def test_family_summary_aggregates_retrieval_quality() -> None:
    summary = _summarize_family(
        "intervention_outcomes",
        "hybrid",
        [_result(recall=1.0, ndcg=0.8), _result(recall=0.5, ndcg=0.6)],
    )

    assert summary.family == "intervention_outcomes"
    assert summary.mode == "hybrid"
    assert summary.query_count == 2
    assert summary.mean_recall_at_k == 0.75
    assert summary.mean_ndcg_at_k == 0.7
    assert summary.mean_unique_evaluations_at_k == 5.0
