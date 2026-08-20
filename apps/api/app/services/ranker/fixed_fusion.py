from app.schemas.benchmark import RankerTrainingRecord
from app.services.ranker.evaluation import (
    evaluate_fusion_sweep,
    score_aidranker_model,
)


def evaluate_aidranker_fixed_fusion_model(
    records: list[RankerTrainingRecord],
    model_path: str,
    *,
    alpha: float,
    candidate_mode: str = "semantic",
    top_k: int = 10,
    batch_size: int = 32,
) -> dict[str, object]:
    """Evaluate one frozen first-stage/AidRanker fusion weight."""

    reranker_scores = score_aidranker_model(
        records,
        model_path,
        candidate_mode=candidate_mode,
        batch_size=batch_size,
    )
    return evaluate_fixed_fusion(
        records,
        reranker_scores,
        alpha=alpha,
        candidate_mode=candidate_mode,
        top_k=top_k,
        model_path=model_path,
    )


def evaluate_fixed_fusion(
    records: list[RankerTrainingRecord],
    reranker_scores: dict[str, float],
    *,
    alpha: float,
    candidate_mode: str = "semantic",
    top_k: int = 10,
    model_path: str | None = None,
) -> dict[str, object]:
    """Evaluate exactly one frozen fusion alpha without model selection."""

    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("Fusion alpha must be between 0 and 1.")

    sweep_report = evaluate_fusion_sweep(
        records,
        reranker_scores,
        candidate_mode=candidate_mode,
        top_k=top_k,
        model_path=model_path,
        alphas=[alpha],
        diversity_tolerance=1.0,
    )
    fused = sweep_report["sweep"][0]

    return {
        "model": model_path,
        "candidate_mode": candidate_mode,
        "alpha": alpha,
        "top_k": top_k,
        "query_count": sweep_report["query_count"],
        "baseline": sweep_report["baseline"],
        "fused": fused["metrics"],
        "mean_candidate_recall_ceiling": sweep_report[
            "mean_candidate_recall_ceiling"
        ],
        "mean_candidate_recall_supporting_ceiling": sweep_report[
            "mean_candidate_recall_supporting_ceiling"
        ],
        "mean_candidate_recall_direct_ceiling": sweep_report[
            "mean_candidate_recall_direct_ceiling"
        ],
        "mean_candidate_graded_recall_ceiling": sweep_report[
            "mean_candidate_graded_recall_ceiling"
        ],
        "families": fused["families"],
        "queries": fused["queries"],
    }
