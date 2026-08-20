import json
import math
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from statistics import mean

from app.schemas.benchmark import RankerTrainingRecord

DEFAULT_FUSION_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def score_aidranker_model(
    records: list[RankerTrainingRecord],
    model_path: str,
    *,
    candidate_mode: str = "semantic",
    batch_size: int = 32,
) -> dict[str, float]:
    """Score one first-stage candidate set with a trained CrossEncoder."""

    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "AidRanker evaluation requires the API ML extras. "
            "Install with `pip install -e '.[ml]'` or use the ML Docker image."
        ) from exc

    candidate_records = [
        record for record in records if _is_candidate(record, candidate_mode)
    ]
    if not candidate_records:
        raise ValueError(f"No candidates found for mode {candidate_mode!r}.")

    model = CrossEncoder(model_path)
    pairs = [(record.query, record.text) for record in candidate_records]
    predictions = model.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=True,
    )
    return {
        str(record.chunk_id): float(score)
        for record, score in zip(candidate_records, predictions, strict=True)
    }


def evaluate_aidranker_model(
    records: list[RankerTrainingRecord],
    model_path: str,
    *,
    candidate_mode: str = "semantic",
    top_k: int = 10,
    batch_size: int = 32,
) -> dict[str, object]:
    """Compare first-stage ranking with AidRanker on the same candidate set."""

    reranker_scores = score_aidranker_model(
        records,
        model_path,
        candidate_mode=candidate_mode,
        batch_size=batch_size,
    )
    return evaluate_reranking(
        records,
        reranker_scores,
        candidate_mode=candidate_mode,
        top_k=top_k,
        model_path=model_path,
    )


def evaluate_aidranker_fusion_sweep_model(
    records: list[RankerTrainingRecord],
    model_path: str,
    *,
    candidate_mode: str = "semantic",
    top_k: int = 10,
    batch_size: int = 32,
    alphas: list[float] | None = None,
    diversity_tolerance: float = 0.05,
) -> dict[str, object]:
    """Sweep one global first-stage/AidRanker fusion weight on a dev split."""

    reranker_scores = score_aidranker_model(
        records,
        model_path,
        candidate_mode=candidate_mode,
        batch_size=batch_size,
    )
    return evaluate_fusion_sweep(
        records,
        reranker_scores,
        candidate_mode=candidate_mode,
        top_k=top_k,
        model_path=model_path,
        alphas=alphas,
        diversity_tolerance=diversity_tolerance,
    )


def evaluate_reranking(
    records: list[RankerTrainingRecord],
    reranker_scores: dict[str, float],
    *,
    candidate_mode: str = "semantic",
    top_k: int = 10,
    model_path: str | None = None,
) -> dict[str, object]:
    """Evaluate baseline and reranked order with pooled judgments as truth."""

    _validate_evaluation_inputs(top_k=top_k, candidate_mode=candidate_mode)
    grouped = _group_records(records)

    query_results: list[dict[str, object]] = []
    for query_id in sorted(grouped):
        all_records = grouped[query_id]
        candidates = _query_candidates(all_records, candidate_mode)
        _require_scores(query_id, candidates, reranker_scores)

        baseline = sorted(
            candidates,
            key=lambda record: _baseline_score(record, candidate_mode),
            reverse=True,
        )
        reranked = sorted(
            candidates,
            key=lambda record: reranker_scores[str(record.chunk_id)],
            reverse=True,
        )
        ceilings = _candidate_recall_ceilings(all_records, candidates)
        query_results.append(
            {
                "query_id": query_id,
                "family": all_records[0].family,
                "candidate_count": len(candidates),
                "candidate_recall_ceiling": ceilings["any"],
                "candidate_recall_ceilings": ceilings,
                "baseline": _ranking_metrics(all_records, baseline, top_k=top_k),
                "reranked": _ranking_metrics(all_records, reranked, top_k=top_k),
            }
        )

    return {
        "model": model_path,
        "candidate_mode": candidate_mode,
        "top_k": top_k,
        "query_count": len(query_results),
        "baseline": _aggregate(query_results, key="baseline"),
        "reranked": _aggregate(query_results, key="reranked"),
        **_aggregate_candidate_ceilings(query_results),
        "families": _family_summaries(query_results),
        "queries": query_results,
    }


def evaluate_fusion_sweep(
    records: list[RankerTrainingRecord],
    reranker_scores: dict[str, float],
    *,
    candidate_mode: str = "semantic",
    top_k: int = 10,
    model_path: str | None = None,
    alphas: list[float] | None = None,
    diversity_tolerance: float = 0.05,
) -> dict[str, object]:
    """Evaluate normalized score fusion and select one dev-only global alpha."""

    _validate_evaluation_inputs(top_k=top_k, candidate_mode=candidate_mode)
    if diversity_tolerance < 0:
        raise ValueError("diversity_tolerance must be non-negative.")

    selected_alphas = _validate_alphas(alphas or DEFAULT_FUSION_ALPHAS)
    grouped = _group_records(records)
    baseline_queries: list[dict[str, object]] = []
    sweep_queries: dict[float, list[dict[str, object]]] = {
        alpha: [] for alpha in selected_alphas
    }

    for query_id in sorted(grouped):
        all_records = grouped[query_id]
        candidates = _query_candidates(all_records, candidate_mode)
        _require_scores(query_id, candidates, reranker_scores)

        baseline = sorted(
            candidates,
            key=lambda record: _baseline_score(record, candidate_mode),
            reverse=True,
        )
        ceilings = _candidate_recall_ceilings(all_records, candidates)
        baseline_queries.append(
            {
                "query_id": query_id,
                "family": all_records[0].family,
                "candidate_count": len(candidates),
                "candidate_recall_ceiling": ceilings["any"],
                "candidate_recall_ceilings": ceilings,
                "metrics": _ranking_metrics(all_records, baseline, top_k=top_k),
            }
        )

        baseline_normalized = _normalize_scores(
            candidates,
            lambda record: _baseline_score(record, candidate_mode),
        )
        ranker_normalized = _normalize_scores(
            candidates,
            lambda record: reranker_scores[str(record.chunk_id)],
        )

        for alpha in selected_alphas:
            fused = sorted(
                candidates,
                key=lambda record: (
                    ((1.0 - alpha) * baseline_normalized[str(record.chunk_id)])
                    + (alpha * ranker_normalized[str(record.chunk_id)]),
                    _baseline_score(record, candidate_mode),
                ),
                reverse=True,
            )
            sweep_queries[alpha].append(
                {
                    "query_id": query_id,
                    "family": all_records[0].family,
                    "metrics": _ranking_metrics(all_records, fused, top_k=top_k),
                }
            )

    baseline_metrics = _aggregate(baseline_queries, key="metrics")
    sweep: list[dict[str, object]] = []
    for alpha in selected_alphas:
        query_results = sweep_queries[alpha]
        metrics = _aggregate(query_results, key="metrics")
        sweep.append(
            {
                "alpha": alpha,
                "metrics": metrics,
                "families": _family_metric_summaries(query_results),
                "queries": query_results,
                "feasible": _fusion_is_feasible(
                    metrics,
                    baseline_metrics,
                    diversity_tolerance=diversity_tolerance,
                ),
            }
        )

    feasible = [item for item in sweep if item["feasible"]]
    selection: dict[str, object]
    if feasible:
        best = max(
            feasible,
            key=lambda item: (
                float(item["metrics"]["mean_ndcg_at_k"]),
                float(item["metrics"]["mean_recall_supporting_at_k"]),
                float(item["metrics"]["mean_recall_direct_at_k"]),
                -float(item["alpha"]),
            ),
        )
        selection = {
            "status": "selected",
            "alpha": best["alpha"],
            "metrics": best["metrics"],
        }
    else:
        selection = {
            "status": "no_feasible_alpha",
            "alpha": None,
            "metrics": None,
        }

    return {
        "model": model_path,
        "candidate_mode": candidate_mode,
        "top_k": top_k,
        "query_count": len(baseline_queries),
        "baseline": baseline_metrics,
        **_aggregate_candidate_ceilings(baseline_queries),
        "selection_constraints": {
            "recall_supporting_at_k": ">= baseline",
            "recall_direct_at_k": ">= baseline",
            "reciprocal_rank": ">= baseline",
            "duplicate_share_tolerance": diversity_tolerance,
        },
        "sweep": sweep,
        "selection": selection,
    }


def write_ranker_report(report: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _validate_evaluation_inputs(*, top_k: int, candidate_mode: str) -> None:
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if candidate_mode not in {"lexical", "semantic", "hybrid"}:
        raise ValueError(f"Unsupported candidate mode: {candidate_mode}")


def _group_records(
    records: list[RankerTrainingRecord],
) -> dict[str, list[RankerTrainingRecord]]:
    grouped: dict[str, list[RankerTrainingRecord]] = defaultdict(list)
    for record in records:
        grouped[record.query_id].append(record)
    return grouped


def _query_candidates(
    records: list[RankerTrainingRecord],
    candidate_mode: str,
) -> list[RankerTrainingRecord]:
    candidates = [record for record in records if _is_candidate(record, candidate_mode)]
    if not candidates:
        query_id = records[0].query_id if records else "unknown"
        raise ValueError(
            f"Query {query_id} has no candidates from mode {candidate_mode!r}."
        )
    return candidates


def _require_scores(
    query_id: str,
    candidates: list[RankerTrainingRecord],
    reranker_scores: dict[str, float],
) -> None:
    missing_scores = [
        str(record.chunk_id)
        for record in candidates
        if str(record.chunk_id) not in reranker_scores
    ]
    if missing_scores:
        raise ValueError(
            f"Query {query_id} is missing {len(missing_scores)} reranker scores."
        )


def _is_candidate(record: RankerTrainingRecord, mode: str) -> bool:
    if mode in record.retrieval_modes or mode in record.mode_ranks:
        return True
    if mode == "semantic":
        return record.semantic_score is not None
    if mode == "lexical":
        return record.lexical_score is not None
    return False


def _baseline_score(record: RankerTrainingRecord, mode: str) -> float:
    if mode == "semantic" and record.semantic_score is not None:
        return record.semantic_score
    if mode == "lexical" and record.lexical_score is not None:
        return record.lexical_score
    rank = record.mode_ranks.get(mode)
    if rank is not None:
        return -float(rank)
    raise ValueError(
        f"Record {record.chunk_id} has no baseline score/rank for mode {mode!r}."
    )


def _candidate_recall_ceilings(
    truth: list[RankerTrainingRecord],
    candidates: list[RankerTrainingRecord],
) -> dict[str, float]:
    return {
        "any": _threshold_recall(truth, candidates, threshold=1),
        "supporting": _threshold_recall(truth, candidates, threshold=2),
        "direct": _direct_recall(truth, candidates),
        "graded": _graded_recall(truth, candidates),
    }


def _candidate_recall_ceiling(
    truth: list[RankerTrainingRecord],
    candidates: list[RankerTrainingRecord],
) -> float:
    """Backward-compatible alias for any-positive candidate recall ceiling."""

    return _candidate_recall_ceilings(truth, candidates)["any"]


def _ranking_metrics(
    truth: list[RankerTrainingRecord],
    ranked: list[RankerTrainingRecord],
    *,
    top_k: int,
) -> dict[str, float | int]:
    top = ranked[:top_k]
    relevant_total = sum(record.relevance > 0 for record in truth)
    retrieved_relevant = sum(record.relevance > 0 for record in top)
    supporting_total = sum(record.relevance >= 2 for record in truth)
    retrieved_supporting = sum(record.relevance >= 2 for record in top)
    direct_total = sum(record.relevance == 3 for record in truth)
    retrieved_direct = sum(record.relevance == 3 for record in top)
    graded_total = sum(record.relevance for record in truth)
    retrieved_graded = sum(record.relevance for record in top)

    recall_any = retrieved_relevant / relevant_total if relevant_total else 0.0
    recall_supporting = (
        retrieved_supporting / supporting_total if supporting_total else 0.0
    )
    recall_direct = retrieved_direct / direct_total if direct_total else 0.0
    graded_recall = retrieved_graded / graded_total if graded_total else 0.0

    reciprocal_rank = 0.0
    for index, record in enumerate(top, start=1):
        if record.relevance > 0:
            reciprocal_rank = 1.0 / index
            break

    gains = [record.relevance for record in top]
    ideal = sorted((record.relevance for record in truth), reverse=True)[:top_k]
    dcg = _dcg(gains)
    ideal_dcg = _dcg(ideal)
    ndcg = dcg / ideal_dcg if ideal_dcg else 0.0

    unique_evaluations = len({record.evaluation_id for record in top})
    duplicate_share = 0.0
    if top:
        duplicate_share = 1.0 - (unique_evaluations / len(top))

    return {
        "recall_at_k": round(recall_any, 6),
        "recall_any_at_k": round(recall_any, 6),
        "recall_supporting_at_k": round(recall_supporting, 6),
        "recall_direct_at_k": round(recall_direct, 6),
        "graded_recall_at_k": round(graded_recall, 6),
        "reciprocal_rank": round(reciprocal_rank, 6),
        "ndcg_at_k": round(ndcg, 6),
        "relevant_count": relevant_total,
        "retrieved_relevant_count": retrieved_relevant,
        "supporting_count": supporting_total,
        "retrieved_supporting_count": retrieved_supporting,
        "direct_count": direct_total,
        "retrieved_direct_count": retrieved_direct,
        "graded_gain_total": graded_total,
        "retrieved_graded_gain": retrieved_graded,
        "unique_evaluations_at_k": unique_evaluations,
        "duplicate_share_at_k": round(duplicate_share, 6),
    }


def _threshold_recall(
    truth: list[RankerTrainingRecord],
    selected: list[RankerTrainingRecord],
    *,
    threshold: int,
) -> float:
    total = sum(record.relevance >= threshold for record in truth)
    if total == 0:
        return 0.0
    available = sum(record.relevance >= threshold for record in selected)
    return round(available / total, 6)


def _direct_recall(
    truth: list[RankerTrainingRecord],
    selected: list[RankerTrainingRecord],
) -> float:
    total = sum(record.relevance == 3 for record in truth)
    if total == 0:
        return 0.0
    available = sum(record.relevance == 3 for record in selected)
    return round(available / total, 6)


def _graded_recall(
    truth: list[RankerTrainingRecord],
    selected: list[RankerTrainingRecord],
) -> float:
    total = sum(record.relevance for record in truth)
    if total == 0:
        return 0.0
    available = sum(record.relevance for record in selected)
    return round(available / total, 6)


def _normalize_scores(
    records: list[RankerTrainingRecord],
    score: Callable[[RankerTrainingRecord], float],
) -> dict[str, float]:
    raw = {str(record.chunk_id): float(score(record)) for record in records}
    low = min(raw.values())
    high = max(raw.values())
    if math.isclose(low, high):
        return {chunk_id: 0.5 for chunk_id in raw}
    span = high - low
    return {chunk_id: (value - low) / span for chunk_id, value in raw.items()}


def _validate_alphas(alphas: list[float]) -> list[float]:
    if not alphas:
        raise ValueError("Provide at least one fusion alpha.")
    invalid = [alpha for alpha in alphas if alpha < 0.0 or alpha > 1.0]
    if invalid:
        raise ValueError("Fusion alphas must be between 0 and 1.")
    return sorted(set(round(float(alpha), 6) for alpha in alphas))


def _fusion_is_feasible(
    metrics: dict[str, float | int],
    baseline: dict[str, float | int],
    *,
    diversity_tolerance: float,
) -> bool:
    epsilon = 1e-9
    return (
        float(metrics["mean_recall_supporting_at_k"])
        + epsilon
        >= float(baseline["mean_recall_supporting_at_k"])
        and float(metrics["mean_recall_direct_at_k"])
        + epsilon
        >= float(baseline["mean_recall_direct_at_k"])
        and float(metrics["mean_reciprocal_rank"])
        + epsilon
        >= float(baseline["mean_reciprocal_rank"])
        and float(metrics["mean_duplicate_share_at_k"])
        <= float(baseline["mean_duplicate_share_at_k"]) + diversity_tolerance + epsilon
    )


def _dcg(relevances: list[int]) -> float:
    return sum(
        ((2**relevance) - 1) / math.log2(index + 2)
        for index, relevance in enumerate(relevances)
    )


def _aggregate(
    query_results: list[dict[str, object]],
    *,
    key: str,
) -> dict[str, float | int]:
    metrics = [item[key] for item in query_results]
    return {
        "query_count": len(metrics),
        "mean_recall_at_k": round(
            mean(float(item["recall_at_k"]) for item in metrics),
            6,
        ),
        "mean_recall_any_at_k": round(
            mean(float(item["recall_any_at_k"]) for item in metrics),
            6,
        ),
        "mean_recall_supporting_at_k": round(
            mean(float(item["recall_supporting_at_k"]) for item in metrics),
            6,
        ),
        "mean_recall_direct_at_k": round(
            mean(float(item["recall_direct_at_k"]) for item in metrics),
            6,
        ),
        "mean_graded_recall_at_k": round(
            mean(float(item["graded_recall_at_k"]) for item in metrics),
            6,
        ),
        "mean_reciprocal_rank": round(
            mean(float(item["reciprocal_rank"]) for item in metrics),
            6,
        ),
        "mean_ndcg_at_k": round(
            mean(float(item["ndcg_at_k"]) for item in metrics),
            6,
        ),
        "mean_unique_evaluations_at_k": round(
            mean(float(item["unique_evaluations_at_k"]) for item in metrics),
            6,
        ),
        "mean_duplicate_share_at_k": round(
            mean(float(item["duplicate_share_at_k"]) for item in metrics),
            6,
        ),
    }


def _aggregate_candidate_ceilings(
    query_results: list[dict[str, object]],
) -> dict[str, float]:
    return {
        "mean_candidate_recall_ceiling": round(
            mean(float(item["candidate_recall_ceilings"]["any"]) for item in query_results),
            6,
        ),
        "mean_candidate_recall_supporting_ceiling": round(
            mean(
                float(item["candidate_recall_ceilings"]["supporting"])
                for item in query_results
            ),
            6,
        ),
        "mean_candidate_recall_direct_ceiling": round(
            mean(
                float(item["candidate_recall_ceilings"]["direct"])
                for item in query_results
            ),
            6,
        ),
        "mean_candidate_graded_recall_ceiling": round(
            mean(
                float(item["candidate_recall_ceilings"]["graded"])
                for item in query_results
            ),
            6,
        ),
    }


def _family_summaries(
    query_results: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in query_results:
        grouped[str(item["family"])].append(item)

    outputs: list[dict[str, object]] = []
    for family in sorted(grouped):
        items = grouped[family]
        outputs.append(
            {
                "family": family,
                "query_count": len(items),
                **_aggregate_candidate_ceilings(items),
                "baseline": _aggregate(items, key="baseline"),
                "reranked": _aggregate(items, key="reranked"),
            }
        )
    return outputs


def _family_metric_summaries(
    query_results: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in query_results:
        grouped[str(item["family"])].append(item)
    return [
        {
            "family": family,
            "query_count": len(items),
            "metrics": _aggregate(items, key="metrics"),
        }
        for family, items in sorted(grouped.items())
    ]
