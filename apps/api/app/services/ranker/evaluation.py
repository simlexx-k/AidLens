import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

from app.schemas.benchmark import RankerTrainingRecord


def evaluate_aidranker_model(
    records: list[RankerTrainingRecord],
    model_path: str,
    *,
    candidate_mode: str = "semantic",
    top_k: int = 10,
    batch_size: int = 32,
) -> dict[str, object]:
    """Compare first-stage ranking with AidRanker on the same candidate set."""

    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "AidRanker evaluation requires the API ML extras. "
            "Install with `pip install -e '.[ml]'` or use the ML Docker image."
        ) from exc

    model = CrossEncoder(model_path)
    candidate_records = [
        record for record in records if _is_candidate(record, candidate_mode)
    ]
    pairs = [(record.query, record.text) for record in candidate_records]
    predictions = model.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=True,
    )
    reranker_scores = {
        str(record.chunk_id): float(score)
        for record, score in zip(candidate_records, predictions, strict=True)
    }
    return evaluate_reranking(
        records,
        reranker_scores,
        candidate_mode=candidate_mode,
        top_k=top_k,
        model_path=model_path,
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

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if candidate_mode not in {"lexical", "semantic", "hybrid"}:
        raise ValueError(f"Unsupported candidate mode: {candidate_mode}")

    grouped: dict[str, list[RankerTrainingRecord]] = defaultdict(list)
    for record in records:
        grouped[record.query_id].append(record)

    query_results: list[dict[str, object]] = []
    for query_id in sorted(grouped):
        all_records = grouped[query_id]
        candidates = [
            record for record in all_records if _is_candidate(record, candidate_mode)
        ]
        if not candidates:
            raise ValueError(
                f"Query {query_id} has no candidates from mode {candidate_mode!r}."
            )

        missing_scores = [
            str(record.chunk_id)
            for record in candidates
            if str(record.chunk_id) not in reranker_scores
        ]
        if missing_scores:
            raise ValueError(
                f"Query {query_id} is missing {len(missing_scores)} reranker scores."
            )

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
        family = all_records[0].family
        query_results.append(
            {
                "query_id": query_id,
                "family": family,
                "candidate_count": len(candidates),
                "candidate_recall_ceiling": _candidate_recall_ceiling(
                    all_records,
                    candidates,
                ),
                "baseline": _ranking_metrics(all_records, baseline, top_k=top_k),
                "reranked": _ranking_metrics(all_records, reranked, top_k=top_k),
            }
        )

    report: dict[str, object] = {
        "model": model_path,
        "candidate_mode": candidate_mode,
        "top_k": top_k,
        "query_count": len(query_results),
        "baseline": _aggregate(query_results, key="baseline"),
        "reranked": _aggregate(query_results, key="reranked"),
        "mean_candidate_recall_ceiling": round(
            mean(float(item["candidate_recall_ceiling"]) for item in query_results),
            6,
        ),
        "families": _family_summaries(query_results),
        "queries": query_results,
    }
    return report


def write_ranker_report(report: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
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


def _candidate_recall_ceiling(
    truth: list[RankerTrainingRecord],
    candidates: list[RankerTrainingRecord],
) -> float:
    total = sum(record.relevance > 0 for record in truth)
    if total == 0:
        return 0.0
    available = sum(record.relevance > 0 for record in candidates)
    return round(available / total, 6)


def _ranking_metrics(
    truth: list[RankerTrainingRecord],
    ranked: list[RankerTrainingRecord],
    *,
    top_k: int,
) -> dict[str, float | int]:
    relevant_total = sum(record.relevance > 0 for record in truth)
    top = ranked[:top_k]
    retrieved_relevant = sum(record.relevance > 0 for record in top)
    recall = retrieved_relevant / relevant_total if relevant_total else 0.0

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
        "recall_at_k": round(recall, 6),
        "reciprocal_rank": round(reciprocal_rank, 6),
        "ndcg_at_k": round(ndcg, 6),
        "relevant_count": relevant_total,
        "retrieved_relevant_count": retrieved_relevant,
        "unique_evaluations_at_k": unique_evaluations,
        "duplicate_share_at_k": round(duplicate_share, 6),
    }


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
                "mean_candidate_recall_ceiling": round(
                    mean(float(item["candidate_recall_ceiling"]) for item in items),
                    6,
                ),
                "baseline": _aggregate(items, key="baseline"),
                "reranked": _aggregate(items, key="reranked"),
            }
        )
    return outputs
