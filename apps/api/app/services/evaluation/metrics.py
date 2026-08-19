import math

from app.schemas.benchmark import RelevanceJudgment, RetrievalMetrics
from app.schemas.evaluation import EvidenceSearchHit


def evaluate_hits(
    hits: list[EvidenceSearchHit],
    judgments: list[RelevanceJudgment],
    *,
    k: int,
) -> RetrievalMetrics:
    ranked = hits[:k]
    matched: set[tuple[str, str | None]] = set()
    gains: list[int] = []

    for hit in ranked:
        best: RelevanceJudgment | None = None
        for judgment in judgments:
            key = (judgment.evaluation_id, judgment.section)
            if key in matched:
                continue
            if judgment.evaluation_id != hit.evaluation_id:
                continue
            if judgment.section is not None and judgment.section != hit.section:
                continue
            if best is None or judgment.relevance > best.relevance:
                best = judgment
        if best is None:
            gains.append(0)
            continue
        matched.add((best.evaluation_id, best.section))
        gains.append(best.relevance)

    relevant_count = len({(item.evaluation_id, item.section) for item in judgments})
    retrieved_relevant_count = len(matched)
    recall = retrieved_relevant_count / relevant_count if relevant_count else 0.0

    first_relevant_rank = next(
        (rank for rank, gain in enumerate(gains, start=1) if gain > 0),
        None,
    )
    reciprocal_rank = 1.0 / first_relevant_rank if first_relevant_rank else 0.0

    dcg = _dcg(gains)
    ideal_gains = sorted((item.relevance for item in judgments), reverse=True)[:k]
    ideal_dcg = _dcg(ideal_gains)
    ndcg = dcg / ideal_dcg if ideal_dcg else 0.0

    return RetrievalMetrics(
        recall_at_k=round(recall, 6),
        reciprocal_rank=round(reciprocal_rank, 6),
        ndcg_at_k=round(ndcg, 6),
        relevant_count=relevant_count,
        retrieved_relevant_count=retrieved_relevant_count,
    )


def _dcg(gains: list[int]) -> float:
    return sum(
        (2**gain - 1) / math.log2(rank + 1)
        for rank, gain in enumerate(gains, start=1)
        if gain > 0
    )
