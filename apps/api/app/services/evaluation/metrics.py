import math
import re

from app.schemas.benchmark import RelevanceJudgment, RetrievalMetrics
from app.schemas.evaluation import EvidenceSearchHit


def evaluate_hits(
    hits: list[EvidenceSearchHit],
    judgments: list[RelevanceJudgment],
    *,
    k: int,
) -> RetrievalMetrics:
    ranked = hits[:k]
    matched: set[tuple[str, str | None, str | None]] = set()
    gains: list[int] = []

    for hit in ranked:
        best: RelevanceJudgment | None = None
        for judgment in judgments:
            key = _judgment_key(judgment)
            if key in matched:
                continue
            if not _matches_judgment(hit, judgment):
                continue
            if best is None or judgment.relevance > best.relevance:
                best = judgment
        if best is None:
            gains.append(0)
            continue
        matched.add(_judgment_key(best))
        gains.append(best.relevance)

    relevant_count = len({_judgment_key(item) for item in judgments})
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

    unique_evaluations = len({hit.evaluation_id for hit in ranked})
    duplicate_share = (
        (len(ranked) - unique_evaluations) / len(ranked) if ranked else 0.0
    )

    return RetrievalMetrics(
        recall_at_k=round(recall, 6),
        reciprocal_rank=round(reciprocal_rank, 6),
        ndcg_at_k=round(ndcg, 6),
        relevant_count=relevant_count,
        retrieved_relevant_count=retrieved_relevant_count,
        unique_evaluations_at_k=unique_evaluations,
        duplicate_share_at_k=round(duplicate_share, 6),
    )


def _matches_judgment(
    hit: EvidenceSearchHit,
    judgment: RelevanceJudgment,
) -> bool:
    if judgment.evaluation_id != hit.evaluation_id:
        return False
    if judgment.section is not None and judgment.section != hit.section:
        return False
    if judgment.anchor_text is not None:
        return _normalize_text(judgment.anchor_text) in _normalize_text(hit.text)
    return True


def _judgment_key(judgment: RelevanceJudgment) -> tuple[str, str | None, str | None]:
    anchor = _normalize_text(judgment.anchor_text) if judgment.anchor_text else None
    return (judgment.evaluation_id, judgment.section, anchor)


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _dcg(gains: list[int]) -> float:
    return sum(
        (2**gain - 1) / math.log2(rank + 1)
        for rank, gain in enumerate(gains, start=1)
        if gain > 0
    )
