import json
import re
from pathlib import Path

from app.schemas.benchmark import (
    BenchmarkQuery,
    RankerTrainingRecord,
    RankingCandidateSet,
    RelevanceJudgment,
)


def compile_labeled_candidates(
    candidate_sets: list[RankingCandidateSet],
) -> tuple[list[BenchmarkQuery], list[RankerTrainingRecord]]:
    """Compile fully labeled candidate pools into benchmark and ranker datasets."""

    benchmark_queries: list[BenchmarkQuery] = []
    ranker_records: list[RankerTrainingRecord] = []

    for item in candidate_sets:
        unlabeled = [
            candidate.rank
            for candidate in item.candidates
            if candidate.relevance is None
        ]
        if unlabeled:
            ranks = ", ".join(str(rank) for rank in unlabeled[:10])
            suffix = "..." if len(unlabeled) > 10 else ""
            raise ValueError(
                f"Query {item.query_id} has unlabeled candidates at ranks {ranks}{suffix}. "
                "Assign every candidate a relevance value from 0 to 3 before compiling."
            )

        positives = [
            candidate
            for candidate in item.candidates
            if candidate.relevance and candidate.relevance > 0
        ]
        if not positives:
            raise ValueError(
                f"Query {item.query_id} has no positive judgments. "
                "Keep the query out of the benchmark or add at least one reviewed positive."
            )

        judgments = [
            RelevanceJudgment(
                evaluation_id=candidate.evaluation_id,
                section=candidate.section,
                anchor_text=make_anchor(candidate.text),
                relevance=int(candidate.relevance),
            )
            for candidate in positives
        ]
        benchmark_queries.append(
            BenchmarkQuery(
                query_id=item.query_id,
                query=item.query,
                family=item.family,
                judgments=judgments,
                notes="Compiled from a fully human-labeled diversified candidate pool.",
            )
        )

        ranker_records.extend(
            RankerTrainingRecord(
                query_id=item.query_id,
                family=item.family,
                query=item.query,
                chunk_id=candidate.chunk_id,
                evaluation_id=candidate.evaluation_id,
                title=candidate.title,
                section=candidate.section,
                text=candidate.text,
                relevance=int(candidate.relevance),
                retrieval_rank=candidate.retrieval_rank,
                score=candidate.score,
                lexical_score=candidate.lexical_score,
                semantic_score=candidate.semantic_score,
                retrieval_modes=candidate.retrieval_modes,
                mode_ranks=candidate.mode_ranks,
            )
            for candidate in item.candidates
            if candidate.relevance is not None
        )

    return benchmark_queries, ranker_records


def make_anchor(text: str, *, max_chars: int = 220) -> str:
    """Create a stable normalized passage anchor for rechunking-resistant judgments."""

    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= max_chars:
        return normalized

    shortened = normalized[:max_chars]
    last_space = shortened.rfind(" ")
    if last_space >= 80:
        shortened = shortened[:last_space]
    return shortened.rstrip(" ,;:-")


def write_benchmark_queries(items: list[BenchmarkQuery], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            payload = json.dumps(item.model_dump(mode="json"), ensure_ascii=False)
            handle.write(payload + "\n")


def write_ranker_records(items: list[RankerTrainingRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            payload = json.dumps(item.model_dump(mode="json"), ensure_ascii=False)
            handle.write(payload + "\n")
