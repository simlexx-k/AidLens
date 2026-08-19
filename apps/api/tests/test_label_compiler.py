import uuid

import pytest

from app.schemas.benchmark import RankingCandidate, RankingCandidateSet
from app.services.evaluation.labels import compile_labeled_candidates, make_anchor


def _candidate(*, rank: int, relevance: int | None, text: str) -> RankingCandidate:
    return RankingCandidate(
        rank=rank,
        retrieval_rank=rank,
        chunk_id=uuid.uuid4(),
        evaluation_id=f"EVAL-{rank}",
        title=f"Evaluation {rank}",
        section="findings",
        text=text,
        score=1.0 / rank,
        semantic_score=0.8,
        relevance=relevance,
    )


def test_compile_labels_builds_benchmark_and_ranker_records() -> None:
    item = RankingCandidateSet(
        query_id="q1",
        query="What worked?",
        family="intervention_outcomes",
        mode="hybrid",
        candidates=[
            _candidate(
                rank=1,
                relevance=3,
                text="The evaluation found that the intervention improved household food security substantially.",
            ),
            _candidate(
                rank=2,
                relevance=0,
                text="This passage is a high-ranked but irrelevant hard negative for the query.",
            ),
        ],
    )

    judgments, records = compile_labeled_candidates([item])

    assert len(judgments) == 1
    assert judgments[0].family == "intervention_outcomes"
    assert len(judgments[0].judgments) == 1
    assert judgments[0].judgments[0].relevance == 3
    assert "improved household food security" in judgments[0].judgments[0].anchor_text
    assert len(records) == 2
    assert [record.relevance for record in records] == [3, 0]
    assert records[1].family == "intervention_outcomes"


def test_compile_labels_rejects_partially_labeled_pool() -> None:
    item = RankingCandidateSet(
        query_id="q1",
        query="What worked?",
        family="general",
        mode="hybrid",
        candidates=[
            _candidate(rank=1, relevance=3, text="A clearly relevant evidence passage."),
            _candidate(rank=2, relevance=None, text="This candidate has not been reviewed yet."),
        ],
    )

    with pytest.raises(ValueError, match="unlabeled candidates"):
        compile_labeled_candidates([item])


def test_make_anchor_is_bounded_and_word_aligned() -> None:
    anchor = make_anchor("word " * 100, max_chars=100)

    assert len(anchor) <= 100
    assert not anchor.endswith(" ")
