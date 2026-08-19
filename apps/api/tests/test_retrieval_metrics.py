import uuid

from app.schemas.benchmark import RelevanceJudgment
from app.schemas.evaluation import EvidenceSearchHit
from app.services.evaluation.metrics import evaluate_hits


def _hit(
    evaluation_id: str,
    section: str | None,
    text: str = "Evidence passage",
) -> EvidenceSearchHit:
    return EvidenceSearchHit(
        chunk_id=uuid.uuid4(),
        evaluation_id=evaluation_id,
        title=f"Evaluation {evaluation_id}",
        section=section,
        text=text,
        score=1.0,
        retrieval_sources=["test"],
        source_url="https://example.com/evaluation",
    )


def test_metrics_use_stable_evaluation_and_section_judgments() -> None:
    judgments = [
        RelevanceJudgment(evaluation_id="A", section="findings", relevance=3),
        RelevanceJudgment(evaluation_id="B", section="recommendations", relevance=2),
    ]
    hits = [
        _hit("X", "findings"),
        _hit("B", "recommendations"),
        _hit("A", "findings"),
    ]

    metrics = evaluate_hits(hits, judgments, k=3)

    assert metrics.recall_at_k == 1.0
    assert metrics.reciprocal_rank == 0.5
    assert 0.0 < metrics.ndcg_at_k < 1.0
    assert metrics.retrieved_relevant_count == 2
    assert metrics.unique_evaluations_at_k == 3
    assert metrics.duplicate_share_at_k == 0.0


def test_duplicate_chunks_do_not_inflate_recall() -> None:
    judgments = [
        RelevanceJudgment(evaluation_id="A", section="findings", relevance=3),
    ]
    hits = [
        _hit("A", "findings"),
        _hit("A", "findings"),
    ]

    metrics = evaluate_hits(hits, judgments, k=2)

    assert metrics.recall_at_k == 1.0
    assert metrics.retrieved_relevant_count == 1
    assert metrics.unique_evaluations_at_k == 1
    assert metrics.duplicate_share_at_k == 0.5


def test_anchor_text_distinguishes_passages_within_same_section() -> None:
    judgments = [
        RelevanceJudgment(
            evaluation_id="A",
            section="findings",
            anchor_text="instrumental in supporting transitory food insecure households",
            relevance=3,
        )
    ]
    hits = [
        _hit("A", "findings", "General findings about food insecurity."),
        _hit(
            "A",
            "findings",
            "The evaluation found that JEOP was instrumental in supporting transitory "
            "food insecure households across the target regions.",
        ),
    ]

    metrics = evaluate_hits(hits, judgments, k=2)

    assert metrics.recall_at_k == 1.0
    assert metrics.reciprocal_rank == 0.5
    assert metrics.retrieved_relevant_count == 1
