import uuid

from app.schemas.benchmark import RelevanceJudgment
from app.schemas.evaluation import EvidenceSearchHit
from app.services.evaluation.metrics import evaluate_hits


def _hit(evaluation_id: str, section: str | None) -> EvidenceSearchHit:
    return EvidenceSearchHit(
        chunk_id=uuid.uuid4(),
        evaluation_id=evaluation_id,
        title=f"Evaluation {evaluation_id}",
        section=section,
        text="Evidence passage",
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
