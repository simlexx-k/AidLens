import uuid

from app.schemas.evaluation import EvidenceSearchHit
from app.services.evaluation.candidates import diversify_candidates


def _hit(evaluation_id: str, score: float) -> EvidenceSearchHit:
    return EvidenceSearchHit(
        chunk_id=uuid.uuid4(),
        evaluation_id=evaluation_id,
        title=f"Evaluation {evaluation_id}",
        publication_year=2024,
        section="findings",
        text="Relevant evidence passage.",
        score=score,
        semantic_score=score,
        retrieval_sources=["semantic"],
        source_url=f"https://example.test/{evaluation_id}",
    )


def test_diversify_candidates_caps_each_evaluation_and_preserves_retrieval_rank() -> None:
    hits = [
        _hit("A", 0.9),
        _hit("A", 0.89),
        _hit("A", 0.88),
        _hit("B", 0.87),
        _hit("B", 0.86),
        _hit("C", 0.85),
        _hit("D", 0.84),
    ]

    selected = diversify_candidates(hits, top_k=5, max_per_evaluation=2)

    assert [(rank, hit.evaluation_id) for rank, hit in selected] == [
        (1, "A"),
        (2, "A"),
        (4, "B"),
        (5, "B"),
        (6, "C"),
    ]


def test_diversify_candidates_returns_available_pool_when_limit_is_tight() -> None:
    hits = [_hit("A", 0.9), _hit("A", 0.8), _hit("B", 0.7)]

    selected = diversify_candidates(hits, top_k=5, max_per_evaluation=1)

    assert [(rank, hit.evaluation_id) for rank, hit in selected] == [
        (1, "A"),
        (3, "B"),
    ]
