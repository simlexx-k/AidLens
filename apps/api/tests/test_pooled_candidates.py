import uuid

from app.schemas.benchmark import RankingCandidate, RankingCandidateSet
from app.schemas.evaluation import EvidenceSearchHit
from app.services.evaluation.candidates import carry_forward_labels, pool_candidates


def _hit(
    evaluation_id: str,
    *,
    chunk_id: uuid.UUID | None = None,
    source: str = "semantic",
    score: float = 1.0,
) -> EvidenceSearchHit:
    return EvidenceSearchHit(
        chunk_id=chunk_id or uuid.uuid4(),
        evaluation_id=evaluation_id,
        title=f"Evaluation {evaluation_id}",
        publication_year=2024,
        section="findings",
        text="Relevant evidence passage.",
        score=score,
        lexical_score=(score if source == "lexical" else None),
        semantic_score=(score if source == "semantic" else None),
        retrieval_sources=[source],
        source_url=f"https://example.test/{evaluation_id}",
    )


def test_pool_candidates_unions_modes_and_tracks_each_rank() -> None:
    shared_id = uuid.uuid4()
    lexical_a = _hit("A", source="lexical")
    lexical_b = _hit("B", source="lexical")
    shared_lexical = _hit("S", chunk_id=shared_id, source="lexical")
    shared_semantic = _hit("S", chunk_id=shared_id, source="semantic")
    shared_hybrid = _hit("S", chunk_id=shared_id, source="hybrid")
    semantic_c = _hit("C", source="semantic")
    hybrid_d = _hit("D", source="hybrid")

    pooled = pool_candidates(
        {
            "lexical": [lexical_a, lexical_b, shared_lexical],
            "semantic": [shared_semantic, semantic_c],
            "hybrid": [hybrid_d, shared_hybrid],
        },
        mode_order=["lexical", "semantic", "hybrid"],
        max_per_evaluation=5,
    )

    assert [item.hit.evaluation_id for item in pooled] == ["A", "S", "D", "C", "B"]
    shared = next(item for item in pooled if item.hit.chunk_id == shared_id)
    assert shared.retrieval_rank == 1
    assert shared.retrieval_modes == ["lexical", "semantic", "hybrid"]
    assert shared.mode_ranks == {"lexical": 3, "semantic": 1, "hybrid": 2}
    assert shared.hit.lexical_score is not None
    assert shared.hit.semantic_score is not None


def test_pool_candidates_applies_cap_after_cross_mode_pooling() -> None:
    pooled = pool_candidates(
        {
            "lexical": [_hit("A", source="lexical"), _hit("A", source="lexical")],
            "semantic": [_hit("A", source="semantic"), _hit("B", source="semantic")],
            "hybrid": [_hit("A", source="hybrid"), _hit("C", source="hybrid")],
        },
        mode_order=["lexical", "semantic", "hybrid"],
        max_per_evaluation=2,
    )

    ids = [item.hit.evaluation_id for item in pooled]
    assert ids.count("A") == 2
    assert "B" in ids
    assert "C" in ids


def _candidate(chunk_id: uuid.UUID, relevance: int | None) -> RankingCandidate:
    return RankingCandidate(
        rank=1,
        retrieval_rank=1,
        chunk_id=chunk_id,
        evaluation_id="A",
        title="Evaluation A",
        section="findings",
        text="Relevant evidence passage.",
        score=1.0,
        relevance=relevance,
    )


def test_carry_forward_labels_reuses_only_matching_query_chunk_pairs() -> None:
    reviewed_id = uuid.uuid4()
    new_id = uuid.uuid4()
    previous = [
        RankingCandidateSet(
            query_id="q001",
            query="What worked?",
            family="outcomes",
            mode="hybrid",
            candidates=[_candidate(reviewed_id, 3)],
        )
    ]
    current = [
        RankingCandidateSet(
            query_id="q001",
            query="What worked?",
            family="outcomes",
            mode="pooled",
            candidates=[
                _candidate(reviewed_id, None),
                _candidate(new_id, None).model_copy(update={"rank": 2}),
            ],
        )
    ]

    copied = carry_forward_labels(current, previous)

    assert copied == 1
    assert current[0].candidates[0].relevance == 3
    assert current[0].candidates[1].relevance is None
