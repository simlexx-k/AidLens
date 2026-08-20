import uuid

from app.schemas.evaluation import EvidenceSearchHit
from app.services.search.intelligence import (
    evidence_role_for_section,
    group_evidence_hits,
    synthesize_evidence_groups,
)


def _hit(
    evaluation_id: str,
    *,
    section: str | None,
    score: float,
    project_title: str | None = None,
    locations: list[str] | None = None,
    institutions: list[str] | None = None,
    keywords: list[str] | None = None,
) -> EvidenceSearchHit:
    return EvidenceSearchHit(
        chunk_id=uuid.uuid4(),
        evaluation_id=evaluation_id,
        title=f"Evaluation {evaluation_id}",
        project_title=project_title,
        publication_year=2024,
        section=section,
        evidence_role=evidence_role_for_section(section),
        text="Evidence passage",
        score=score,
        semantic_score=score,
        retrieval_sources=["semantic"],
        locations=locations if locations is not None else ["Kenya"],
        institutions=institutions if institutions is not None else ["USAID"],
        keywords=keywords if keywords is not None else ["education"],
        source_url="https://example.test/evaluation",
    )


def test_evidence_role_classification_is_conservative() -> None:
    assert evidence_role_for_section("findings") == "outcome"
    assert evidence_role_for_section("sustainability") == "sustainability"
    assert evidence_role_for_section("recommendations") == "recommendation"
    assert evidence_role_for_section("methodology") == "method"
    assert evidence_role_for_section("executive_summary") == "supporting"
    assert evidence_role_for_section(None) == "supporting"


def test_grouping_preserves_rank_order_and_intervention_metadata() -> None:
    hits = [
        _hit("B", section="findings", score=0.9, project_title="Teacher coaching"),
        _hit("A", section="methodology", score=0.8),
        _hit("B", section="recommendations", score=0.7, project_title="Teacher coaching"),
    ]

    groups = group_evidence_hits(hits)

    assert [group.evaluation_id for group in groups] == ["B", "A"]
    assert groups[0].intervention == "Teacher coaching"
    assert groups[0].outcome_evidence_count == 1
    assert groups[0].evidence_roles == ["outcome", "recommendation"]
    assert [hit.score for hit in groups[0].hits] == [0.9, 0.7]


def test_grouping_falls_back_to_evaluation_title_for_intervention() -> None:
    group = group_evidence_hits([_hit("A", section=None, score=0.8)])[0]

    assert group.intervention == "Evaluation A"
    assert group.outcome_evidence_count == 0


def test_synthesis_counts_roles_and_recurring_facets_across_evaluations() -> None:
    groups = group_evidence_hits(
        [
            _hit(
                "A",
                section="findings",
                score=0.9,
                locations=["Kenya"],
                institutions=["USAID"],
                keywords=["Education", "Youth"],
            ),
            _hit(
                "A",
                section="recommendations",
                score=0.8,
                locations=["Kenya"],
                institutions=["USAID"],
                keywords=["Education", "Youth"],
            ),
            _hit(
                "B",
                section="findings",
                score=0.7,
                locations=["kenya"],
                institutions=["UNICEF"],
                keywords=["education", "Health"],
            ),
            _hit(
                "C",
                section="methodology",
                score=0.6,
                locations=["Uganda"],
                institutions=["USAID"],
                keywords=["Agriculture"],
            ),
        ]
    )

    synthesis = synthesize_evidence_groups(groups)

    assert synthesis.evaluation_count == 3
    assert synthesis.passage_count == 4
    assert synthesis.outcome_evaluation_count == 2
    role_summary = [
        (item.role, item.evaluation_count, item.passage_count)
        for item in synthesis.role_coverage
    ]
    assert role_summary == [
        ("outcome", 2, 2),
        ("recommendation", 1, 1),
        ("method", 1, 1),
    ]
    facet_summary = [
        (facet.kind, facet.value, facet.evaluation_count)
        for facet in synthesis.recurring_facets
    ]
    assert facet_summary == [
        ("location", "Kenya", 2),
        ("institution", "USAID", 2),
        ("keyword", "Education", 2),
    ]
    assert synthesis.caveats


def test_transferability_pairs_are_metadata_overlap_not_zero_overlap_pairs() -> None:
    groups = group_evidence_hits(
        [
            _hit(
                "A",
                section="findings",
                score=0.9,
                locations=["Kenya"],
                institutions=["USAID"],
                keywords=["Education", "Youth"],
            ),
            _hit(
                "B",
                section="findings",
                score=0.8,
                locations=["Kenya"],
                institutions=["UNICEF"],
                keywords=["Education", "Health"],
            ),
            _hit(
                "C",
                section="findings",
                score=0.7,
                locations=["Uganda"],
                institutions=["USAID"],
                keywords=["Agriculture"],
            ),
        ]
    )

    pairs = synthesize_evidence_groups(groups).transferability_pairs

    assert [(pair.left_evaluation_id, pair.right_evaluation_id) for pair in pairs] == [
        ("A", "B"),
        ("A", "C"),
    ]
    assert pairs[0].context_overlap_score == 0.333
    assert pairs[0].shared_signal_count == 2
    assert pairs[0].shared_locations == ["Kenya"]
    assert pairs[0].shared_keywords == ["Education"]
    assert pairs[1].context_overlap_score == 0.167
    assert pairs[1].shared_institutions == ["USAID"]
