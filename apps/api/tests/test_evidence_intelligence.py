import uuid

from app.schemas.evaluation import EvidenceSearchHit
from app.services.search.intelligence import evidence_role_for_section, group_evidence_hits


def _hit(
    evaluation_id: str,
    *,
    section: str | None,
    score: float,
    project_title: str | None = None,
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
        locations=["Kenya"],
        institutions=["USAID"],
        keywords=["education"],
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
