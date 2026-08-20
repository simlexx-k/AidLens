import uuid

from app.schemas.evaluation import EvidenceSearchHit
from app.services.search.claims import extract_grounded_claims, stance_coverage
from app.services.search.intelligence import group_evidence_hits


def _hit(text: str, *, section: str = "findings", evaluation_id: str = "A") -> EvidenceSearchHit:
    return EvidenceSearchHit(
        chunk_id=uuid.uuid4(),
        evaluation_id=evaluation_id,
        title=f"Evaluation {evaluation_id}",
        project_title="Teacher coaching",
        publication_year=2024,
        section=section,
        evidence_role="outcome" if section == "findings" else "method",
        text=text,
        score=0.9,
        semantic_score=0.8,
        retrieval_sources=["semantic"],
        locations=["Kenya"],
        institutions=["USAID"],
        keywords=["education"],
        source_url="https://example.test/evaluation",
    )


def _claim(text: str, *, section: str = "findings"):
    hit = _hit(text, section=section)
    group = group_evidence_hits([hit])
    claim = extract_grounded_claims(group)[0]
    return hit, claim


def test_supporting_claim_is_exact_source_span_with_condition() -> None:
    text = (
        "The intervention improved household food security when local partners "
        "met weekly. A separate paragraph described implementation logistics."
    )
    hit, claim = _claim(text)

    assert claim.stance == "supports"
    assert claim.confidence == 0.84
    assert "improved" in [item.casefold() for item in claim.stance_basis]
    assert claim.explicit_conditions == ["when local partners met weekly"]
    assert hit.text[claim.source_span_start : claim.source_span_end] == claim.statement
    assert claim.statement == (
        "The intervention improved household food security when local partners met weekly."
    )


def test_negative_effect_language_is_classified_as_contradicting() -> None:
    _, claim = _claim("The evaluation found no significant effect on learning outcomes.")

    assert claim.stance == "contradicts"
    assert claim.stance_basis == ["no significant effect"]


def test_negated_improvement_without_contrast_is_not_mixed() -> None:
    _, claim = _claim("The intervention did not improve learning outcomes.")

    assert claim.stance == "contradicts"
    assert claim.stance_basis == ["did not improve"]


def test_conflicting_explicit_signals_are_mixed() -> None:
    _, claim = _claim(
        "The intervention improved attendance, but did not improve learning outcomes."
    )

    assert claim.stance == "mixed"
    assert "improved" in [item.casefold() for item in claim.stance_basis]
    assert any("did not improve" in item.casefold() for item in claim.stance_basis)


def test_effect_language_without_direction_abstains_as_insufficient() -> None:
    _, claim = _claim("The evaluation assessed the impact of the program on household income.")

    assert claim.stance == "insufficient"
    assert claim.confidence < 0.5
    assert claim.stance_basis == ["impact"]


def test_section_role_does_not_create_effect_stance() -> None:
    _, claim = _claim("The evaluation described survey procedures and sampling methods.")

    assert claim.evidence_role == "outcome"
    assert claim.stance == "not_an_effect_claim"
    assert claim.stance_basis == []


def test_method_passage_can_only_gain_stance_from_its_text() -> None:
    _, claim = _claim(
        "The methodology notes that the intervention improved attendance in pilot schools.",
        section="methodology",
    )

    assert claim.evidence_role == "method"
    assert claim.stance == "supports"


def test_claims_limit_effect_sentences_per_passage_without_rewriting() -> None:
    text = (
        "The intervention improved attendance. "
        "There was no significant effect on learning outcomes. "
        "The program strengthened school management."
    )
    hit = _hit(text)
    claims = extract_grounded_claims(group_evidence_hits([hit]))

    assert len(claims) == 2
    assert [claim.stance for claim in claims] == ["supports", "contradicts"]
    for claim in claims:
        assert hit.text[claim.source_span_start : claim.source_span_end] == claim.statement


def test_stance_coverage_counts_claims_and_distinct_evaluations() -> None:
    groups = group_evidence_hits(
        [
            _hit("The intervention improved attendance.", evaluation_id="A"),
            _hit("The program strengthened school management.", evaluation_id="A"),
            _hit("There was no significant effect on learning.", evaluation_id="B"),
            _hit("The evaluation described survey procedures.", evaluation_id="C"),
        ]
    )
    coverage = stance_coverage(extract_grounded_claims(groups))
    summary = {
        item.stance: (item.evaluation_count, item.claim_count)
        for item in coverage
    }

    assert summary["supports"] == (1, 2)
    assert summary["contradicts"] == (1, 1)
    assert summary["not_an_effect_claim"] == (1, 1)
