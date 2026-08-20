from collections import defaultdict
from itertools import combinations

from app.schemas.evaluation import (
    CrossEvaluationFacet,
    CrossEvaluationSynthesis,
    EvidenceEvaluationGroup,
    EvidenceFacetKind,
    EvidenceRole,
    EvidenceRoleCoverage,
    EvidenceSearchHit,
    TransferabilityPair,
)
from app.services.search.claims import (
    CLAIM_EXTRACTOR,
    effect_claim_evaluation_count,
    extract_grounded_claims,
    stance_coverage,
)

_ROLE_BY_SECTION: dict[str, EvidenceRole] = {
    "findings": "outcome",
    "results": "outcome",
    "conclusions": "outcome",
    "recommendations": "recommendation",
    "methodology": "method",
    "evaluation_questions": "method",
    "abstract": "context",
    "introduction": "context",
    "background": "context",
    "lessons_learned": "implementation",
    "sustainability": "sustainability",
}

_ROLE_ORDER: tuple[EvidenceRole, ...] = (
    "outcome",
    "sustainability",
    "implementation",
    "recommendation",
    "context",
    "method",
    "supporting",
)
_FACET_ORDER: dict[EvidenceFacetKind, int] = {
    "location": 0,
    "institution": 1,
    "keyword": 2,
}
_SYNTHESIS_CAVEATS = [
    (
        "This synthesis covers only the final ranked passages returned for this query, "
        "not the full corpus."
    ),
    (
        "Recurring facets are structured metadata co-occurrence signals, not causal "
        "'what works' claims."
    ),
    (
        "Context overlap is a metadata similarity signal, not evidence that an "
        "intervention will transfer."
    ),
    (
        "V1.3 stance labels are conservative explicit-text signals, not adjudicated "
        "causal conclusions; inspect each cited source sentence before use."
    ),
    (
        "The insufficient and not_an_effect_claim states are intentional abstentions "
        "when the returned text does not support a stronger stance."
    ),
]


def evidence_role_for_section(section: str | None) -> EvidenceRole:
    if not section:
        return "supporting"
    return _ROLE_BY_SECTION.get(section.lower(), "supporting")


def group_evidence_hits(hits: list[EvidenceSearchHit]) -> list[EvidenceEvaluationGroup]:
    """Group final ranked passages into evaluation-level evidence units.

    Group order follows first appearance in the ranked hit list. The grouping layer
    deliberately preserves source metadata rather than generating unsupported
    intervention or outcome summaries.
    """

    by_evaluation: dict[str, list[EvidenceSearchHit]] = {}
    order: list[str] = []
    for hit in hits:
        if hit.evaluation_id not in by_evaluation:
            order.append(hit.evaluation_id)
            by_evaluation[hit.evaluation_id] = []
        by_evaluation[hit.evaluation_id].append(hit)

    groups: list[EvidenceEvaluationGroup] = []
    for evaluation_id in order:
        evaluation_hits = by_evaluation[evaluation_id]
        first = evaluation_hits[0]
        roles = list(dict.fromkeys(hit.evidence_role for hit in evaluation_hits))
        groups.append(
            EvidenceEvaluationGroup(
                evaluation_id=evaluation_id,
                title=first.title,
                project_title=first.project_title,
                intervention=first.project_title or first.title,
                publication_year=first.publication_year,
                locations=first.locations,
                institutions=first.institutions,
                keywords=first.keywords,
                evidence_roles=roles,
                outcome_evidence_count=sum(
                    hit.evidence_role in {"outcome", "sustainability"}
                    for hit in evaluation_hits
                ),
                top_score=max(hit.score for hit in evaluation_hits),
                source_url=first.source_url,
                hits=evaluation_hits,
            )
        )
    return groups


def synthesize_evidence_groups(
    groups: list[EvidenceEvaluationGroup],
) -> CrossEvaluationSynthesis:
    """Build deterministic cross-evaluation signals from final result groups.

    V1.3 keeps V1.2 metadata/context synthesis and adds source-span-grounded claim
    assessments. Stance is inferred only from explicit lexical evidence inside the
    returned sentence; section labels and metadata never determine stance.
    """

    claims = extract_grounded_claims(groups)
    return CrossEvaluationSynthesis(
        evaluation_count=len(groups),
        passage_count=sum(len(group.hits) for group in groups),
        outcome_evaluation_count=sum(group.outcome_evidence_count > 0 for group in groups),
        role_coverage=_role_coverage(groups),
        recurring_facets=_recurring_facets(groups),
        transferability_pairs=_transferability_pairs(groups),
        claim_extractor=CLAIM_EXTRACTOR,
        claims=claims,
        stance_coverage=stance_coverage(claims),
        effect_claim_evaluation_count=effect_claim_evaluation_count(claims),
        caveats=list(_SYNTHESIS_CAVEATS),
    )


def _role_coverage(groups: list[EvidenceEvaluationGroup]) -> list[EvidenceRoleCoverage]:
    coverage: list[EvidenceRoleCoverage] = []
    for role in _ROLE_ORDER:
        passage_count = sum(
            hit.evidence_role == role
            for group in groups
            for hit in group.hits
        )
        if passage_count == 0:
            continue
        coverage.append(
            EvidenceRoleCoverage(
                role=role,
                evaluation_count=sum(role in group.evidence_roles for group in groups),
                passage_count=passage_count,
            )
        )
    return coverage


def _recurring_facets(groups: list[EvidenceEvaluationGroup]) -> list[CrossEvaluationFacet]:
    evaluations_by_facet: dict[tuple[EvidenceFacetKind, str], set[str]] = defaultdict(set)
    display_value: dict[tuple[EvidenceFacetKind, str], str] = {}

    for group in groups:
        for kind, values in _group_facets(group).items():
            seen: set[str] = set()
            for value in values:
                normalized = _normalize(value)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                key = (kind, normalized)
                evaluations_by_facet[key].add(group.evaluation_id)
                display_value.setdefault(key, value.strip())

    facets = [
        CrossEvaluationFacet(
            kind=kind,
            value=display_value[(kind, normalized)],
            evaluation_count=len(evaluation_ids),
            evaluation_ids=sorted(evaluation_ids),
        )
        for (kind, normalized), evaluation_ids in evaluations_by_facet.items()
        if len(evaluation_ids) >= 2
    ]
    facets.sort(
        key=lambda facet: (
            -facet.evaluation_count,
            _FACET_ORDER[facet.kind],
            facet.value.casefold(),
        )
    )
    return facets[:20]


def _transferability_pairs(
    groups: list[EvidenceEvaluationGroup],
) -> list[TransferabilityPair]:
    pairs: list[TransferabilityPair] = []
    for left, right in combinations(groups, 2):
        shared_locations = _shared_values(left.locations, right.locations)
        shared_institutions = _shared_values(left.institutions, right.institutions)
        shared_keywords = _shared_values(left.keywords, right.keywords)
        shared_signal_count = (
            len(shared_locations) + len(shared_institutions) + len(shared_keywords)
        )
        if shared_signal_count == 0:
            continue

        left_tokens = _context_tokens(left)
        right_tokens = _context_tokens(right)
        union = left_tokens | right_tokens
        overlap_score = len(left_tokens & right_tokens) / len(union) if union else 0.0
        pairs.append(
            TransferabilityPair(
                left_evaluation_id=left.evaluation_id,
                right_evaluation_id=right.evaluation_id,
                context_overlap_score=round(overlap_score, 3),
                shared_signal_count=shared_signal_count,
                shared_locations=shared_locations,
                shared_institutions=shared_institutions,
                shared_keywords=shared_keywords,
            )
        )

    pairs.sort(
        key=lambda pair: (
            -pair.context_overlap_score,
            -pair.shared_signal_count,
            pair.left_evaluation_id,
            pair.right_evaluation_id,
        )
    )
    return pairs[:10]


def _group_facets(
    group: EvidenceEvaluationGroup,
) -> dict[EvidenceFacetKind, list[str]]:
    return {
        "location": group.locations,
        "institution": group.institutions,
        "keyword": group.keywords,
    }


def _context_tokens(group: EvidenceEvaluationGroup) -> set[str]:
    tokens: set[str] = set()
    for kind, values in _group_facets(group).items():
        tokens.update(
            f"{kind}:{normalized}"
            for value in values
            if (normalized := _normalize(value))
        )
    return tokens


def _shared_values(left: list[str], right: list[str]) -> list[str]:
    right_normalized = {_normalize(value) for value in right}
    shared: list[str] = []
    seen: set[str] = set()
    for value in left:
        normalized = _normalize(value)
        if normalized and normalized in right_normalized and normalized not in seen:
            shared.append(value.strip())
            seen.add(normalized)
    return shared


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())
