from app.schemas.evaluation import (
    EvidenceEvaluationGroup,
    EvidenceRole,
    EvidenceSearchHit,
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
