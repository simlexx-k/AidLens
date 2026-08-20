from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.evaluation import Evaluation, EvaluationChunk
from app.schemas.evaluation import EvidenceSearchHit, EvidenceSearchRequest
from app.services.search.intelligence import evidence_role_for_section
from app.services.search.lexical import _apply_filters, _snippet


async def semantic_search(
    session: AsyncSession,
    payload: EvidenceSearchRequest,
    query_vector: list[float],
) -> list[EvidenceSearchHit]:
    distance = EvaluationChunk.embedding.cosine_distance(query_vector).label("distance")
    statement = (
        select(EvaluationChunk, Evaluation, distance)
        .join(Evaluation, Evaluation.id == EvaluationChunk.evaluation_id)
        .where(EvaluationChunk.embedding.is_not(None))
    )
    statement = _apply_filters(statement, payload)
    statement = statement.order_by(distance).limit(payload.top_k)
    rows = (await session.execute(statement)).all()
    hits: list[EvidenceSearchHit] = []
    for chunk, evaluation, raw_distance in rows:
        semantic_score = max(0.0, 1.0 - float(raw_distance))
        hits.append(
            EvidenceSearchHit(
                chunk_id=chunk.id,
                evaluation_id=evaluation.external_id,
                title=evaluation.title,
                project_title=evaluation.project_title,
                publication_year=evaluation.publication_year,
                section=chunk.section,
                evidence_role=evidence_role_for_section(chunk.section),
                text=_snippet(chunk.text),
                score=semantic_score,
                semantic_score=semantic_score,
                retrieval_sources=["semantic"],
                locations=evaluation.locations,
                institutions=evaluation.institutions,
                keywords=evaluation.keywords,
                source_url=evaluation.source_url,
            )
        )
    return hits
