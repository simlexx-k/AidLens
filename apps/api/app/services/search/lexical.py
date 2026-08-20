from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.evaluation import Evaluation, EvaluationChunk
from app.schemas.evaluation import EvidenceSearchHit, EvidenceSearchRequest
from app.services.search.intelligence import evidence_role_for_section


async def lexical_search(
    session: AsyncSession,
    payload: EvidenceSearchRequest,
) -> list[EvidenceSearchHit]:
    vector = func.to_tsvector("english", EvaluationChunk.text)
    query = func.plainto_tsquery("english", payload.query)
    rank = func.ts_rank_cd(vector, query).label("rank")
    statement = (
        select(EvaluationChunk, Evaluation, rank)
        .join(Evaluation, Evaluation.id == EvaluationChunk.evaluation_id)
        .where(vector.op("@@")(query))
    )
    statement = _apply_filters(statement, payload)
    statement = statement.order_by(desc(rank)).limit(payload.top_k)
    rows = (await session.execute(statement)).all()
    return [
        EvidenceSearchHit(
            chunk_id=chunk.id,
            evaluation_id=evaluation.external_id,
            title=evaluation.title,
            project_title=evaluation.project_title,
            publication_year=evaluation.publication_year,
            section=chunk.section,
            evidence_role=evidence_role_for_section(chunk.section),
            text=_snippet(chunk.text),
            score=float(score),
            lexical_score=float(score),
            retrieval_sources=["lexical"],
            locations=evaluation.locations,
            institutions=evaluation.institutions,
            keywords=evaluation.keywords,
            source_url=evaluation.source_url,
        )
        for chunk, evaluation, score in rows
    ]


def _apply_filters(statement, payload: EvidenceSearchRequest):
    if payload.publication_year_from is not None:
        statement = statement.where(Evaluation.publication_year >= payload.publication_year_from)
    if payload.publication_year_to is not None:
        statement = statement.where(Evaluation.publication_year <= payload.publication_year_to)
    if payload.section:
        statement = statement.where(EvaluationChunk.section == payload.section)
    return statement


def _snippet(text: str, limit: int = 700) -> str:
    clean = text.strip()
    return clean if len(clean) <= limit else clean[: limit - 1].rstrip() + "…"
