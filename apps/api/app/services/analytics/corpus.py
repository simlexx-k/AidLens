from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.evaluation import Evaluation, EvaluationChunk
from app.schemas.analytics import CorpusStats, LabelCount, QualityFlag


async def corpus_stats(session: AsyncSession) -> CorpusStats:
    evaluation_count = await _scalar(session, select(func.count(Evaluation.id)))
    chunk_count = await _scalar(session, select(func.count(EvaluationChunk.id)))
    embedded_chunk_count = await _scalar(
        session,
        select(func.count(EvaluationChunk.id)).where(EvaluationChunk.embedding.is_not(None)),
    )
    year_min, year_max = (
        await session.execute(
            select(func.min(Evaluation.publication_year), func.max(Evaluation.publication_year))
        )
    ).one()
    section_rows = (
        await session.execute(
            select(
                func.coalesce(EvaluationChunk.section, "unsectioned").label("label"),
                func.count(EvaluationChunk.id).label("count"),
            )
            .group_by(func.coalesce(EvaluationChunk.section, "unsectioned"))
            .order_by(func.count(EvaluationChunk.id).desc())
        )
    ).all()
    missing_year = await _scalar(
        session,
        select(func.count(Evaluation.id)).where(Evaluation.publication_year.is_(None)),
    )
    missing_abstract = await _scalar(
        session,
        select(func.count(Evaluation.id)).where(Evaluation.abstract.is_(None)),
    )
    missing_text_source = await _scalar(
        session,
        select(func.count(Evaluation.id)).where(Evaluation.text_url.is_(None)),
    )
    unsectioned = await _scalar(
        session,
        select(func.count(EvaluationChunk.id)).where(EvaluationChunk.section.is_(None)),
    )
    duplicate_title_groups = await _scalar(
        session,
        select(func.count()).select_from(
            select(Evaluation.title)
            .group_by(Evaluation.title)
            .having(func.count(Evaluation.id) > 1)
            .subquery()
        ),
    )
    top_keywords = await _jsonb_top_values(session, "keywords")
    top_institutions = await _jsonb_top_values(session, "institutions")
    settings = get_settings()
    coverage = round((embedded_chunk_count / chunk_count) * 100, 2) if chunk_count else 0.0
    return CorpusStats(
        evaluation_count=evaluation_count,
        chunk_count=chunk_count,
        embedded_chunk_count=embedded_chunk_count,
        embedding_coverage_percent=coverage,
        embedding_model=settings.embedding_model if embedded_chunk_count else None,
        publication_year_min=year_min,
        publication_year_max=year_max,
        section_counts=[LabelCount(label=str(label), count=int(count)) for label, count in section_rows],
        top_keywords=top_keywords,
        top_institutions=top_institutions,
        quality_flags=[
            QualityFlag(code="missing_publication_year", count=missing_year, description="Evaluations without a parsed publication year."),
            QualityFlag(code="missing_abstract", count=missing_abstract, description="Evaluations without an abstract in source metadata."),
            QualityFlag(code="missing_text_source", count=missing_text_source, description="Evaluations without a plaintext source URL."),
            QualityFlag(code="unsectioned_chunks", count=unsectioned, description="Evidence chunks not assigned to a recognized report section."),
            QualityFlag(code="duplicate_title_groups", count=duplicate_title_groups, description="Distinct title groups that occur more than once."),
        ],
    )


async def _scalar(session: AsyncSession, statement) -> int:
    return int((await session.execute(statement)).scalar_one())


async def _jsonb_top_values(session: AsyncSession, column_name: str, limit: int = 12) -> list[LabelCount]:
    if column_name not in {"keywords", "institutions"}:
        raise ValueError("Unsupported JSONB facet")
    query = text(
        f"""
        SELECT value AS label, COUNT(*)::int AS count
        FROM evaluations
        CROSS JOIN LATERAL jsonb_array_elements_text({column_name}) AS facet(value)
        WHERE value <> ''
        GROUP BY value
        ORDER BY count DESC, label
        LIMIT :limit
        """
    )
    rows = (await session.execute(query, {"limit": limit})).all()
    return [LabelCount(label=str(label), count=int(count)) for label, count in rows]
