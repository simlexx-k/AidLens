from datetime import UTC, datetime

from sqlalchemy import func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.evaluation import Evaluation, EvaluationChunk
from app.schemas.analytics import (
    CorpusAudit,
    CorpusAuditEvaluation,
    CorpusStats,
    DuplicateTitleGroup,
    LabelCount,
    QualityFlag,
)
from app.services.ingestion.chunker import CHUNKER_VERSION


async def corpus_stats(session: AsyncSession) -> CorpusStats:
    evaluation_count = await _scalar(session, select(func.count(Evaluation.id)))
    chunk_count = await _scalar(session, select(func.count(EvaluationChunk.id)))
    embedded_chunk_count = await _scalar(
        session,
        select(func.count(EvaluationChunk.id)).where(
            EvaluationChunk.embedding.is_not(None)
        ),
    )

    year_min, year_max = (
        await session.execute(
            select(
                func.min(Evaluation.publication_year),
                func.max(Evaluation.publication_year),
            )
        )
    ).one()

    section_rows = (await session.execute(_section_counts_statement())).all()
    chunker_rows = (await session.execute(_chunker_counts_statement())).all()

    missing_year = await _scalar(
        session,
        select(func.count(Evaluation.id)).where(Evaluation.publication_year.is_(None)),
    )
    future_year = await _scalar(
        session,
        select(func.count(Evaluation.id)).where(
            Evaluation.publication_year > datetime.now(UTC).year
        ),
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
    stale_chunker = await _scalar(
        session,
        select(func.count(EvaluationChunk.id)).where(
            or_(
                EvaluationChunk.chunker_version.is_(None),
                EvaluationChunk.chunker_version != CHUNKER_VERSION,
            )
        ),
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

    coverage = (
        round((embedded_chunk_count / chunk_count) * 100, 2)
        if chunk_count
        else 0.0
    )

    return CorpusStats(
        evaluation_count=evaluation_count,
        chunk_count=chunk_count,
        embedded_chunk_count=embedded_chunk_count,
        embedding_coverage_percent=coverage,
        embedding_model=(settings.embedding_model if embedded_chunk_count else None),
        publication_year_min=year_min,
        publication_year_max=year_max,
        section_counts=[
            LabelCount(label=str(label), count=int(count))
            for label, count in section_rows
        ],
        chunker_versions=[
            LabelCount(label=str(label), count=int(count))
            for label, count in chunker_rows
        ],
        top_keywords=top_keywords,
        top_institutions=top_institutions,
        quality_flags=[
            QualityFlag(
                code="missing_publication_year",
                count=missing_year,
                description="Evaluations without a parsed publication year.",
            ),
            QualityFlag(
                code="future_publication_year",
                count=future_year,
                description="Evaluations with a publication year later than the current year.",
            ),
            QualityFlag(
                code="missing_abstract",
                count=missing_abstract,
                description="Evaluations without an abstract in source metadata.",
            ),
            QualityFlag(
                code="missing_text_source",
                count=missing_text_source,
                description="Evaluations without a plaintext source URL.",
            ),
            QualityFlag(
                code="unsectioned_chunks",
                count=unsectioned,
                description=(
                    "Evidence chunks not assigned to a recognized report section."
                ),
            ),
            QualityFlag(
                code="stale_chunker_chunks",
                count=stale_chunker,
                description=(
                    f"Chunks not produced by the current {CHUNKER_VERSION} chunker."
                ),
            ),
            QualityFlag(
                code="duplicate_title_groups",
                count=duplicate_title_groups,
                description="Distinct title groups that occur more than once.",
            ),
        ],
    )


async def corpus_audit(session: AsyncSession) -> CorpusAudit:
    """Return record-level details for corpus anomalies that need human review."""

    current_year = datetime.now(UTC).year
    future_rows = (
        await session.execute(
            select(
                Evaluation.external_id,
                Evaluation.title,
                Evaluation.publication_year,
                Evaluation.source_url,
            )
            .where(Evaluation.publication_year > current_year)
            .order_by(Evaluation.publication_year.desc(), Evaluation.external_id)
        )
    ).all()
    missing_text_rows = (
        await session.execute(
            select(
                Evaluation.external_id,
                Evaluation.title,
                Evaluation.publication_year,
                Evaluation.source_url,
            )
            .where(Evaluation.text_url.is_(None))
            .order_by(Evaluation.external_id)
        )
    ).all()
    duplicate_rows = (
        await session.execute(
            select(
                Evaluation.title,
                func.array_agg(Evaluation.external_id).label("evaluation_ids"),
            )
            .group_by(Evaluation.title)
            .having(func.count(Evaluation.id) > 1)
            .order_by(Evaluation.title)
        )
    ).all()

    return CorpusAudit(
        future_publication_years=[
            CorpusAuditEvaluation(
                external_id=str(external_id),
                title=str(title),
                publication_year=publication_year,
                source_url=str(source_url),
            )
            for external_id, title, publication_year, source_url in future_rows
        ],
        missing_text_sources=[
            CorpusAuditEvaluation(
                external_id=str(external_id),
                title=str(title),
                publication_year=publication_year,
                source_url=str(source_url),
            )
            for external_id, title, publication_year, source_url in missing_text_rows
        ],
        duplicate_titles=[
            DuplicateTitleGroup(
                title=str(title),
                evaluation_ids=sorted(str(item) for item in evaluation_ids),
            )
            for title, evaluation_ids in duplicate_rows
        ],
    )


def _section_counts_statement():
    """Build a PostgreSQL-safe section aggregation query."""
    return (
        select(
            func.coalesce(EvaluationChunk.section, "unsectioned").label("label"),
            func.count(EvaluationChunk.id).label("count"),
        )
        .group_by(EvaluationChunk.section)
        .order_by(func.count(EvaluationChunk.id).desc())
    )


def _chunker_counts_statement():
    return (
        select(
            func.coalesce(EvaluationChunk.chunker_version, "legacy").label("label"),
            func.count(EvaluationChunk.id).label("count"),
        )
        .group_by(EvaluationChunk.chunker_version)
        .order_by(func.count(EvaluationChunk.id).desc())
    )


async def _scalar(session: AsyncSession, statement) -> int:
    return int((await session.execute(statement)).scalar_one())


async def _jsonb_top_values(
    session: AsyncSession,
    column_name: str,
    limit: int = 12,
) -> list[LabelCount]:
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
