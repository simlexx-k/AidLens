from sqlalchemy.dialects import postgresql

from app.services.analytics.corpus import (
    _chunker_counts_statement,
    _section_counts_statement,
)


def _compile(statement) -> str:
    return str(
        statement.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": False},
        )
    )


def test_section_counts_group_by_underlying_column() -> None:
    compiled = _compile(_section_counts_statement())

    assert "GROUP BY evaluation_chunks.section" in compiled
    assert "GROUP BY coalesce" not in compiled


def test_chunker_counts_group_by_underlying_column() -> None:
    compiled = _compile(_chunker_counts_statement())

    assert "GROUP BY evaluation_chunks.chunker_version" in compiled
    assert "GROUP BY coalesce" not in compiled
