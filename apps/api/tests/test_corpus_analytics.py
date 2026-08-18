from sqlalchemy.dialects import postgresql

from app.services.analytics.corpus import _section_counts_statement


def test_section_counts_group_by_underlying_column() -> None:
    compiled = str(
        _section_counts_statement().compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": False},
        )
    )

    assert "GROUP BY evaluation_chunks.section" in compiled
    assert "GROUP BY coalesce" not in compiled
