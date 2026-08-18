"""Initial AidLens schema.

Revision ID: 0001_initial
Revises:
Create Date: 2026-08-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "evaluations",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("external_id", sa.String(length=32), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("publication_year", sa.Integer(), nullable=True),
        sa.Column("language", sa.String(length=64), nullable=True),
        sa.Column("project_title", sa.Text(), nullable=True),
        sa.Column("abstract", sa.Text(), nullable=True),
        sa.Column("authors", postgresql.JSONB(), nullable=False),
        sa.Column("institutions", postgresql.JSONB(), nullable=False),
        sa.Column("keywords", postgresql.JSONB(), nullable=False),
        sa.Column("locations", postgresql.JSONB(), nullable=False),
        sa.Column("contract_codes", postgresql.JSONB(), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("pdf_url", sa.Text(), nullable=True),
        sa.Column("text_url", sa.Text(), nullable=True),
        sa.Column("file_size_kb", sa.Integer(), nullable=True),
        sa.Column("raw_metadata", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("external_id"),
    )
    op.create_index("ix_evaluations_external_id", "evaluations", ["external_id"])
    op.create_index(
        "ix_evaluations_publication_year",
        "evaluations",
        ["publication_year"],
    )
    op.create_index("ix_evaluations_language", "evaluations", ["language"])

    op.create_table(
        "evaluation_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("evaluation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("section", sa.String(length=128), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(768), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["evaluation_id"],
            ["evaluations.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "evaluation_id",
            "ordinal",
            name="uq_chunk_evaluation_ordinal",
        ),
    )
    op.create_index(
        "ix_evaluation_chunks_evaluation_id",
        "evaluation_chunks",
        ["evaluation_id"],
    )
    op.create_index("ix_evaluation_chunks_section", "evaluation_chunks", ["section"])
    op.execute(
        "CREATE INDEX ix_evaluation_chunks_fts ON evaluation_chunks "
        "USING GIN (to_tsvector('english', text))"
    )


def downgrade() -> None:
    op.drop_index("ix_evaluation_chunks_fts", table_name="evaluation_chunks")
    op.drop_table("evaluation_chunks")
    op.drop_table("evaluations")
