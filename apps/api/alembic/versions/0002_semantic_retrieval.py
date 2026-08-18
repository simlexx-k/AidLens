"""Add semantic retrieval metadata and vector index.

Revision ID: 0002_semantic_retrieval
Revises: 0001_initial
Create Date: 2026-08-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_semantic_retrieval"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "evaluation_chunks",
        sa.Column("embedding_model", sa.String(length=255), nullable=True),
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_evaluation_chunks_embedding_hnsw
        ON evaluation_chunks
        USING hnsw (embedding vector_cosine_ops)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_evaluation_chunks_embedding_hnsw")
    op.drop_column("evaluation_chunks", "embedding_model")
