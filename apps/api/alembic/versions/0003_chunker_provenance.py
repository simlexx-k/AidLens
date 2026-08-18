"""Track chunker provenance on evaluation chunks.

Revision ID: 0003_chunker_provenance
Revises: 0002_semantic_retrieval
Create Date: 2026-08-19
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_chunker_provenance"
down_revision: str | None = "0002_semantic_retrieval"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "evaluation_chunks",
        sa.Column("chunker_version", sa.String(length=32), nullable=True),
    )
    op.create_index(
        "ix_evaluation_chunks_chunker_version",
        "evaluation_chunks",
        ["chunker_version"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_evaluation_chunks_chunker_version",
        table_name="evaluation_chunks",
    )
    op.drop_column("evaluation_chunks", "chunker_version")
