import uuid
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class Evaluation(TimestampMixin, Base):
    __tablename__ = "evaluations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    external_id: Mapped[str] = mapped_column(String(32), unique=True, index=True, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    publication_year: Mapped[int | None] = mapped_column(Integer, index=True)
    language: Mapped[str | None] = mapped_column(String(64), index=True)
    project_title: Mapped[str | None] = mapped_column(Text)
    abstract: Mapped[str | None] = mapped_column(Text)
    authors: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    institutions: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    keywords: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    locations: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    contract_codes: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    pdf_url: Mapped[str | None] = mapped_column(Text)
    text_url: Mapped[str | None] = mapped_column(Text)
    file_size_kb: Mapped[int | None] = mapped_column(Integer)
    raw_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)

    chunks: Mapped[list["EvaluationChunk"]] = relationship(
        back_populates="evaluation",
        cascade="all, delete-orphan",
        order_by="EvaluationChunk.ordinal",
    )


class EvaluationChunk(TimestampMixin, Base):
    __tablename__ = "evaluation_chunks"
    __table_args__ = (
        UniqueConstraint("evaluation_id", "ordinal", name="uq_chunk_evaluation_ordinal"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    evaluation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("evaluations.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    section: Mapped[str | None] = mapped_column(String(128), index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(768), nullable=True)

    evaluation: Mapped[Evaluation] = relationship(back_populates="chunks")
