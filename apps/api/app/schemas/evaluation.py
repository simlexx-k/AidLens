import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SearchMode = Literal["auto", "lexical", "semantic", "hybrid"]


class EvaluationSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    external_id: str
    title: str
    publication_year: int | None = None
    language: str | None = None
    authors: list[str] = Field(default_factory=list)
    institutions: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    source_url: str


class EvaluationDetail(EvaluationSummary):
    project_title: str | None = None
    abstract: str | None = None
    contract_codes: list[str] = Field(default_factory=list)
    pdf_url: str | None = None
    text_url: str | None = None
    file_size_kb: int | None = None


class EvidenceSearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    top_k: int = Field(default=10, ge=1, le=50)
    mode: SearchMode = "auto"
    publication_year_from: int | None = Field(default=None, ge=1900, le=2100)
    publication_year_to: int | None = Field(default=None, ge=1900, le=2100)
    section: str | None = Field(default=None, max_length=128)
    max_per_evaluation: int | None = Field(default=None, ge=1, le=20)


class EvidenceSearchHit(BaseModel):
    chunk_id: uuid.UUID
    evaluation_id: str
    title: str
    publication_year: int | None = None
    section: str | None = None
    text: str
    score: float
    lexical_score: float | None = None
    semantic_score: float | None = None
    retrieval_sources: list[str] = Field(default_factory=list)
    source_url: str


class EvidenceSearchResponse(BaseModel):
    query: str
    mode: str
    embedding_model: str | None = None
    max_per_evaluation: int | None = None
    hits: list[EvidenceSearchHit]
