import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SearchMode = Literal["auto", "lexical", "semantic", "hybrid"]
RerankMode = Literal["auto", "disabled", "aidranker"]
EvidenceRole = Literal[
    "outcome",
    "recommendation",
    "method",
    "context",
    "implementation",
    "sustainability",
    "supporting",
]


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
    rerank: RerankMode = "auto"
    publication_year_from: int | None = Field(default=None, ge=1900, le=2100)
    publication_year_to: int | None = Field(default=None, ge=1900, le=2100)
    section: str | None = Field(default=None, max_length=128)
    max_per_evaluation: int | None = Field(default=None, ge=1, le=20)


class EvidenceSearchHit(BaseModel):
    chunk_id: uuid.UUID
    evaluation_id: str
    title: str
    project_title: str | None = None
    publication_year: int | None = None
    section: str | None = None
    evidence_role: EvidenceRole = "supporting"
    text: str
    score: float
    lexical_score: float | None = None
    semantic_score: float | None = None
    reranker_score: float | None = None
    fusion_score: float | None = None
    retrieval_sources: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    institutions: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    source_url: str


class EvidenceEvaluationGroup(BaseModel):
    evaluation_id: str
    title: str
    project_title: str | None = None
    intervention: str
    publication_year: int | None = None
    locations: list[str] = Field(default_factory=list)
    institutions: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    evidence_roles: list[EvidenceRole] = Field(default_factory=list)
    outcome_evidence_count: int = 0
    top_score: float
    source_url: str
    hits: list[EvidenceSearchHit]


class EvidenceSearchResponse(BaseModel):
    query: str
    mode: str
    embedding_model: str | None = None
    max_per_evaluation: int | None = None
    reranker_applied: bool = False
    reranker: str | None = None
    reranker_model: str | None = None
    reranker_model_fingerprint: str | None = None
    reranker_alpha: float | None = None
    reranker_fallback_reason: str | None = None
    ranking_pipeline: list[str] = Field(default_factory=list)
    first_stage_latency_ms: float | None = None
    reranker_latency_ms: float | None = None
    total_search_latency_ms: float | None = None
    groups: list[EvidenceEvaluationGroup] = Field(default_factory=list)
    hits: list[EvidenceSearchHit]
