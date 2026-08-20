from pydantic import BaseModel, Field


class LabelCount(BaseModel):
    label: str
    count: int


class QualityFlag(BaseModel):
    code: str
    count: int
    description: str


class CorpusStats(BaseModel):
    evaluation_count: int
    chunk_count: int
    embedded_chunk_count: int
    embedding_coverage_percent: float
    embedding_model: str | None = None
    publication_year_min: int | None = None
    publication_year_max: int | None = None
    section_counts: list[LabelCount] = Field(default_factory=list)
    chunker_versions: list[LabelCount] = Field(default_factory=list)
    top_keywords: list[LabelCount] = Field(default_factory=list)
    top_institutions: list[LabelCount] = Field(default_factory=list)
    quality_flags: list[QualityFlag] = Field(default_factory=list)


class CorpusAuditEvaluation(BaseModel):
    external_id: str
    title: str
    publication_year: int | None = None
    source_url: str


class DuplicateTitleGroup(BaseModel):
    title: str
    evaluation_ids: list[str]


class CorpusAudit(BaseModel):
    future_publication_years: list[CorpusAuditEvaluation] = Field(default_factory=list)
    missing_text_sources: list[CorpusAuditEvaluation] = Field(default_factory=list)
    duplicate_titles: list[DuplicateTitleGroup] = Field(default_factory=list)
