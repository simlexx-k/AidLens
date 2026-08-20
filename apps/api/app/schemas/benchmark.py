import uuid

from pydantic import BaseModel, Field


class RelevanceJudgment(BaseModel):
    evaluation_id: str = Field(min_length=1)
    section: str | None = Field(default=None, max_length=128)
    anchor_text: str | None = Field(default=None, min_length=8, max_length=300)
    relevance: int = Field(default=1, ge=1, le=3)


class BenchmarkQuery(BaseModel):
    query_id: str = Field(min_length=1)
    query: str = Field(min_length=2, max_length=500)
    family: str = Field(default="general", min_length=2, max_length=64)
    judgments: list[RelevanceJudgment] = Field(min_length=1)
    notes: str | None = None


class CandidateQuery(BaseModel):
    query_id: str = Field(min_length=1)
    query: str = Field(min_length=2, max_length=500)
    family: str = Field(default="general", min_length=2, max_length=64)


class RankingCandidate(BaseModel):
    rank: int
    retrieval_rank: int
    chunk_id: uuid.UUID
    evaluation_id: str
    title: str
    section: str | None = None
    text: str
    score: float
    lexical_score: float | None = None
    semantic_score: float | None = None
    relevance: int | None = Field(default=None, ge=0, le=3)


class RankingCandidateSet(BaseModel):
    query_id: str
    query: str
    family: str = "general"
    mode: str
    candidates: list[RankingCandidate]


class RankerTrainingRecord(BaseModel):
    query_id: str
    family: str
    query: str
    chunk_id: uuid.UUID
    evaluation_id: str
    title: str
    section: str | None = None
    text: str
    relevance: int = Field(ge=0, le=3)
    retrieval_rank: int
    score: float
    lexical_score: float | None = None
    semantic_score: float | None = None


class RetrievalMetrics(BaseModel):
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    relevant_count: int
    retrieved_relevant_count: int
    unique_evaluations_at_k: int
    duplicate_share_at_k: float


class BenchmarkQueryResult(BaseModel):
    query_id: str
    query: str
    family: str = "general"
    mode: str
    metrics: RetrievalMetrics
    top_evaluation_ids: list[str]
    top_sections: list[str | None]


class BenchmarkModeSummary(BaseModel):
    mode: str
    query_count: int
    mean_recall_at_k: float
    mean_reciprocal_rank: float
    mean_ndcg_at_k: float
    mean_unique_evaluations_at_k: float
    mean_duplicate_share_at_k: float


class BenchmarkFamilySummary(BenchmarkModeSummary):
    family: str


class BenchmarkReport(BaseModel):
    dataset: str
    top_k: int
    embedding_model: str | None = None
    max_per_evaluation: int | None = None
    modes: list[BenchmarkModeSummary]
    families: list[BenchmarkFamilySummary] = Field(default_factory=list)
    queries: list[BenchmarkQueryResult]
