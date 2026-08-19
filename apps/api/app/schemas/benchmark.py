import uuid

from pydantic import BaseModel, Field


class RelevanceJudgment(BaseModel):
    evaluation_id: str = Field(min_length=1)
    section: str | None = Field(default=None, max_length=128)
    relevance: int = Field(default=1, ge=1, le=3)


class BenchmarkQuery(BaseModel):
    query_id: str = Field(min_length=1)
    query: str = Field(min_length=2, max_length=500)
    judgments: list[RelevanceJudgment] = Field(min_length=1)
    notes: str | None = None


class CandidateQuery(BaseModel):
    query_id: str = Field(min_length=1)
    query: str = Field(min_length=2, max_length=500)


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
    mode: str
    candidates: list[RankingCandidate]


class RetrievalMetrics(BaseModel):
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    relevant_count: int
    retrieved_relevant_count: int


class BenchmarkQueryResult(BaseModel):
    query_id: str
    query: str
    mode: str
    metrics: RetrievalMetrics
    top_evaluation_ids: list[str]


class BenchmarkModeSummary(BaseModel):
    mode: str
    query_count: int
    mean_recall_at_k: float
    mean_reciprocal_rank: float
    mean_ndcg_at_k: float


class BenchmarkReport(BaseModel):
    dataset: str
    top_k: int
    embedding_model: str | None = None
    modes: list[BenchmarkModeSummary]
    queries: list[BenchmarkQueryResult]
