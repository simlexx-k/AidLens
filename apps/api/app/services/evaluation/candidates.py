import asyncio
import json
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.benchmark import (
    CandidateQuery,
    RankingCandidate,
    RankingCandidateSet,
)
from app.schemas.evaluation import EvidenceSearchHit, EvidenceSearchRequest
from app.services.search.engine import execute_search


class QueryEncoderProtocol:
    model_name: str

    def encode_query(self, query: str) -> list[float]: ...


def load_candidate_queries(path: Path) -> list[CandidateQuery]:
    queries: list[CandidateQuery] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                queries.append(CandidateQuery.model_validate_json(line))
            except Exception as exc:
                raise ValueError(
                    f"Invalid candidate query JSONL at {path}:{line_number}: {exc}"
                ) from exc
    if not queries:
        raise ValueError(f"Candidate query file {path} contains no queries.")
    return queries


async def generate_candidate_sets(
    session: AsyncSession,
    queries: list[CandidateQuery],
    *,
    mode: str,
    top_k: int,
    encoder: QueryEncoderProtocol | None = None,
    max_per_evaluation: int = 3,
) -> list[RankingCandidateSet]:
    if mode not in {"lexical", "semantic", "hybrid"}:
        raise ValueError(f"Unsupported candidate retrieval mode: {mode}")
    if mode in {"semantic", "hybrid"} and encoder is None:
        raise ValueError("Semantic candidate generation requires an embedding encoder.")
    if max_per_evaluation < 1:
        raise ValueError("max_per_evaluation must be at least 1.")

    outputs: list[RankingCandidateSet] = []
    pool_k = min(max(top_k * 3, top_k), 50)
    for query in queries:
        query_vector = None
        if encoder is not None and mode in {"semantic", "hybrid"}:
            query_vector = await asyncio.to_thread(encoder.encode_query, query.query)
        response = await execute_search(
            session,
            EvidenceSearchRequest(query=query.query, mode=mode, top_k=pool_k),
            query_vector=query_vector,
            embedding_model=(encoder.model_name if encoder else None),
        )
        diversified = diversify_candidates(
            response.hits,
            top_k=top_k,
            max_per_evaluation=max_per_evaluation,
        )
        outputs.append(
            RankingCandidateSet(
                query_id=query.query_id,
                query=query.query,
                mode=response.mode,
                candidates=[
                    RankingCandidate(
                        rank=annotation_rank,
                        retrieval_rank=retrieval_rank,
                        chunk_id=hit.chunk_id,
                        evaluation_id=hit.evaluation_id,
                        title=hit.title,
                        section=hit.section,
                        text=hit.text,
                        score=hit.score,
                        lexical_score=hit.lexical_score,
                        semantic_score=hit.semantic_score,
                    )
                    for annotation_rank, (retrieval_rank, hit) in enumerate(
                        diversified,
                        start=1,
                    )
                ],
            )
        )
    return outputs


def diversify_candidates(
    hits: list[EvidenceSearchHit],
    *,
    top_k: int,
    max_per_evaluation: int,
) -> list[tuple[int, EvidenceSearchHit]]:
    """Build a broader annotation pool without changing production ranking."""

    selected: list[tuple[int, EvidenceSearchHit]] = []
    counts: dict[str, int] = {}
    for retrieval_rank, hit in enumerate(hits, start=1):
        count = counts.get(hit.evaluation_id, 0)
        if count >= max_per_evaluation:
            continue
        selected.append((retrieval_rank, hit))
        counts[hit.evaluation_id] = count + 1
        if len(selected) >= top_k:
            break
    return selected


def write_candidate_sets(items: list[RankingCandidateSet], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item.model_dump(mode="json")) + "\n")
