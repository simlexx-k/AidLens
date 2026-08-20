import asyncio
import json
import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.benchmark import (
    CandidateQuery,
    RankingCandidate,
    RankingCandidateSet,
)
from app.schemas.evaluation import EvidenceSearchHit, EvidenceSearchRequest
from app.services.search.engine import execute_search

SUPPORTED_RETRIEVAL_MODES = {"lexical", "semantic", "hybrid"}


class QueryEncoderProtocol:
    model_name: str

    def encode_query(self, query: str) -> list[float]: ...


@dataclass(slots=True)
class PooledCandidate:
    hit: EvidenceSearchHit
    retrieval_rank: int
    retrieval_modes: list[str]
    mode_ranks: dict[str, int]


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
    if mode not in SUPPORTED_RETRIEVAL_MODES:
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
                family=query.family,
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
                        retrieval_modes=[response.mode],
                        mode_ranks={response.mode: retrieval_rank},
                    )
                    for annotation_rank, (retrieval_rank, hit) in enumerate(
                        diversified,
                        start=1,
                    )
                ],
            )
        )
    return outputs


async def generate_pooled_candidate_sets(
    session: AsyncSession,
    queries: list[CandidateQuery],
    *,
    modes: list[str],
    per_mode_k: int,
    encoder: QueryEncoderProtocol | None = None,
    max_per_evaluation: int = 5,
) -> list[RankingCandidateSet]:
    """Pool independent retriever outputs for system-neutral human judging."""

    selected_modes = list(dict.fromkeys(mode.strip().lower() for mode in modes))
    if not selected_modes:
        raise ValueError("Provide at least one candidate retrieval mode.")
    unsupported = set(selected_modes) - SUPPORTED_RETRIEVAL_MODES
    if unsupported:
        raise ValueError(
            f"Unsupported candidate retrieval mode(s): {', '.join(sorted(unsupported))}"
        )
    if any(mode in {"semantic", "hybrid"} for mode in selected_modes) and encoder is None:
        raise ValueError("Semantic pooled candidate generation requires an embedding encoder.")
    if per_mode_k < 1:
        raise ValueError("per_mode_k must be at least 1.")
    if max_per_evaluation < 1:
        raise ValueError("max_per_evaluation must be at least 1.")

    outputs: list[RankingCandidateSet] = []
    for query in queries:
        query_vector = None
        if encoder is not None and any(
            mode in {"semantic", "hybrid"} for mode in selected_modes
        ):
            query_vector = await asyncio.to_thread(encoder.encode_query, query.query)

        mode_hits: dict[str, list[EvidenceSearchHit]] = {}
        for mode in selected_modes:
            response = await execute_search(
                session,
                EvidenceSearchRequest(query=query.query, mode=mode, top_k=per_mode_k),
                query_vector=(query_vector if mode in {"semantic", "hybrid"} else None),
                embedding_model=(encoder.model_name if encoder else None),
            )
            mode_hits[mode] = response.hits

        pooled = pool_candidates(
            mode_hits,
            mode_order=selected_modes,
            max_per_evaluation=max_per_evaluation,
        )
        outputs.append(
            RankingCandidateSet(
                query_id=query.query_id,
                query=query.query,
                family=query.family,
                mode="pooled",
                candidates=[
                    RankingCandidate(
                        rank=annotation_rank,
                        retrieval_rank=item.retrieval_rank,
                        chunk_id=item.hit.chunk_id,
                        evaluation_id=item.hit.evaluation_id,
                        title=item.hit.title,
                        section=item.hit.section,
                        text=item.hit.text,
                        score=item.hit.score,
                        lexical_score=item.hit.lexical_score,
                        semantic_score=item.hit.semantic_score,
                        retrieval_modes=item.retrieval_modes,
                        mode_ranks=item.mode_ranks,
                    )
                    for annotation_rank, item in enumerate(pooled, start=1)
                ],
            )
        )
    return outputs


def pool_candidates(
    mode_hits: dict[str, list[EvidenceSearchHit]],
    *,
    mode_order: list[str],
    max_per_evaluation: int,
) -> list[PooledCandidate]:
    """Pool ranked lists without letting one retrieval system define ground truth."""

    if max_per_evaluation < 1:
        raise ValueError("max_per_evaluation must be at least 1.")

    entries: dict[str, dict[str, object]] = {}
    ranked_keys: dict[str, list[str]] = {}
    for mode in mode_order:
        ranked_keys[mode] = []
        for rank, hit in enumerate(mode_hits.get(mode, []), start=1):
            key = str(hit.chunk_id)
            ranked_keys[mode].append(key)
            entry = entries.get(key)
            if entry is None:
                entries[key] = {
                    "hit": hit,
                    "mode_ranks": {mode: rank},
                }
                continue

            mode_ranks = entry["mode_ranks"]
            assert isinstance(mode_ranks, dict)
            mode_ranks[mode] = rank

            current_hit = entry["hit"]
            assert isinstance(current_hit, EvidenceSearchHit)
            entry["hit"] = _merge_hit_scores(current_hit, hit)

    selected_keys: list[str] = []
    selected: set[str] = set()
    counts: dict[str, int] = {}
    max_depth = max((len(items) for items in ranked_keys.values()), default=0)

    # Rotate the starting mode at each rank depth so ties do not always favor
    # the first configured retriever. Every unique top-k candidate remains
    # eligible; only the explicit per-evaluation annotation cap can exclude it.
    for depth in range(max_depth):
        offset = depth % len(mode_order) if mode_order else 0
        traversal = mode_order[offset:] + mode_order[:offset]
        for mode in traversal:
            keys = ranked_keys.get(mode, [])
            if depth >= len(keys):
                continue
            key = keys[depth]
            if key in selected:
                continue
            entry = entries[key]
            hit = entry["hit"]
            assert isinstance(hit, EvidenceSearchHit)
            count = counts.get(hit.evaluation_id, 0)
            if count >= max_per_evaluation:
                continue
            selected.add(key)
            selected_keys.append(key)
            counts[hit.evaluation_id] = count + 1

    pooled: list[PooledCandidate] = []
    for key in selected_keys:
        entry = entries[key]
        hit = entry["hit"]
        mode_ranks = entry["mode_ranks"]
        assert isinstance(hit, EvidenceSearchHit)
        assert isinstance(mode_ranks, dict)
        typed_ranks = {str(mode): int(rank) for mode, rank in mode_ranks.items()}
        retrieval_modes = [mode for mode in mode_order if mode in typed_ranks]
        pooled.append(
            PooledCandidate(
                hit=hit,
                retrieval_rank=min(typed_ranks.values()),
                retrieval_modes=retrieval_modes,
                mode_ranks=typed_ranks,
            )
        )
    return pooled


def _merge_hit_scores(
    current: EvidenceSearchHit,
    incoming: EvidenceSearchHit,
) -> EvidenceSearchHit:
    """Preserve available component scores when one chunk appears in many modes."""

    return current.model_copy(
        update={
            "lexical_score": (
                current.lexical_score
                if current.lexical_score is not None
                else incoming.lexical_score
            ),
            "semantic_score": (
                current.semantic_score
                if current.semantic_score is not None
                else incoming.semantic_score
            ),
            "retrieval_sources": list(
                dict.fromkeys(current.retrieval_sources + incoming.retrieval_sources)
            ),
        }
    )


def carry_forward_labels(
    candidate_sets: list[RankingCandidateSet],
    previous_sets: list[RankingCandidateSet],
) -> int:
    """Copy reviewed labels onto matching query/chunk pairs in a new pool."""

    labels: dict[tuple[str, uuid.UUID], int] = {}
    for item in previous_sets:
        for candidate in item.candidates:
            if candidate.relevance is None:
                continue
            labels[(item.query_id, candidate.chunk_id)] = int(candidate.relevance)

    copied = 0
    for item in candidate_sets:
        for candidate in item.candidates:
            relevance = labels.get((item.query_id, candidate.chunk_id))
            if relevance is None:
                continue
            candidate.relevance = relevance
            copied += 1
    return copied


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


def load_candidate_sets(path: Path) -> list[RankingCandidateSet]:
    items: list[RankingCandidateSet] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                items.append(RankingCandidateSet.model_validate_json(line))
            except Exception as exc:
                raise ValueError(
                    f"Invalid candidate-set JSONL at {path}:{line_number}: {exc}"
                ) from exc
    if not items:
        raise ValueError(f"Candidate-set file {path} contains no queries.")
    return items


def write_candidate_sets(items: list[RankingCandidateSet], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item.model_dump(mode="json")) + "\n")
