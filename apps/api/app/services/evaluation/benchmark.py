import asyncio
import json
from collections.abc import Iterable
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.benchmark import (
    BenchmarkFamilySummary,
    BenchmarkModeSummary,
    BenchmarkQuery,
    BenchmarkQueryResult,
    BenchmarkReport,
)
from app.schemas.evaluation import EvidenceSearchRequest
from app.services.evaluation.metrics import evaluate_hits
from app.services.search.engine import execute_search


class QueryEncoderProtocol:
    model_name: str

    def encode_query(self, query: str) -> list[float]: ...


def load_benchmark_dataset(path: Path) -> list[BenchmarkQuery]:
    queries: list[BenchmarkQuery] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                queries.append(BenchmarkQuery.model_validate_json(line))
            except Exception as exc:
                raise ValueError(
                    f"Invalid benchmark JSONL at {path}:{line_number}: {exc}"
                ) from exc
    if not queries:
        raise ValueError(f"Benchmark dataset {path} contains no queries.")
    return queries


async def run_benchmark(
    session: AsyncSession,
    queries: list[BenchmarkQuery],
    *,
    modes: Iterable[str],
    top_k: int,
    encoder: QueryEncoderProtocol | None = None,
    dataset_name: str = "benchmark",
    max_per_evaluation: int | None = None,
) -> BenchmarkReport:
    normalized_modes = list(dict.fromkeys(modes))
    unsupported = set(normalized_modes) - {"lexical", "semantic", "hybrid"}
    if unsupported:
        raise ValueError(f"Unsupported retrieval mode(s): {', '.join(sorted(unsupported))}")
    if any(mode in {"semantic", "hybrid"} for mode in normalized_modes) and encoder is None:
        raise ValueError("Semantic benchmark modes require an embedding encoder.")

    query_results: list[BenchmarkQueryResult] = []
    by_mode: dict[str, list[BenchmarkQueryResult]] = {
        mode: [] for mode in normalized_modes
    }
    by_family_mode: dict[tuple[str, str], list[BenchmarkQueryResult]] = {}

    for query in queries:
        query_vector = None
        if encoder is not None and any(
            mode in {"semantic", "hybrid"} for mode in normalized_modes
        ):
            query_vector = await asyncio.to_thread(encoder.encode_query, query.query)

        for mode in normalized_modes:
            payload = EvidenceSearchRequest(
                query=query.query,
                top_k=top_k,
                mode=mode,
                max_per_evaluation=max_per_evaluation,
            )
            response = await execute_search(
                session,
                payload,
                query_vector=(query_vector if mode != "lexical" else None),
                embedding_model=(encoder.model_name if encoder and mode != "lexical" else None),
            )
            result = BenchmarkQueryResult(
                query_id=query.query_id,
                query=query.query,
                family=query.family,
                mode=mode,
                metrics=evaluate_hits(response.hits, query.judgments, k=top_k),
                top_evaluation_ids=[hit.evaluation_id for hit in response.hits],
                top_sections=[hit.section for hit in response.hits],
            )
            query_results.append(result)
            by_mode[mode].append(result)
            by_family_mode.setdefault((query.family, mode), []).append(result)

    summaries = [_summarize_mode(mode, items) for mode, items in by_mode.items()]
    family_summaries = [
        _summarize_family(family, mode, items)
        for (family, mode), items in sorted(by_family_mode.items())
    ]
    return BenchmarkReport(
        dataset=dataset_name,
        top_k=top_k,
        embedding_model=encoder.model_name if encoder else None,
        max_per_evaluation=max_per_evaluation,
        modes=summaries,
        families=family_summaries,
        queries=query_results,
    )


def write_report(report: BenchmarkReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.model_dump(), indent=2) + "\n",
        encoding="utf-8",
    )


def _summary_values(results: list[BenchmarkQueryResult]) -> dict[str, float | int]:
    count = len(results)
    if count == 0:
        return {
            "query_count": 0,
            "mean_recall_at_k": 0.0,
            "mean_reciprocal_rank": 0.0,
            "mean_ndcg_at_k": 0.0,
            "mean_unique_evaluations_at_k": 0.0,
            "mean_duplicate_share_at_k": 0.0,
        }
    return {
        "query_count": count,
        "mean_recall_at_k": round(
            sum(item.metrics.recall_at_k for item in results) / count,
            6,
        ),
        "mean_reciprocal_rank": round(
            sum(item.metrics.reciprocal_rank for item in results) / count,
            6,
        ),
        "mean_ndcg_at_k": round(
            sum(item.metrics.ndcg_at_k for item in results) / count,
            6,
        ),
        "mean_unique_evaluations_at_k": round(
            sum(item.metrics.unique_evaluations_at_k for item in results) / count,
            6,
        ),
        "mean_duplicate_share_at_k": round(
            sum(item.metrics.duplicate_share_at_k for item in results) / count,
            6,
        ),
    }


def _summarize_mode(
    mode: str,
    results: list[BenchmarkQueryResult],
) -> BenchmarkModeSummary:
    return BenchmarkModeSummary(mode=mode, **_summary_values(results))


def _summarize_family(
    family: str,
    mode: str,
    results: list[BenchmarkQueryResult],
) -> BenchmarkFamilySummary:
    return BenchmarkFamilySummary(
        family=family,
        mode=mode,
        **_summary_values(results),
    )
