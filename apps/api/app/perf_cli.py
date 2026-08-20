import json
import math
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Annotated

import httpx
import typer

FROZEN_ALPHA = 0.50

cli = typer.Typer(
    no_args_is_help=True,
    help="AidLens serving-performance experiments that preserve the frozen ranking contract.",
)


@cli.command("benchmark-batches")
def benchmark_batches_command(
    model_path: Annotated[
        str,
        typer.Option("--model-path", help="Fine-tuned AidRanker V1 directory."),
    ] = "models/aidranker-v1.local",
    api_url: Annotated[
        str,
        typer.Option(help="Evidence search endpoint used to fetch the semantic candidate pool."),
    ] = "http://localhost:8000/api/v1/search/evidence",
    query: Annotated[
        str,
        typer.Option(help="Representative production evidence query."),
    ] = "What implementation factors improved program outcomes?",
    batch_sizes: Annotated[
        str,
        typer.Option(help="Comma-separated Torch batch sizes to compare."),
    ] = "8,16,32,40",
    repeats: Annotated[int, typer.Option(min=2, max=20)] = 3,
    candidate_k: Annotated[int, typer.Option(min=10, max=100)] = 40,
    top_k: Annotated[int, typer.Option(min=1, max=50)] = 10,
    max_per_evaluation: Annotated[int, typer.Option(min=1, max=20)] = 3,
    device: Annotated[str, typer.Option(help="Torch device or 'auto'.")] = "auto",
    tolerance: Annotated[
        float,
        typer.Option(min=0.0, max=0.1, help="Maximum raw-score delta allowed for parity."),
    ] = 1e-5,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", dir_okay=False),
    ] = None,
) -> None:
    """Compare Torch batch sizes on the exact same 40 semantic candidates."""

    sizes = _parse_batch_sizes(batch_sizes)
    candidates = _fetch_semantic_candidates(
        api_url,
        query=query,
        candidate_k=candidate_k,
    )
    if len(candidates) != candidate_k:
        raise typer.BadParameter(
            f"Expected {candidate_k} semantic candidates, received {len(candidates)}."
        )

    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:  # pragma: no cover - local ML environment only
        raise typer.BadParameter(
            "benchmark-batches requires the API ML extras."
        ) from exc

    load_started = perf_counter()
    model = CrossEncoder(
        model_path,
        device=None if device == "auto" else device,
    )
    model_load_ms = _elapsed_ms(load_started)
    resolved_device = str(getattr(model, "device", device))
    pairs = [(query, candidate["text"]) for candidate in candidates]

    # Establish the production baseline with the currently deployed batch size.
    baseline_scores = _predict(model, pairs, batch_size=32)
    baseline_final = _final_top_ids(
        candidates,
        baseline_scores,
        top_k=top_k,
        max_per_evaluation=max_per_evaluation,
    )

    results: list[dict[str, object]] = []
    for size in sizes:
        try:
            # One unmeasured pass removes per-shape warmup from the timings.
            _predict(model, pairs, batch_size=size)
            samples: list[float] = []
            scores: list[float] = []
            for _ in range(repeats):
                started = perf_counter()
                scores = _predict(model, pairs, batch_size=size)
                samples.append(_elapsed_ms(started))

            max_delta = max(
                abs(current - baseline)
                for current, baseline in zip(scores, baseline_scores, strict=True)
            )
            final_top = _final_top_ids(
                candidates,
                scores,
                top_k=top_k,
                max_per_evaluation=max_per_evaluation,
            )
            result = {
                "batch_size": size,
                "status": "ok",
                "latency_ms": _summary(samples),
                "max_abs_raw_score_delta_vs_batch32": max_delta,
                "final_top10_identical_to_batch32": final_top == baseline_final,
                "passes_parity": max_delta <= tolerance and final_top == baseline_final,
                "final_top_chunk_ids": final_top,
            }
        except RuntimeError as exc:
            result = {
                "batch_size": size,
                "status": "error",
                "error": str(exc),
                "passes_parity": False,
            }
        results.append(result)

    viable = [
        result
        for result in results
        if result.get("passes_parity") and result.get("status") == "ok"
    ]
    selected = min(
        viable,
        key=lambda item: float(item["latency_ms"]["median"]),  # type: ignore[index]
    ) if viable else None

    report = {
        "query": query,
        "model_path": model_path,
        "backend": "torch",
        "resolved_device": resolved_device,
        "model_load_ms": model_load_ms,
        "candidate_k": candidate_k,
        "top_k": top_k,
        "max_per_evaluation": max_per_evaluation,
        "frozen_alpha": FROZEN_ALPHA,
        "baseline_batch_size": 32,
        "baseline_final_top_chunk_ids": baseline_final,
        "results": results,
        "selection": (
            {
                "status": "selected",
                "batch_size": selected["batch_size"],
                "median_ms": selected["latency_ms"]["median"],  # type: ignore[index]
            }
            if selected is not None
            else {"status": "no_parity_candidate"}
        ),
    }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        typer.echo(f"report={output}")
    typer.echo(json.dumps(report, indent=2))


def _fetch_semantic_candidates(
    api_url: str,
    *,
    query: str,
    candidate_k: int,
) -> list[dict[str, object]]:
    payload = {
        "query": query,
        "mode": "semantic",
        "rerank": "disabled",
        "top_k": candidate_k,
    }
    with httpx.Client(timeout=120.0) as client:
        response = client.post(api_url, json=payload)
        response.raise_for_status()
        body = response.json()
    return list(body.get("hits", []))


def _predict(model, pairs: list[tuple[str, str]], *, batch_size: int) -> list[float]:
    predictions = model.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    return [float(score) for score in predictions]


def _final_top_ids(
    candidates: list[dict[str, object]],
    reranker_scores: list[float],
    *,
    top_k: int,
    max_per_evaluation: int,
) -> list[str]:
    semantic_scores = [
        float(
            candidate.get("semantic_score")
            if candidate.get("semantic_score") is not None
            else candidate["score"]
        )
        for candidate in candidates
    ]
    semantic_normalized = _minmax(semantic_scores)
    ranker_normalized = _minmax(reranker_scores)
    fused = [
        ((1.0 - FROZEN_ALPHA) * semantic) + (FROZEN_ALPHA * reranker)
        for semantic, reranker in zip(
            semantic_normalized,
            ranker_normalized,
            strict=True,
        )
    ]
    ranked = sorted(
        range(len(candidates)),
        key=lambda index: (fused[index], semantic_scores[index]),
        reverse=True,
    )

    selected: list[str] = []
    counts: dict[str, int] = {}
    for index in ranked:
        evaluation_id = str(candidates[index]["evaluation_id"])
        if counts.get(evaluation_id, 0) >= max_per_evaluation:
            continue
        selected.append(str(candidates[index]["chunk_id"]))
        counts[evaluation_id] = counts.get(evaluation_id, 0) + 1
        if len(selected) >= top_k:
            break
    return selected


def _parse_batch_sizes(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise typer.BadParameter("Batch sizes must be integers.") from exc
    if not values or any(size < 1 or size > 256 for size in values):
        raise typer.BadParameter("Batch sizes must be between 1 and 256.")
    return list(dict.fromkeys(values))


def _minmax(values: list[float]) -> list[float]:
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return [0.5 for _ in values]
    span = high - low
    return [(value - low) / span for value in values]


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "min": round(min(values), 3),
        "median": round(median(values), 3),
        "p95": round(_percentile(values, 0.95), 3),
        "max": round(max(values), 3),
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000.0, 3)


if __name__ == "__main__":
    cli()
