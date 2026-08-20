import json
import math
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Annotated

import httpx
import typer

from app.services.ranker.dataset import (
    load_ranker_records,
    split_ranker_records,
    write_ranker_split,
)
from app.services.ranker.evaluation import (
    DEFAULT_FUSION_ALPHAS,
    evaluate_aidranker_fusion_sweep_model,
    evaluate_aidranker_model,
    write_ranker_report,
)
from app.services.ranker.fixed_fusion import evaluate_aidranker_fixed_fusion_model
from app.services.ranker.training import DEFAULT_RANKER_MODEL, train_aidranker

cli = typer.Typer(
    no_args_is_help=True,
    help="Offline AidRanker V1 experiments and serving benchmarks.",
)


@cli.command("split")
def split_command(
    dataset: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Compiled AidRanker JSONL records.",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory for train/dev/test JSONL."),
    ],
    seed: Annotated[int, typer.Option(help="Deterministic query split seed.")] = 42,
) -> None:
    """Create a 3/1/1 query split per evidence family."""

    records = load_ranker_records(dataset)
    split = split_ranker_records(records, seed=seed)
    write_ranker_split(split, output_dir, seed=seed)
    typer.echo(
        " ".join(
            [
                f"train_queries={len(split.query_ids['train'])}",
                f"dev_queries={len(split.query_ids['dev'])}",
                f"test_queries={len(split.query_ids['test'])}",
                f"train_records={len(split.train)}",
                f"dev_records={len(split.dev)}",
                f"test_records={len(split.test)}",
                f"output={output_dir}",
            ]
        )
    )


@cli.command("train")
def train_command(
    dataset: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Training split JSONL.",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Model output directory."),
    ],
    model_name: Annotated[
        str,
        typer.Option("--model", help="Pretrained CrossEncoder model."),
    ] = DEFAULT_RANKER_MODEL,
    epochs: Annotated[int, typer.Option(min=1, max=10)] = 1,
    batch_size: Annotated[int, typer.Option(min=1, max=128)] = 8,
    learning_rate: Annotated[float, typer.Option(min=1e-7, max=1e-3)] = 2e-5,
) -> None:
    """Fine-tune AidRanker locally using ML extras."""

    records = load_ranker_records(dataset)
    metadata = train_aidranker(
        records,
        output_dir,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    typer.echo(json.dumps(metadata, indent=2))


@cli.command("evaluate")
def evaluate_command(
    dataset: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Dev or held-out test split JSONL.",
        ),
    ],
    model_path: Annotated[
        str,
        typer.Option("--model-path", help="Fine-tuned CrossEncoder directory."),
    ],
    candidate_mode: Annotated[
        str,
        typer.Option(
            "--candidate-mode",
            help="First-stage candidates to rerank: semantic, lexical, or hybrid.",
        ),
    ] = "semantic",
    top_k: Annotated[int, typer.Option(min=1, max=50)] = 10,
    batch_size: Annotated[int, typer.Option(min=1, max=256)] = 32,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", dir_okay=False),
    ] = None,
) -> None:
    """Compare first-stage ranking with AidRanker on the same candidate set."""

    records = load_ranker_records(dataset)
    report = evaluate_aidranker_model(
        records,
        model_path,
        candidate_mode=candidate_mode,
        top_k=top_k,
        batch_size=batch_size,
    )
    if output:
        write_ranker_report(report, output)
        typer.echo(f"report={output}")
    typer.echo(json.dumps(report, indent=2))


@cli.command("sweep-fusion")
def sweep_fusion_command(
    dataset: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Development split JSONL. Do not use held-out test data here.",
        ),
    ],
    model_path: Annotated[
        str,
        typer.Option("--model-path", help="Fine-tuned CrossEncoder directory."),
    ],
    candidate_mode: Annotated[
        str,
        typer.Option(
            "--candidate-mode",
            help="First-stage candidates to fuse with AidRanker scores.",
        ),
    ] = "semantic",
    alphas: Annotated[
        str,
        typer.Option(
            help=(
                "Comma-separated global AidRanker weights. "
                "0=first-stage only, 1=AidRanker only."
            )
        ),
    ] = ",".join(str(alpha) for alpha in DEFAULT_FUSION_ALPHAS),
    diversity_tolerance: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=1.0,
            help="Maximum allowed increase in mean duplicate share.",
        ),
    ] = 0.05,
    top_k: Annotated[int, typer.Option(min=1, max=50)] = 10,
    batch_size: Annotated[int, typer.Option(min=1, max=256)] = 32,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", dir_okay=False),
    ] = None,
) -> None:
    """Select one dev-only semantic/AidRanker fusion weight."""

    records = load_ranker_records(dataset)
    report = evaluate_aidranker_fusion_sweep_model(
        records,
        model_path,
        candidate_mode=candidate_mode,
        top_k=top_k,
        batch_size=batch_size,
        alphas=_parse_alphas(alphas),
        diversity_tolerance=diversity_tolerance,
    )
    if output:
        write_ranker_report(report, output)
        typer.echo(f"report={output}")
    typer.echo(json.dumps(report, indent=2))


@cli.command("evaluate-fusion")
def evaluate_fusion_command(
    dataset: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Held-out test split JSONL for one frozen fusion alpha.",
        ),
    ],
    model_path: Annotated[
        str,
        typer.Option("--model-path", help="Fine-tuned CrossEncoder directory."),
    ],
    alpha: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=1.0,
            help="Frozen global AidRanker fusion weight selected on dev.",
        ),
    ],
    candidate_mode: Annotated[
        str,
        typer.Option(
            "--candidate-mode",
            help="First-stage candidates to fuse with AidRanker scores.",
        ),
    ] = "semantic",
    top_k: Annotated[int, typer.Option(min=1, max=50)] = 10,
    batch_size: Annotated[int, typer.Option(min=1, max=256)] = 32,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", dir_okay=False),
    ] = None,
) -> None:
    """Evaluate one already-frozen fusion alpha without test-time selection."""

    records = load_ranker_records(dataset)
    report = evaluate_aidranker_fixed_fusion_model(
        records,
        model_path,
        alpha=alpha,
        candidate_mode=candidate_mode,
        top_k=top_k,
        batch_size=batch_size,
    )
    if output:
        write_ranker_report(report, output)
        typer.echo(f"report={output}")
    typer.echo(json.dumps(report, indent=2))


@cli.command("benchmark-serving")
def benchmark_serving_command(
    api_url: Annotated[
        str,
        typer.Option(help="Evidence search endpoint URL."),
    ] = "http://localhost:8000/api/v1/search/evidence",
    query: Annotated[
        str,
        typer.Option(help="Representative production evidence query."),
    ] = "What implementation factors improved program outcomes?",
    repeats: Annotated[int, typer.Option(min=2, max=100)] = 7,
    top_k: Annotated[int, typer.Option(min=1, max=50)] = 10,
    max_per_evaluation: Annotated[int, typer.Option(min=1, max=20)] = 3,
    timeout_seconds: Annotated[float, typer.Option(min=1.0, max=300.0)] = 120.0,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", dir_okay=False),
    ] = None,
) -> None:
    """Benchmark repeated warm production-path searches against a running API."""

    payload = {
        "query": query,
        "mode": "auto",
        "rerank": "aidranker",
        "top_k": top_k,
        "max_per_evaluation": max_per_evaluation,
    }
    samples: list[dict[str, object]] = []
    with httpx.Client(timeout=timeout_seconds) as client:
        for index in range(repeats):
            started = perf_counter()
            response = client.post(api_url, json=payload)
            wall_ms = round((perf_counter() - started) * 1000.0, 3)
            response.raise_for_status()
            body = response.json()
            if not body.get("reranker_applied"):
                raise typer.BadParameter(
                    "Benchmark requires AidRanker to be applied; check API serving configuration."
                )
            samples.append(
                {
                    "run": index + 1,
                    "wall_ms": wall_ms,
                    "query_encoding_ms": body.get("query_encoding_latency_ms"),
                    "first_stage_ms": body.get("first_stage_latency_ms"),
                    "reranker_ms": body.get("reranker_latency_ms"),
                    "search_internal_ms": body.get("total_search_latency_ms"),
                    "request_ms": body.get("request_latency_ms"),
                    "model_load_ms": body.get("reranker_model_load_latency_ms"),
                    "backend": body.get("reranker_backend"),
                    "batch_size": body.get("reranker_batch_size"),
                    "device": body.get("reranker_device"),
                    "hits": len(body.get("hits", [])),
                    "groups": len(body.get("groups", [])),
                }
            )

    report = {
        "api_url": api_url,
        "query": query,
        "repeats": repeats,
        "samples": samples,
        "summary": {
            key: _latency_summary(samples, key)
            for key in (
                "wall_ms",
                "query_encoding_ms",
                "first_stage_ms",
                "reranker_ms",
                "search_internal_ms",
                "request_ms",
            )
        },
        "serving": {
            "backend": samples[-1]["backend"],
            "batch_size": samples[-1]["batch_size"],
            "device": samples[-1]["device"],
            "model_load_ms": samples[-1]["model_load_ms"],
        },
    }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        typer.echo(f"report={output}")
    typer.echo(json.dumps(report, indent=2))


def _latency_summary(samples: list[dict[str, object]], key: str) -> dict[str, float] | None:
    values = [float(sample[key]) for sample in samples if sample.get(key) is not None]
    if not values:
        return None
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


def _parse_alphas(value: str) -> list[float]:
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if not parts:
        raise typer.BadParameter("Provide at least one fusion alpha.")
    try:
        alphas = [float(item) for item in parts]
    except ValueError as exc:
        raise typer.BadParameter("Fusion alphas must be numeric.") from exc
    invalid = [alpha for alpha in alphas if alpha < 0.0 or alpha > 1.0]
    if invalid:
        raise typer.BadParameter("Fusion alphas must be between 0 and 1.")
    return list(dict.fromkeys(alphas))


if __name__ == "__main__":
    cli()
