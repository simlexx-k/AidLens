import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer

from app.core.config import get_settings
from app.services.analytics.corpus import corpus_audit, corpus_stats
from app.services.archive.aiddata import AidDataArchiveClient
from app.services.embeddings.indexer import embed_missing_chunks
from app.services.embeddings.sentence_transformer import SentenceTransformerEncoder
from app.services.evaluation.benchmark import (
    load_benchmark_dataset,
    run_benchmark,
    write_report,
)
from app.services.evaluation.candidates import (
    carry_forward_labels,
    generate_candidate_sets,
    generate_pooled_candidate_sets,
    load_candidate_queries,
    load_candidate_sets,
    write_candidate_sets,
)
from app.services.evaluation.labels import (
    compile_labeled_candidates,
    write_benchmark_queries,
    write_ranker_records,
)
from app.services.ingestion.archive import ArchiveIngestor

cli = typer.Typer(no_args_is_help=True)


@cli.callback()
def main() -> None:
    """AidLens command-line utilities."""


@cli.command()
def ingest(
    pages: int = typer.Option(1, min=1, help="Number of archive result pages to ingest."),
    start_page: int = typer.Option(1, min=1, help="First archive page to ingest."),
    concurrency: int = typer.Option(
        4,
        min=1,
        max=10,
        help="Concurrent evaluation fetches.",
    ),
    skip_existing: bool = typer.Option(
        False,
        "--skip-existing",
        help="Skip evaluations already present in PostgreSQL.",
    ),
) -> None:
    """Ingest evaluation metadata and text from the AidData USAID archive."""

    async def run() -> None:
        from app.core.db import SessionLocal

        settings = get_settings()
        async with AidDataArchiveClient(settings) as client:
            ingestor = ArchiveIngestor(client, SessionLocal, concurrency=concurrency)
            stats = await ingestor.ingest_pages(
                pages=pages,
                start_page=start_page,
                skip_existing=skip_existing,
            )
            typer.echo(
                " ".join(
                    [
                        f"discovered={stats['discovered']}",
                        f"ingested={stats['ingested']}",
                        f"skipped={stats['skipped']}",
                        f"failed={stats['failed']}",
                    ]
                )
            )

    asyncio.run(run())


@cli.command("refresh-evaluation")
def refresh_evaluation(external_id: str) -> None:
    """Refresh one evaluation by AidData external ID."""

    async def run() -> None:
        from app.core.db import SessionLocal

        settings = get_settings()
        async with AidDataArchiveClient(settings) as client:
            ingestor = ArchiveIngestor(client, SessionLocal, concurrency=1)
            refreshed = await ingestor.ingest_evaluation(external_id)
        typer.echo(f"refreshed={refreshed}")

    asyncio.run(run())


@cli.command()
def embed(
    batch_size: int = typer.Option(32, min=1, max=256, help="Embedding batch size."),
    limit: int | None = typer.Option(
        None,
        min=1,
        help="Maximum unembedded chunks to process.",
    ),
) -> None:
    """Generate embeddings for chunks that do not have vectors yet."""

    async def run() -> None:
        from app.core.db import SessionLocal

        settings, encoder = _semantic_encoder()
        processed = await embed_missing_chunks(
            SessionLocal,
            encoder,
            batch_size=batch_size,
            limit=limit,
        )
        typer.echo(f"embedded={processed} model={settings.embedding_model}")

    asyncio.run(run())


@cli.command("corpus-report")
def corpus_report() -> None:
    """Print corpus coverage and quality statistics as JSON."""

    async def run() -> None:
        from app.core.db import SessionLocal

        async with SessionLocal() as session:
            stats = await corpus_stats(session)
        typer.echo(json.dumps(stats.model_dump(), indent=2))

    asyncio.run(run())


@cli.command("corpus-audit")
def corpus_audit_command() -> None:
    """Print record-level corpus anomalies that need human review."""

    async def run() -> None:
        from app.core.db import SessionLocal

        async with SessionLocal() as session:
            audit = await corpus_audit(session)
        typer.echo(json.dumps(audit.model_dump(), indent=2))

    asyncio.run(run())


@cli.command()
def benchmark(
    dataset: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="JSONL benchmark judgments.",
        ),
    ],
    modes: Annotated[
        str,
        typer.Option(help="Comma-separated retrieval modes."),
    ] = "lexical,semantic,hybrid",
    top_k: Annotated[
        int,
        typer.Option(min=1, max=50, help="Evaluation cutoff."),
    ] = 10,
    max_per_evaluation: Annotated[
        int | None,
        typer.Option(
            min=1,
            max=20,
            help="Optional per-evaluation result cap for diversity experiments.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            dir_okay=False,
            help="Optional JSON report path.",
        ),
    ] = None,
) -> None:
    """Evaluate retrieval against human relevance judgments."""

    async def run() -> None:
        from app.core.db import SessionLocal

        selected_modes = _parse_modes(modes)
        encoder = None
        if any(mode in {"semantic", "hybrid"} for mode in selected_modes):
            _, encoder = _semantic_encoder()
        queries = load_benchmark_dataset(dataset)
        async with SessionLocal() as session:
            report = await run_benchmark(
                session,
                queries,
                modes=selected_modes,
                top_k=top_k,
                encoder=encoder,
                dataset_name=dataset.name,
                max_per_evaluation=max_per_evaluation,
            )
        if output:
            write_report(report, output)
            typer.echo(f"report={output}")
        typer.echo(json.dumps(report.model_dump(), indent=2))

    asyncio.run(run())


@cli.command("export-ranking-candidates")
def export_ranking_candidates(
    queries: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="JSONL query file.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            dir_okay=False,
            help="Output JSONL candidate pool.",
        ),
    ],
    mode: Annotated[
        str,
        typer.Option(help="lexical, semantic, or hybrid."),
    ] = "hybrid",
    top_k: Annotated[
        int,
        typer.Option(min=1, max=50, help="Candidates per query."),
    ] = 20,
    max_per_evaluation: Annotated[
        int,
        typer.Option(
            min=1,
            max=20,
            help="Maximum passages from one evaluation in an annotation pool.",
        ),
    ] = 3,
) -> None:
    """Export diversified candidates from one retrieval mode."""

    async def run() -> None:
        from app.core.db import SessionLocal

        selected_mode = mode.strip().lower()
        encoder = None
        if selected_mode in {"semantic", "hybrid"}:
            _, encoder = _semantic_encoder()
        query_items = load_candidate_queries(queries)
        async with SessionLocal() as session:
            candidate_sets = await generate_candidate_sets(
                session,
                query_items,
                mode=selected_mode,
                top_k=top_k,
                encoder=encoder,
                max_per_evaluation=max_per_evaluation,
            )
        write_candidate_sets(candidate_sets, output)
        typer.echo(
            " ".join(
                [
                    f"queries={len(candidate_sets)}",
                    f"output={output}",
                    f"mode={selected_mode}",
                    f"max_per_evaluation={max_per_evaluation}",
                ]
            )
        )

    asyncio.run(run())


@cli.command("export-pooled-candidates")
def export_pooled_candidates(
    queries: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="JSONL query file.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            dir_okay=False,
            help="Output pooled JSONL candidate set.",
        ),
    ],
    modes: Annotated[
        str,
        typer.Option(help="Comma-separated retrievers to pool."),
    ] = "lexical,semantic,hybrid",
    per_mode_k: Annotated[
        int,
        typer.Option(min=1, max=50, help="Candidates retrieved from each mode."),
    ] = 20,
    max_per_evaluation: Annotated[
        int,
        typer.Option(
            min=1,
            max=20,
            help="Maximum pooled passages from one evaluation.",
        ),
    ] = 5,
) -> None:
    """Export a system-neutral pool from independent retriever rankings."""

    async def run() -> None:
        from app.core.db import SessionLocal

        selected_modes = _parse_modes(modes)
        encoder = None
        if any(mode in {"semantic", "hybrid"} for mode in selected_modes):
            _, encoder = _semantic_encoder()
        query_items = load_candidate_queries(queries)
        async with SessionLocal() as session:
            candidate_sets = await generate_pooled_candidate_sets(
                session,
                query_items,
                modes=selected_modes,
                per_mode_k=per_mode_k,
                encoder=encoder,
                max_per_evaluation=max_per_evaluation,
            )
        write_candidate_sets(candidate_sets, output)
        candidate_count = sum(len(item.candidates) for item in candidate_sets)
        typer.echo(
            " ".join(
                [
                    f"queries={len(candidate_sets)}",
                    f"candidates={candidate_count}",
                    f"output={output}",
                    f"modes={','.join(selected_modes)}",
                    f"per_mode_k={per_mode_k}",
                    f"max_per_evaluation={max_per_evaluation}",
                ]
            )
        )

    asyncio.run(run())


@cli.command("carry-forward-labels")
def carry_forward_labels_command(
    candidates: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="New candidate pool that should receive existing labels.",
        ),
    ],
    previous: Annotated[
        Path,
        typer.Option(
            "--previous",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Previously reviewed candidate JSONL.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            dir_okay=False,
            help="Output candidate pool with reused labels.",
        ),
    ],
) -> None:
    """Reuse labels for query/chunk pairs already reviewed in an older pool."""

    candidate_sets = load_candidate_sets(candidates)
    previous_sets = load_candidate_sets(previous)
    copied = carry_forward_labels(candidate_sets, previous_sets)
    remaining = sum(
        candidate.relevance is None
        for item in candidate_sets
        for candidate in item.candidates
    )
    write_candidate_sets(candidate_sets, output)
    typer.echo(
        " ".join(
            [
                f"queries={len(candidate_sets)}",
                f"labels_copied={copied}",
                f"unlabeled_remaining={remaining}",
                f"output={output}",
            ]
        )
    )


@cli.command("compile-labels")
def compile_labels(
    candidates: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Fully human-labeled candidate JSONL.",
        ),
    ],
    judgments_output: Annotated[
        Path,
        typer.Option(
            "--judgments-output",
            dir_okay=False,
            help="Anchor-aware benchmark JSONL output.",
        ),
    ],
    ranker_output: Annotated[
        Path,
        typer.Option(
            "--ranker-output",
            dir_okay=False,
            help="AidRanker query-passage training JSONL output.",
        ),
    ],
) -> None:
    """Compile reviewed 0-3 labels into benchmark truth and ranker records."""

    candidate_sets = load_candidate_sets(candidates)
    judgments, ranker_records = compile_labeled_candidates(candidate_sets)
    write_benchmark_queries(judgments, judgments_output)
    write_ranker_records(ranker_records, ranker_output)
    positives = sum(record.relevance > 0 for record in ranker_records)
    negatives = len(ranker_records) - positives
    typer.echo(
        " ".join(
            [
                f"queries={len(judgments)}",
                f"records={len(ranker_records)}",
                f"positives={positives}",
                f"hard_negatives={negatives}",
                f"judgments={judgments_output}",
                f"ranker={ranker_output}",
            ]
        )
    )


def _semantic_encoder():
    settings = get_settings()
    if settings.embedding_provider != "sentence-transformers":
        raise typer.BadParameter(
            "Set AIDLENS_EMBEDDING_PROVIDER=sentence-transformers before using "
            "semantic retrieval."
        )
    return settings, SentenceTransformerEncoder(settings.embedding_model)


def _parse_modes(value: str) -> list[str]:
    modes = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not modes:
        raise typer.BadParameter("Provide at least one retrieval mode.")
    unsupported = set(modes) - {"lexical", "semantic", "hybrid"}
    if unsupported:
        raise typer.BadParameter(
            f"Unsupported retrieval mode(s): {', '.join(sorted(unsupported))}"
        )
    return list(dict.fromkeys(modes))


if __name__ == "__main__":
    cli()
