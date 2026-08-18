import asyncio
import json
from pathlib import Path

import typer

from app.core.config import get_settings
from app.services.analytics.corpus import corpus_stats
from app.services.archive.aiddata import AidDataArchiveClient
from app.services.embeddings.indexer import embed_missing_chunks
from app.services.embeddings.sentence_transformer import SentenceTransformerEncoder
from app.services.evaluation.benchmark import (
    load_benchmark_dataset,
    run_benchmark,
    write_report,
)
from app.services.evaluation.candidates import (
    generate_candidate_sets,
    load_candidate_queries,
    write_candidate_sets,
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


@cli.command()
def benchmark(
    dataset: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="JSONL benchmark judgments.",
    ),
    modes: str = typer.Option(
        "lexical,semantic,hybrid",
        help="Comma-separated retrieval modes.",
    ),
    top_k: int = typer.Option(10, min=1, max=50, help="Evaluation cutoff."),
    output: Path | None = typer.Option(
        None,
        dir_okay=False,
        help="Optional JSON report path.",
    ),
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
            )
        if output:
            write_report(report, output)
            typer.echo(f"report={output}")
        typer.echo(json.dumps(report.model_dump(), indent=2))

    asyncio.run(run())


@cli.command("export-ranking-candidates")
def export_ranking_candidates(
    queries: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="JSONL query file.",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        dir_okay=False,
        help="Output JSONL candidate pool.",
    ),
    mode: str = typer.Option("hybrid", help="lexical, semantic, or hybrid."),
    top_k: int = typer.Option(20, min=1, max=50, help="Candidates per query."),
) -> None:
    """Export retrieval candidates for AidRanker relevance labeling."""

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
            )
        write_candidate_sets(candidate_sets, output)
        typer.echo(
            f"queries={len(candidate_sets)} output={output} mode={selected_mode}"
        )

    asyncio.run(run())


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
