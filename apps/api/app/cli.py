import asyncio
import json

import typer

from app.core.config import get_settings
from app.core.db import SessionLocal
from app.services.analytics.corpus import corpus_stats
from app.services.archive.aiddata import AidDataArchiveClient
from app.services.embeddings.indexer import embed_missing_chunks
from app.services.embeddings.sentence_transformer import SentenceTransformerEncoder
from app.services.ingestion.archive import ArchiveIngestor

cli = typer.Typer(no_args_is_help=True)


@cli.callback()
def main() -> None:
    """AidLens command-line utilities."""


@cli.command()
def ingest(
    pages: int = typer.Option(1, min=1, help="Number of archive result pages to ingest."),
    start_page: int = typer.Option(1, min=1, help="First archive page to ingest."),
    concurrency: int = typer.Option(4, min=1, max=10, help="Concurrent evaluation fetches."),
) -> None:
    """Ingest evaluation metadata and text from the AidData USAID archive."""

    async def run() -> None:
        settings = get_settings()
        async with AidDataArchiveClient(settings) as client:
            ingestor = ArchiveIngestor(client, SessionLocal, concurrency=concurrency)
            stats = await ingestor.ingest_pages(pages=pages, start_page=start_page)
            typer.echo(
                " ".join(
                    [
                        f"discovered={stats['discovered']}",
                        f"ingested={stats['ingested']}",
                        f"failed={stats['failed']}",
                    ]
                )
            )
    asyncio.run(run())


@cli.command()
def embed(
    batch_size: int = typer.Option(32, min=1, max=256, help="Embedding batch size."),
    limit: int | None = typer.Option(None, min=1, help="Maximum unembedded chunks to process."),
) -> None:
    """Generate embeddings for chunks that do not have vectors yet."""

    async def run() -> None:
        settings = get_settings()
        if settings.embedding_provider != "sentence-transformers":
            raise typer.BadParameter(
                "Set AIDLENS_EMBEDDING_PROVIDER=sentence-transformers before embedding."
            )
        encoder = SentenceTransformerEncoder(settings.embedding_model)
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
        async with SessionLocal() as session:
            stats = await corpus_stats(session)
        typer.echo(json.dumps(stats.model_dump(), indent=2))
    asyncio.run(run())


if __name__ == "__main__":
    cli()
