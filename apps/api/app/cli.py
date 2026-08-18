import asyncio

import typer

from app.core.config import get_settings
from app.core.db import SessionLocal
from app.services.archive.aiddata import AidDataArchiveClient
from app.services.ingestion.archive import ArchiveIngestor

cli = typer.Typer(no_args_is_help=True)


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


if __name__ == "__main__":
    cli()
