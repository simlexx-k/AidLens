from types import SimpleNamespace

import pytest

from app.services.ingestion.archive import ArchiveIngestor


@pytest.mark.asyncio
async def test_ingest_pages_skips_existing_records(monkeypatch) -> None:
    ingestor = ArchiveIngestor(SimpleNamespace(), object(), concurrency=1)

    async def list_evaluation_ids(page):
        return ["A", "B", "C"]

    async def existing_ids(ids):
        return {"A", "C"}

    ingested: list[str] = []

    async def ingest_one(external_id):
        ingested.append(external_id)
        return external_id

    ingestor.client.list_evaluation_ids = list_evaluation_ids
    monkeypatch.setattr(ingestor, "_existing_ids", existing_ids)
    monkeypatch.setattr(ingestor, "_ingest_one", ingest_one)

    stats = await ingestor.ingest_pages(1, skip_existing=True)

    assert stats == {"discovered": 3, "ingested": 1, "skipped": 2, "failed": 0}
    assert ingested == ["B"]


@pytest.mark.asyncio
async def test_ingest_evaluation_refreshes_one_external_id(monkeypatch) -> None:
    ingestor = ArchiveIngestor(object(), object(), concurrency=1)

    async def ingest_one(external_id):
        return f"refreshed-{external_id}"

    monkeypatch.setattr(ingestor, "_ingest_one", ingest_one)

    result = await ingestor.ingest_evaluation("ABC123")

    assert result == "refreshed-ABC123"
