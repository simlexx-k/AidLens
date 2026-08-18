import pytest

from app.services.ingestion.archive import ArchiveIngestor


class FakeArchiveClient:
    async def list_evaluation_ids(self, page: int) -> list[str]:
        assert page == 3
        return ["EXISTING", "NEW-1", "NEW-2"]


class ResumeTestIngestor(ArchiveIngestor):
    async def _existing_ids(self, ids: list[str]) -> set[str]:
        assert "EXISTING" in ids
        return {"EXISTING"}

    async def _ingest_one(self, external_id: str) -> str:
        return external_id


@pytest.mark.asyncio
async def test_skip_existing_avoids_refetching_known_evaluations() -> None:
    ingestor = ResumeTestIngestor(FakeArchiveClient(), None, concurrency=2)  # type: ignore[arg-type]

    stats = await ingestor.ingest_pages(
        pages=1,
        start_page=3,
        skip_existing=True,
    )

    assert stats == {
        "discovered": 3,
        "ingested": 2,
        "skipped": 1,
        "failed": 0,
    }
