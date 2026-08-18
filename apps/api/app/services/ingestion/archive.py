import asyncio

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.evaluation import Evaluation, EvaluationChunk
from app.services.archive.aiddata import AidDataArchiveClient, ArchiveEvaluation
from app.services.ingestion.chunker import chunk_document


class ArchiveIngestor:
    def __init__(
        self,
        client: AidDataArchiveClient,
        session_factory: async_sessionmaker[AsyncSession],
        concurrency: int = 4,
    ) -> None:
        self.client = client
        self.session_factory = session_factory
        self.semaphore = asyncio.Semaphore(max(1, concurrency))

    async def ingest_pages(self, pages: int, start_page: int = 1) -> dict[str, int]:
        discovered = 0
        ingested = 0
        failed = 0

        for page in range(start_page, start_page + pages):
            ids = await self.client.list_evaluation_ids(page)
            discovered += len(ids)
            results = await asyncio.gather(
                *(self._ingest_one(item) for item in ids),
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, Exception):
                    failed += 1
                else:
                    ingested += 1

        return {"discovered": discovered, "ingested": ingested, "failed": failed}

    async def _ingest_one(self, external_id: str) -> str:
        async with self.semaphore:
            detail = await self.client.fetch_evaluation(external_id)
            text = await self.client.fetch_text(detail.text_url) if detail.text_url else None

        async with self.session_factory() as session:
            evaluation = await self._upsert_evaluation(session, detail)
            if text:
                await session.execute(
                    delete(EvaluationChunk).where(
                        EvaluationChunk.evaluation_id == evaluation.id
                    )
                )
                for chunk in chunk_document(text):
                    session.add(
                        EvaluationChunk(
                            evaluation_id=evaluation.id,
                            ordinal=chunk.ordinal,
                            section=chunk.section,
                            text=chunk.text,
                        )
                    )
            await session.commit()
        return external_id

    @staticmethod
    async def _upsert_evaluation(
        session: AsyncSession,
        detail: ArchiveEvaluation,
    ) -> Evaluation:
        result = await session.execute(
            select(Evaluation).where(Evaluation.external_id == detail.external_id)
        )
        evaluation = result.scalar_one_or_none()
        values = {
            "title": detail.title,
            "publication_year": detail.publication_year,
            "language": detail.language,
            "project_title": detail.project_title,
            "abstract": detail.abstract,
            "authors": detail.authors,
            "institutions": detail.institutions,
            "keywords": detail.keywords,
            "locations": detail.locations,
            "contract_codes": detail.contract_codes,
            "source_url": detail.source_url,
            "pdf_url": detail.pdf_url,
            "text_url": detail.text_url,
            "file_size_kb": detail.file_size_kb,
            "raw_metadata": detail.raw_metadata,
        }
        if evaluation is None:
            evaluation = Evaluation(external_id=detail.external_id, **values)
            session.add(evaluation)
            await session.flush()
        else:
            for key, value in values.items():
                setattr(evaluation, key, value)
            await session.flush()
        return evaluation
