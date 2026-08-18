from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.evaluation import EvaluationChunk
from app.services.embeddings.sentence_transformer import SentenceTransformerEncoder


async def embed_missing_chunks(
    session_factory: async_sessionmaker[AsyncSession],
    encoder: SentenceTransformerEncoder,
    *,
    batch_size: int = 32,
    limit: int | None = None,
) -> int:
    processed = 0
    while limit is None or processed < limit:
        current_batch_size = batch_size
        if limit is not None:
            current_batch_size = min(current_batch_size, limit - processed)
        if current_batch_size <= 0:
            break
        async with session_factory() as session:
            chunks = (
                await session.execute(
                    select(EvaluationChunk)
                    .where(EvaluationChunk.embedding.is_(None))
                    .order_by(EvaluationChunk.created_at, EvaluationChunk.id)
                    .limit(current_batch_size)
                )
            ).scalars().all()
            if not chunks:
                break
            vectors = encoder.encode_documents(
                [chunk.text for chunk in chunks], batch_size=current_batch_size
            )
            for chunk, vector in zip(chunks, vectors, strict=True):
                chunk.embedding = vector
                chunk.embedding_model = encoder.model_name
            await session.commit()
            processed += len(chunks)
    return processed
