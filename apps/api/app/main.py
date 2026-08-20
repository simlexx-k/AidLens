import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import get_settings
from app.core.db import engine
from app.services.embeddings.provider import get_embedding_encoder
from app.services.ranker.provider import get_aidranker_service

logger = logging.getLogger(__name__)
settings = get_settings()


async def _warm_serving_models() -> None:
    if settings.embedding_provider == "sentence-transformers":
        try:
            encoder = get_embedding_encoder(settings.embedding_model)
            await asyncio.to_thread(encoder.encode_query, "AidLens startup warmup")
            logger.info("Semantic encoder warmup complete: %s", settings.embedding_model)
        except Exception:  # pragma: no cover - deployment defensive path
            logger.exception("Semantic encoder warmup failed; requests may fail or cold-start.")

    if (
        settings.aidranker_provider == "sentence-transformers"
        and settings.aidranker_warmup
    ):
        service = get_aidranker_service(
            settings.aidranker_model,
            settings.aidranker_candidate_k,
            settings.aidranker_batch_size,
            settings.aidranker_device,
            settings.aidranker_fail_open,
        )
        try:
            await service.warmup()
            logger.info(
                "AidRanker warmup complete: model=%s load_ms=%s batch=%s device=%s",
                settings.aidranker_model,
                service.model_load_latency_ms,
                service.batch_size,
                service.device,
            )
        except Exception:  # pragma: no cover - deployment defensive path
            logger.exception("AidRanker warmup failed.")
            if not settings.aidranker_fail_open:
                raise


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    await _warm_serving_models()
    yield
    await engine.dispose()


app = FastAPI(
    title="AidLens API",
    version="0.1.0",
    description="Development Evidence Intelligence API",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router, prefix="/api/v1")
