from fastapi import APIRouter

from app.api.dependencies import DbSession
from app.schemas.analytics import CorpusStats
from app.services.analytics.corpus import corpus_stats

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/corpus", response_model=CorpusStats)
async def get_corpus_stats(db: DbSession) -> CorpusStats:
    return await corpus_stats(db)
