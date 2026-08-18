from fastapi import APIRouter
from sqlalchemy import text

from app.api.dependencies import DbSession

router = APIRouter(tags=["system"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "aidlens-api"}


@router.get("/ready")
async def ready(db: DbSession) -> dict[str, str]:
    await db.execute(text("SELECT 1"))
    return {"status": "ready", "database": "ok"}
