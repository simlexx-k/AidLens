from fastapi import APIRouter

from app.api.dependencies import DbSession
from app.schemas.evaluation import EvidenceSearchRequest, EvidenceSearchResponse
from app.services.search.lexical import lexical_search

router = APIRouter(prefix="/search", tags=["search"])


@router.post("/evidence", response_model=EvidenceSearchResponse)
async def search_evidence(
    payload: EvidenceSearchRequest,
    db: DbSession,
) -> EvidenceSearchResponse:
    hits = await lexical_search(db, payload)
    return EvidenceSearchResponse(query=payload.query, mode="lexical-baseline", hits=hits)
