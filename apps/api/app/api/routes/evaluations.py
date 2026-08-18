from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import func, select

from app.api.dependencies import DbSession
from app.models.evaluation import Evaluation
from app.schemas.evaluation import EvaluationDetail, EvaluationSummary

router = APIRouter(prefix="/evaluations", tags=["evaluations"])


@router.get("")
async def list_evaluations(
    db: DbSession,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict[str, object]:
    total = (await db.execute(select(func.count()).select_from(Evaluation))).scalar_one()
    rows = (
        await db.execute(
            select(Evaluation)
            .order_by(Evaluation.publication_year.desc().nullslast(), Evaluation.title)
            .offset(offset)
            .limit(limit)
        )
    ).scalars().all()
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": [EvaluationSummary.model_validate(row) for row in rows],
    }


@router.get("/{external_id}", response_model=EvaluationDetail)
async def get_evaluation(
    external_id: str,
    db: DbSession,
) -> EvaluationDetail:
    row = (
        await db.execute(select(Evaluation).where(Evaluation.external_id == external_id))
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return EvaluationDetail.model_validate(row)
