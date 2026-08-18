from fastapi import APIRouter

from app.api.routes import analytics, evaluations, health, search

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(evaluations.router)
api_router.include_router(search.router)
api_router.include_router(analytics.router)
