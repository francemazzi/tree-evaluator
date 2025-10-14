from fastapi import APIRouter

from app.api.v1.endpoints.health import router as health_router
from app.api.v1.endpoints.co2 import router as co2_router
from app.api.v1.endpoints.environment import router as environment_router


api_v1_router = APIRouter(prefix="/api/v1")

# Register endpoint groups
api_v1_router.include_router(health_router, tags=["health"])
api_v1_router.include_router(co2_router, tags=["co2"])
api_v1_router.include_router(environment_router, tags=["environment"])


