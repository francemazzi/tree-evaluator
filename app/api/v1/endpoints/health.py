from fastapi import APIRouter, Depends

from app.core.config import get_app_config
from app.models.response import HealthCheckResponse
from app.services.health_service import HealthService


router = APIRouter(prefix="/health")


def get_health_service() -> HealthService:
    return HealthService(config=get_app_config())


@router.get("/", response_model=HealthCheckResponse)
async def health(service: HealthService = Depends(get_health_service)) -> HealthCheckResponse:
    return service.get_health()


