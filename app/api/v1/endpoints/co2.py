from fastapi import APIRouter, Depends

from app.models.co2 import CO2CalculationRequest, CO2CalculationResponse
from app.services.co2_service import CO2CalculationService


router = APIRouter(prefix="/co2")


def get_co2_service() -> CO2CalculationService:
    return CO2CalculationService()


@router.post("/calc", response_model=CO2CalculationResponse)
async def calculate_co2(
    payload: CO2CalculationRequest,
    service: CO2CalculationService = Depends(get_co2_service),
) -> CO2CalculationResponse:
    return service.calculate(payload)


