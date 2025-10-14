from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.models.environment import (
    EnvironmentalEstimatesRequest,
    EnvironmentalEstimatesResponse,
    ErrorDetail,
    ErrorResponse,
)
from app.services.environment_service import EnvironmentalEstimationService


router = APIRouter(prefix="/environment")


def get_environment_service() -> EnvironmentalEstimationService:
    return EnvironmentalEstimationService()


@router.post(
    "/estimates",
    response_model=EnvironmentalEstimatesResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
)
async def compute_environmental_estimates(
    request: Request,
    service: EnvironmentalEstimationService = Depends(get_environment_service),
) -> JSONResponse | EnvironmentalEstimatesResponse:
    try:
        payload: Dict[str, Any] = await request.json()
        validated = EnvironmentalEstimatesRequest.model_validate(payload)
    except ValidationError as ve:
        # Map Pydantic errors to structured details
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error=ErrorDetail(
                    code="VALIDATION_ERROR",
                    message="Invalid input",
                    details={"errors": ve.errors()},
                )
            ).model_dump(),
        )
    try:
        result = service.computeEnvironmentalEstimates(validated)
        return result
    except HTTPException:
        raise
    except Exception as exc:  # Catch-all to return structured error
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(
                    code="COMPUTATION_ERROR",
                    message="Failed to compute environmental estimates",
                    details={"reason": str(exc)},
                )
            ).model_dump(),
        )


