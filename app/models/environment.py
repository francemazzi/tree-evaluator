from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class TreeInput(BaseModel):
    """Tree-related inputs.

    Units:
    - diameter_cm: centimeters (cm)
    - height_m: meters (m)
    - wood_density_kg_m3: kilograms per cubic meter (kg/m^3)
    - carbon_fraction: fraction in (0,1], default 0.47
    """

    diameter_cm: float = Field(gt=0, description="Diameter at breast height in centimeters (cm)")
    height_m: Optional[float] = Field(default=None, ge=0, description="Tree height in meters (m)")
    wood_density_kg_m3: Optional[float] = Field(
        default=None, gt=0, description="Wood density in kilograms per cubic meter (kg/m^3)"
    )
    carbon_fraction: Optional[float] = Field(
        default=None, gt=0, le=1, description="Carbon fraction of dry biomass in (0,1]"
    )


class SiteInput(BaseModel):
    site_id: str = Field(min_length=1)
    lat: Optional[float] = Field(default=None)
    lon: Optional[float] = Field(default=None)

    @field_validator("lat")
    @classmethod
    def validate_lat(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return v
        if v < -90 or v > 90:
            raise ValueError("lat must be in [-90, 90]")
        return v

    @field_validator("lon")
    @classmethod
    def validate_lon(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return v
        if v < -180 or v > 180:
            raise ValueError("lon must be in [-180, 180]")
        return v


class MethodInput(BaseModel):
    use_log_form: bool = Field(default=False)
    rsr_override: Optional[float] = Field(default=None, ge=0)
    bef_mode: Literal["none", "stemBased", "volumeBased"] = Field(default="none")


class FeedbackInput(BaseModel):
    observed_biomass_kg: Optional[float] = Field(default=None, gt=0)
    notes: Optional[str] = Field(default=None)


class MetaInput(BaseModel):
    request_id: str
    timestamp: datetime
    source: Literal["api", "batch", "ui"]


class CoefficientsInput(BaseModel):
    """Optional configurable coefficients for allometric equations."""

    volume_with_h_coef: float = Field(default=0.039, gt=0, description="V = c * D^2 * H")
    volume_without_h_coef: float = Field(default=0.77, gt=0, description="V = c * D^2")
    biomass_a: float = Field(default=0.035, gt=0, description="Y = a * D^b")
    biomass_b: float = Field(default=2.71, gt=0, description="Y = a * D^b")


class EnvironmentalEstimatesRequest(BaseModel):
    """Input schema for environmental estimates computation.

    The model validates and normalizes user-provided inputs.
    """

    tree: TreeInput
    site: SiteInput
    method: MethodInput
    feedback: Optional[FeedbackInput] = Field(default=None)
    meta: MetaInput
    coeffs: Optional[CoefficientsInput] = Field(default=None)


class ConfidenceBlock(BaseModel):
    method: Literal["analytical", "heuristic"]
    notes: str
    relative_error_rd: Optional[float] = None


class Citation(BaseModel):
    source: str
    equations: List[str]


class LoggingInfo(BaseModel):
    logged: bool
    log_id: Optional[str] = None


class EnvironmentalEstimatesResults(BaseModel):
    volume_dm3: float
    biomass_kg: float
    carbon_stock_kg: float
    rsr_used: float
    bef: Optional[float] = None
    confidence: ConfidenceBlock


class EnvironmentalEstimatesResponse(BaseModel):
    request_id: str
    model_version: str
    inputs: Dict[str, Any]
    results: EnvironmentalEstimatesResults
    citations: List[Citation]
    logging: LoggingInfo

    model_config = ConfigDict(from_attributes=True)


class ErrorDetail(BaseModel):
    code: Literal[
        "VALIDATION_ERROR",
        "COMPUTATION_ERROR",
        "INTERNAL_ERROR",
    ]
    message: str
    details: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


