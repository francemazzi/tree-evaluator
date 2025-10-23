from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# Add app directory to path
app_dir = Path(__file__).parent.parent.parent / "app"
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir.parent))

from app.models.environment import EnvironmentalEstimatesRequest, TreeInput
from app.services.environment_service import EnvironmentalEstimationService


class EnvironmentEstimationInput(BaseModel):
    """Input schema for environmental estimation tool."""

    diameter_cm: float = Field(description="Tree diameter at breast height in centimeters (must be > 0)")
    height_m: Optional[float] = Field(
        default=None,
        description="Tree height in meters (optional, if not provided uses diameter-only formula)",
    )
    carbon_fraction: float = Field(
        default=0.47,
        description="Carbon fraction of dry biomass (default 0.47)",
    )


class EnvironmentEstimationTool(BaseTool):
    """Tool to compute environmental estimates (volume, biomass, carbon) using existing service."""

    name: str = "calculate_environmental_estimates"
    description: str = """
    Calculate environmental estimates for a tree including volume, biomass, and carbon stock.
    
    Inputs:
    - diameter_cm: diameter at breast height in centimeters
    - height_m: tree height in meters (optional)
    - carbon_fraction: carbon fraction (default 0.47)
    
    Returns JSON with:
    - volume_dm3: tree volume in cubic decimeters
    - biomass_kg: biomass in kilograms
    - carbon_stock_kg: carbon stock in kilograms
    - confidence metrics and other analytical data
    
    Use this for environmental impact assessment, volume estimation, or when the user asks about
    tree measurements and environmental parameters beyond just CO2.
    """
    args_schema: Type[BaseModel] = EnvironmentEstimationInput

    _service: EnvironmentalEstimationService

    def __init__(self, service: Optional[EnvironmentalEstimationService] = None, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "_service", service or EnvironmentalEstimationService())

    def _run(
        self,
        diameter_cm: float,
        height_m: Optional[float] = None,
        carbon_fraction: float = 0.47,
    ) -> dict:
        """Execute the environmental estimation."""
        request = EnvironmentalEstimatesRequest(
            tree=TreeInput(
                diameter_cm=diameter_cm,
                height_m=height_m,
                carbon_fraction=carbon_fraction,
            )
        )
        response = self._service.computeEnvironmentalEstimates(request)
        return response.model_dump()

