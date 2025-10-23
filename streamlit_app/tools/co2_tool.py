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

from app.models.co2 import CO2CalculationRequest
from app.services.co2_service import CO2CalculationService


class CO2CalculationInput(BaseModel):
    """Input schema for CO2 calculation tool."""

    dbh_cm: float = Field(description="Diameter at breast height in centimeters (must be > 0)")
    height_m: float = Field(description="Tree height in meters (must be > 0)")
    wood_density_g_cm3: float = Field(
        default=0.6,
        description="Wood density in g/cm³ (default 0.6 for generic species, typical range 0.3-1.0)",
    )
    carbon_fraction: float = Field(
        default=0.47,
        description="Carbon fraction of dry biomass (default 0.47)",
    )
    root_shoot_ratio: float = Field(
        default=0.24,
        description="Root to shoot biomass ratio (default 0.24)",
    )
    annual_biomass_increment_t: Optional[float] = Field(
        default=None,
        description="Annual increment of biomass in tonnes per year (optional)",
    )


class CO2CalculationTool(BaseTool):
    """Tool to calculate CO2 sequestration and biomass for a single tree using existing service."""

    name: str = "calculate_co2_sequestration"
    description: str = """
    Calculate CO2 sequestration and biomass for a single tree.
    
    Inputs:
    - dbh_cm: diameter at breast height in centimeters
    - height_m: tree height in meters
    - wood_density_g_cm3: wood density (default 0.6 for generic, use species-specific if known)
    - carbon_fraction: carbon fraction (default 0.47)
    - root_shoot_ratio: root-to-shoot ratio (default 0.24)
    - annual_biomass_increment_t: optional annual biomass increment in tonnes/year
    
    Returns JSON with:
    - agb_t: above-ground biomass in tonnes
    - bgb_t: below-ground biomass in tonnes
    - total_biomass_t: total biomass in tonnes
    - carbon_t: carbon stock in tonnes
    - co2_stock_t: CO2 stock in tonnes
    - co2_annual_t: annual CO2 uptake in tonnes/year (if increment provided)
    
    Use this when user asks about CO2, carbon sequestration, biomass for specific tree measurements.
    """
    args_schema: Type[BaseModel] = CO2CalculationInput

    _service: CO2CalculationService

    def __init__(self, service: Optional[CO2CalculationService] = None, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "_service", service or CO2CalculationService())

    def _run(
        self,
        dbh_cm: float,
        height_m: float,
        wood_density_g_cm3: float = 0.6,
        carbon_fraction: float = 0.47,
        root_shoot_ratio: float = 0.24,
        annual_biomass_increment_t: Optional[float] = None,
    ) -> dict:
        """Execute the CO2 calculation."""
        request = CO2CalculationRequest(
            dbh_cm=dbh_cm,
            height_m=height_m,
            wood_density_g_cm3=wood_density_g_cm3,
            carbon_fraction=carbon_fraction,
            root_shoot_ratio=root_shoot_ratio,
            annual_biomass_increment_t=annual_biomass_increment_t,
        )
        response = self._service.calculate(request)
        return response.model_dump()

