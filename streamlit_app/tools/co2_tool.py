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
        description="Carbon fraction of dry biomass (default 0.47). Ignored if species is provided.",
    )
    root_shoot_ratio: float = Field(
        default=0.24,
        description="Root to shoot biomass ratio (default 0.24)",
    )
    species: Optional[str] = Field(
        default=None,
        description="Tree species name (e.g., 'Oak', 'Maple', 'Pine'). If provided, carbon_fraction will be automatically looked up from dataset.",
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
    - carbon_fraction: carbon fraction (default 0.47, ignored if species is provided)
    - root_shoot_ratio: root-to-shoot ratio (default 0.24)
    - species: tree species name (optional, e.g., 'Oak', 'Maple', 'Pine'). If provided, carbon_fraction will be automatically looked up from dataset
    - annual_biomass_increment_t: optional annual biomass increment in tonnes/year
    
    Returns JSON with:
    - agb_t: above-ground biomass in tonnes
    - bgb_t: below-ground biomass in tonnes
    - total_biomass_t: total biomass in tonnes
    - carbon_t: carbon stock in tonnes
    - co2_stock_t: CO2 stock in tonnes
    - co2_annual_t: annual CO2 uptake in tonnes/year (if increment provided)
    
    Use this when user asks about CO2, carbon sequestration, biomass for specific tree measurements.
    If the user specifies a tree species, use the species parameter to get species-specific carbon content.
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
        species: Optional[str] = None,
        annual_biomass_increment_t: Optional[float] = None,
    ) -> dict:
        """Execute the CO2 calculation."""
        request = CO2CalculationRequest(
            dbh_cm=dbh_cm,
            height_m=height_m,
            wood_density_g_cm3=wood_density_g_cm3,
            carbon_fraction=carbon_fraction,
            root_shoot_ratio=root_shoot_ratio,
            species=species,
            annual_biomass_increment_t=annual_biomass_increment_t,
        )
        response = self._service.calculate(request)
        result = response.model_dump()
        
        # Add scientific sources for each formula used
        result["formulas"] = {
            "agb": {
                "formula": "AGB = 0.0673 × (WD × DBH² × H)^0.976",
                "description": "Chave et al. (2014) generalized equation for above-ground biomass",
                "source": {
                    "title": "Improved allometric models to estimate the aboveground biomass of tropical trees",
                    "url": "https://doi.org/10.1111/gcb.12629"
                }
            },
            "bgb": {
                "formula": "BGB = AGB × R/S",
                "description": "Below-ground biomass from root-to-shoot ratio",
                "source": {
                    "title": "Root biomass allocation in the world's upland forests",
                    "url": "https://doi.org/10.1007/s004420050128"
                }
            },
            "total_biomass": {
                "formula": "Biomassa totale = AGB + BGB",
                "description": "Sum of above-ground and below-ground biomass"
            },
            "carbon": {
                "formula": "C = Biomassa totale × CF",
                "description": "Carbon content from biomass using species-specific or default carbon fraction",
                "source": {
                    "title": "Carbon Content of Tree Tissues: A Synthesis",
                    "url": "https://doi.org/10.1007/s10021-017-0198-4"
                }
            },
            "co2_stock": {
                "formula": "CO2 = C × (44/12)",
                "description": "CO2 equivalent from carbon using molecular weight ratio (CO2=44, C=12)"
            },
            "co2_annual": {
                "formula": "CO2 annuale = Incremento annuale × CF × (44/12)",
                "description": "Annual CO2 sequestration from biomass increment"
            }
        }
        
        # Add parameters used
        result["parameters"] = {
            "wood_density": {
                "value": wood_density_g_cm3,
                "unit": "g/cm³",
                "description": "Densità del legno (WD)"
            },
            "carbon_fraction": {
                "value": carbon_fraction,
                "unit": "adimensionale",
                "description": f"Frazione di carbonio (CF) = {carbon_fraction*100:.0f}%"
            },
            "root_shoot_ratio": {
                "value": root_shoot_ratio,
                "unit": "adimensionale",
                "description": "Rapporto radici/chioma (R/S)"
            }
        }
        
        return result

