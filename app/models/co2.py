from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class CO2CalculationRequest(BaseModel):
    dbh_cm: float = Field(gt=0, description="Diameter at breast height in centimeters")
    height_m: float = Field(gt=0, description="Tree height in meters")
    wood_density_g_cm3: float = Field(gt=0, description="Wood density in g/cm^3")
    carbon_fraction: float = Field(default=0.47, gt=0, lt=1, description="Carbon fraction of dry biomass")
    root_shoot_ratio: float = Field(default=0.24, gt=0, description="Root to shoot biomass ratio")

    # Optional input to estimate annual CO2 absorption (flow)
    annual_biomass_increment_t: Optional[float] = Field(
        default=None,
        ge=0,
        description="Annual increment of biomass (tonnes dry matter per year)",
    )


class CO2CalculationResponse(BaseModel):
    agb_t: float
    bgb_t: float
    total_biomass_t: float
    carbon_t: float
    co2_stock_t: float
    co2_annual_t: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)


