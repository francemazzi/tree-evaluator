from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.models.co2 import CO2CalculationRequest, CO2CalculationResponse


@dataclass
class AllometryCoefficients:
    intercept: float = 0.0673
    exponent: float = 0.976


class CO2CalculationService:
    def __init__(self, coefficients: Optional[AllometryCoefficients] = None) -> None:
        self._coefficients = coefficients or AllometryCoefficients()

    def calculate(self, request: CO2CalculationRequest) -> CO2CalculationResponse:
        agb_t = self._estimate_agb(
            dbh_cm=request.dbh_cm,
            height_m=request.height_m,
            wood_density_g_cm3=request.wood_density_g_cm3,
        )

        bgb_t = request.root_shoot_ratio * agb_t
        total_biomass_t = agb_t + bgb_t
        carbon_t = total_biomass_t * request.carbon_fraction
        co2_stock_t = carbon_t * (44.0 / 12.0)

        co2_annual_t: Optional[float] = None
        if request.annual_biomass_increment_t is not None:
            co2_annual_t = request.annual_biomass_increment_t * request.carbon_fraction * (44.0 / 12.0)

        return CO2CalculationResponse(
            agb_t=round(agb_t, 6),
            bgb_t=round(bgb_t, 6),
            total_biomass_t=round(total_biomass_t, 6),
            carbon_t=round(carbon_t, 6),
            co2_stock_t=round(co2_stock_t, 6),
            co2_annual_t=None if co2_annual_t is None else round(co2_annual_t, 6),
        )

    def _estimate_agb(self, dbh_cm: float, height_m: float, wood_density_g_cm3: float) -> float:
        # Chave et al. (2014) generalized equation: AGB = a*(WD*DBH^2*H)^b
        # Convert kg to tonnes by dividing by 1000
        a = self._coefficients.intercept
        b = self._coefficients.exponent
        agb_kg = a * ((wood_density_g_cm3 * (dbh_cm ** 2) * height_m) ** b)
        return agb_kg / 1000.0


