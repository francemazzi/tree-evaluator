import math
from typing import Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class TotalBiomassInput(BaseModel):
    """Input per il calcolo della biomassa totale."""
    diameter_cm: float = Field(description="Diametro DBH in cm (D)")
    height_m: float = Field(description="Altezza in metri (H)")
    age_years: float = Field(description="Età in anni")
    root_shoot_ratio: float = Field(description="Rapporto biomassa radici/chioma")

class TotalBiomassTool(BaseTool):
    """Tool per calcolare la Biomassa Totale."""
    name: str = "calculate_total_biomass"
    description: str = """
    Calcola la biomassa totale: Total = e^(-4.2) * D^1.36 * H^0.57 * age^1.67 * (R_S)^(-0.3) * 1.23.
    Combina componente aerea e radicale.
    """
    args_schema: Type[BaseModel] = TotalBiomassInput

    def _run(self, diameter_cm: float, height_m: float, age_years: float, root_shoot_ratio: float) -> dict:
        # Total = e^(-4.2) * D^1.36 * H^0.57 * age^1.67 * (Root_to_shoot_ratio)^(-0.3) * 1.23
        
        term1 = math.exp(-4.2)
        term2 = diameter_cm ** 1.36
        term3 = height_m ** 0.57
        term4 = age_years ** 1.67
        term5 = root_shoot_ratio ** (-0.3)
        
        biomass = term1 * term2 * term3 * term4 * term5 * 1.23
        
        return {
            "total_biomass": biomass,
            "unit": "kg (estimated)",
            "formula": "Total = e^(-4.2) * D^1.36 * H^0.57 * age^1.67 * (R_S)^(-0.3) * 1.23",
            "method": "Total Biomass Equation"
        }

