import math
from typing import Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class StemBiomassInput(BaseModel):
    """Input per il calcolo della biomassa del fusto."""
    diameter: float = Field(description="Diametro del fusto (D). Nota: Verificare unità (probabilmente metri per evitare overflow).")
    age_years: float = Field(description="Età dell'albero in anni")

class StemBiomassTool(BaseTool):
    """Tool per calcolare la Biomassa del Fusto."""
    name: str = "calculate_stem_biomass"
    description: str = """
    Calcola la biomassa del fusto: Stem = e^(-8.15 + 2.20D + 1.24 age - 0.35 D * age) * 1.41.
    Include un termine di interazione (D * age).
    """
    args_schema: Type[BaseModel] = StemBiomassInput

    def _run(self, diameter: float, age_years: float) -> dict:
        # Stem = e^(-8.15 + 2.20D + 1.24 age - 0.35 D * age) * 1.41
        
        exponent = -8.15 + (2.20 * diameter) + (1.24 * age_years) - (0.35 * diameter * age_years)
        
        try:
            term_exp = math.exp(exponent)
            biomass = term_exp * 1.41
        except OverflowError:
            return {"error": "Calculation resulted in overflow. Check input units (Diameter might need to be in meters)."}

        return {
            "stem_biomass": biomass,
            "unit": "kg (estimated)",
            "formula": "Stem = e^(-8.15 + 2.20D + 1.24 age - 0.35 D * age) * 1.41",
            "method": "Stem Biomass Equation"
        }

