import math
from typing import Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class LeafBiomassInput(BaseModel):
    """Input per il calcolo della biomassa fogliare."""
    diameter_cm: float = Field(description="Diametro a petto d'uomo (DBH) in cm (D)")
    height_m: float = Field(description="Altezza dell'albero in metri (H)")
    age_years: float = Field(description="Età dell'albero in anni")

class LeafBiomassTool(BaseTool):
    """Tool per calcolare la Biomassa Fogliare."""
    name: str = "calculate_leaf_biomass"
    description: str = """
    Calcola la biomassa fogliare: Leaf = e^(-7.21) * (D² * H)^0.6 * age^3.2 * 1.28.
    Le potenze descrivono l'effetto non lineare di dimensioni e età.
    """
    args_schema: Type[BaseModel] = LeafBiomassInput

    def _run(self, diameter_cm: float, height_m: float, age_years: float) -> dict:
        # Leaf = e^(-7.21) * (D² * H)^0.6 * age^3.2 * 1.28
        # Assuming D in formula might need consistent units (e.g. cm or m).
        # Based on coefficients, we calculate as provided.
        
        term1 = math.exp(-7.21)
        d2h = (diameter_cm ** 2) * height_m
        term2 = d2h ** 0.6
        term3 = age_years ** 3.2
        
        biomass = term1 * term2 * term3 * 1.28
        
        return {
            "leaf_biomass": biomass,
            "unit": "kg (estimated)",
            "formula": "Leaf = e^(-7.21) * (D² * H)^0.6 * age^3.2 * 1.28",
            "method": "Leaf Biomass Equation"
        }

