from typing import Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class SimplifiedVolumeInput(BaseModel):
    """Input per l'equazione semplificata del volume."""
    diameter_m: float = Field(description="Diametro a petto d'uomo (DBH) in metri (D)")

class SimplifiedVolumeTool(BaseTool):
    """Tool per calcolare il volume usando l'Equazione Semplificata."""
    name: str = "calculate_simplified_volume"
    description: str = """
    Calcola il volume del fusto usando l'Equazione semplificata: V = 0.77 * D².
    Stima molto semplificata utilizzabile quando l'altezza non è nota.
    """
    args_schema: Type[BaseModel] = SimplifiedVolumeInput

    def _run(self, diameter_m: float) -> dict:
        volume = 0.77 * (diameter_m ** 2)
        return {
            "volume": volume,
            "unit": "m³",
            "formula": "V = 0.77 * D²",
            "method": "Simplified Volume Equation"
        }

