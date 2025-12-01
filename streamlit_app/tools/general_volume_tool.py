from typing import Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class GeneralVolumeInput(BaseModel):
    """Input per l'equazione generale del volume."""
    diameter_m: float = Field(description="Diametro a petto d'uomo (DBH) in metri (D)")
    height_m: float = Field(description="Altezza dell'albero in metri (H)")
    coefficient_a: float = Field(description="Coefficiente empirico (a), dipende dalla specie")

class GeneralVolumeTool(BaseTool):
    """Tool per calcolare il volume usando l'Equazione Generale."""
    name: str = "calculate_general_volume"
    description: str = """
    Calcola il volume del fusto usando l'Equazione generale: V = a * D² * H.
    Modello allometrico classico per stimare il volume a partire da diametro e altezza.
    """
    args_schema: Type[BaseModel] = GeneralVolumeInput

    def _run(self, diameter_m: float, height_m: float, coefficient_a: float) -> dict:
        volume = coefficient_a * (diameter_m ** 2) * height_m
        return {
            "volume": volume,
            "unit": "m³",
            "formula": "V = a * D² * H",
            "method": "General Volume Equation"
        }

