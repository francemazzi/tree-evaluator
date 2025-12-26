from typing import Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class AllometricRelationInput(BaseModel):
    """Input per la relazione allometrica generale."""
    variable_x: float = Field(description="Variabile indipendente (X), es. D, H o D²H")
    coeff_a: float = Field(description="Intercetta (a)")
    exponent_b: float = Field(description="Esponente (b)")

class AllometricRelationTool(BaseTool):
    """Tool per calcolare la Relazione Allometrica Generale."""
    name: str = "calculate_allometric_relation"
    description: str = """
    Calcola il valore Y usando la relazione allometrica: Y = a * X^b.
    Equazione fondamentale che lega biomassa/volume a variabili predittive.
    """
    args_schema: Type[BaseModel] = AllometricRelationInput

    def _run(self, variable_x: float, coeff_a: float, exponent_b: float) -> dict:
        y_value = coeff_a * (variable_x ** exponent_b)
        return {
            "result_y": y_value,
            "formula": "Y = a * X^b",
            "method": "Allometric Relation",
            "source": {
                "title": "Allometric relationships for volume and biomass for stone pine (Pinus pinea L.) in Italian coastal stands",
                "url": "https://www.researchgate.net/publication/256199126_Allometric_relationships_for_volume_and_biomass_for_stone_pine_Pinus_pinea_L_in_Italian_coastal_stands"
            }
        }

