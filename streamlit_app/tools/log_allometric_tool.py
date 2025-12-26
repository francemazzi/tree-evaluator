import math
from typing import Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class LogAllometricInput(BaseModel):
    """Input per la forma logaritmica della relazione allometrica."""
    variable_x: float = Field(description="Variabile predittiva (X)")
    coeff_a: float = Field(description="Parametro a (non logaritmico)")
    coeff_b: float = Field(description="Parametro b (coefficiente angolare)")

class LogAllometricTool(BaseTool):
    """Tool per calcolare la Forma Logaritmica della Relazione Allometrica."""
    name: str = "calculate_log_allometric"
    description: str = """
    Calcola ln(Y) usando la forma linearizzata: ln Y = ln a + b ln X.
    Usata per regressioni lineari su dati trasformati.
    """
    args_schema: Type[BaseModel] = LogAllometricInput

    def _run(self, variable_x: float, coeff_a: float, coeff_b: float) -> dict:
        # ln Y = ln(a) + b * ln(X)
        ln_x = math.log(variable_x)
        ln_a = math.log(coeff_a)
        ln_y = ln_a + coeff_b * ln_x
        return {
            "ln_y": ln_y,
            "ln_x": ln_x,
            "formula": "ln Y = ln a + b ln X",
            "method": "Logarithmic Allometric",
            "source": {
                "title": "Allometric equations to calculate living and dead fuel loads in Mediterranean species",
                "url": "https://www.researchgate.net/publication/377661797_Allometric_equations_to_calculate_living_and_dead_fuel_loads_in_Mediterranean_species"
            }
        }

