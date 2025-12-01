import math
from typing import Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class LogFuelBiomassInput(BaseModel):
    """Input per l'equazione logaritmica fuel/biomassa."""
    variable_x: float = Field(description="Variabile predittiva (X), es. D, H")
    correction_factor: float = Field(description="Correction Factor (CF)")
    intercept_a: float = Field(description="Intercetta regressione logaritmica (a)")
    slope_b: float = Field(description="Coefficiente regressione logaritmica (b)")

class LogFuelBiomassTool(BaseTool):
    """Tool per calcolare Biomassa/Fuel Load con correzione logaritmica."""
    name: str = "calculate_log_fuel_biomass"
    description: str = """
    Calcola Y = CF * exp(a + b * Log(X)).
    Back-transform dell’equazione logaritmica con Correction Factor.
    """
    args_schema: Type[BaseModel] = LogFuelBiomassInput

    def _run(self, variable_x: float, correction_factor: float, intercept_a: float, slope_b: float) -> dict:
        # Y = CF * exp(a + b * ln(X))
        log_x = math.log(variable_x)
        exponent = intercept_a + slope_b * log_x
        y_value = correction_factor * math.exp(exponent)
        
        return {
            "result_y": y_value,
            "formula": "Y = CF * exp(a + b * Log(X))",
            "method": "Logarithmic Equation with CF"
        }

