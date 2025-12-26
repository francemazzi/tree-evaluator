from typing import Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class ModelErrorInput(BaseModel):
    """Input per il calcolo dell'errore relativo."""
    measured_value: float = Field(description="Valore reale osservato (y)")
    estimated_value: float = Field(description="Valore stimato dal modello (ŷ)")

class ModelErrorTool(BaseTool):
    """Tool per calcolare l'Errore Relativo del Modello."""
    name: str = "calculate_model_error"
    description: str = """
    Calcola l'errore relativo: RD = |y - ŷ| / y.
    Misura l'accuratezza della previsione rispetto al valore reale.
    """
    args_schema: Type[BaseModel] = ModelErrorInput

    def _run(self, measured_value: float, estimated_value: float) -> dict:
        if measured_value == 0:
            return {"error": "Measured value cannot be zero"}
        
        relative_error = abs(measured_value - estimated_value) / measured_value
        return {
            "relative_error": relative_error,
            "percentage_error": relative_error * 100,
            "formula": "RD = |y - ŷ| / y",
            "method": "Relative Model Error",
            "sources": [
                {
                    "title": "Standard committee on the exchange of observations",
                    "url": "https://journals.ametsoc.org/view/journals/bams/63/11/1520-0477_1982_063_1309_scoteo_2_0_co_2.xml"
                },
                {
                    "title": "Classical and Modern Regression With Applications",
                    "url": "https://www.researchgate.net/publication/261580010_Classical_and_Modern_Regression_With_Applications"
                }
            ]
        }

