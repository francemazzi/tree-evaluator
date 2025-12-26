import math
from typing import Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class RootBiomassInput(BaseModel):
    """Input per il calcolo della biomassa radicale."""
    diameter_cm: float = Field(description="Diametro DBH in cm (D)")
    height_m: float = Field(description="Altezza in metri (H)")
    age_years: float = Field(description="Età in anni")
    root_shoot_ratio: float = Field(description="Rapporto biomassa radici/chioma (es. 0.24)")

class RootBiomassTool(BaseTool):
    """Tool per calcolare la Biomassa Radicale."""
    name: str = "calculate_root_biomass"
    description: str = """
    Calcola la biomassa radicale: Root = e^(-5) * D^1.48 * H^0.4 * age^1.38 * (R_S)^0.31 * 1.26.
    Integra dimensioni, età e rapporto root-to-shoot.
    """
    args_schema: Type[BaseModel] = RootBiomassInput

    def _run(self, diameter_cm: float, height_m: float, age_years: float, root_shoot_ratio: float) -> dict:
        # Root = e^(-5) * D^1.48 * H^0.4 * age^1.38 * (Root_to_shoot_ratio)^0.31 * 1.26
        
        term1 = math.exp(-5)
        term2 = diameter_cm ** 1.48
        term3 = height_m ** 0.4
        term4 = age_years ** 1.38
        term5 = root_shoot_ratio ** 0.31
        
        biomass = term1 * term2 * term3 * term4 * term5 * 1.26
        
        return {
            "root_biomass": biomass,
            "unit": "kg (estimated)",
            "formula": "Root = e^(-5) * D^1.48 * H^0.4 * age^1.38 * (R_S)^0.31 * 1.26",
            "method": "Root Biomass Equation",
            "source": {
                "title": "Development of Allometric Equations to Determine the Biomass of Plant Components and the Total Storage of Carbon Dioxide in Young Mediterranean Argan Trees",
                "url": "https://www.researchgate.net/publication/380957635_Development_of_Allometric_Equations_to_Determine_the_Biomass_of_Plant_Components_and_the_Total_Storage_of_Carbon_Dioxide_in_Young_Mediterranean_Argan_Trees"
            }
        }

