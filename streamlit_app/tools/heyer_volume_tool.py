from typing import List, Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class HeyerVolumeInput(BaseModel):
    """Input per il calcolo del volume con formula di Heyer."""
    sections: List[float] = Field(description="Lista delle superfici delle sezioni trasversali del tronco (S1...Sn) in m²")

class HeyerVolumeTool(BaseTool):
    """Tool per calcolare il volume del fusto usando la Formula di Heyer."""
    name: str = "calculate_heyer_volume"
    description: str = """
    Calcola il volume del fusto usando la Formula di Heyer: V = S1 + S2 + ... + Sn.
    Somma delle superfici delle sezioni trasversali del tronco lungo la sua altezza.
    Utile per stimare il volume attraverso sezioni multiple.
    """
    args_schema: Type[BaseModel] = HeyerVolumeInput

    def _run(self, sections: List[float]) -> dict:
        volume = sum(sections)
        return {
            "volume": volume,
            "unit": "m³",
            "formula": "V = S1 + S2 + ... + Sn",
            "method": "Heyer Formula",
            "source": {
                "title": "Allometric relationships for volume and biomass for stone pine (Pinus pinea L.) in Italian coastal stands",
                "url": "https://www.researchgate.net/publication/256199126_Allometric_relationships_for_volume_and_biomass_for_stone_pine_Pinus_pinea_L_in_Italian_coastal_stands"
            }
        }

