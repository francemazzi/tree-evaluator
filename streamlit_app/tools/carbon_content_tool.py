from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# Add app directory to path
app_dir = Path(__file__).parent.parent.parent / "app"
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir.parent))

from app.services.carbon_content_service import CarbonContentService


class CarbonContentInput(BaseModel):
    """Input schema for carbon content lookup tool."""

    species: Optional[str] = Field(
        default=None,
        description="Tree species name to lookup (e.g., 'Maple', 'Oak', 'Pine'). If None, returns all available species.",
    )


class CarbonContentTool(BaseTool):
    """Tool to lookup carbon content data for tree species from the carbon_content.csv dataset."""

    name: str = "lookup_carbon_content"
    description: str = """
    Lookup carbon content fraction for tree species from the carbon_content.csv dataset.
    
    Use this tool when the user asks about:
    - Carbon content or carbon fraction for a specific species
    - Which species are available in the dataset
    - Carbon percentage or carbon assorbimento (absorption) for a species
    - "Stock di carbonio" o "carbonio assorbito" inteso come contenuto di carbonio della specie (non CO2)
    
    Input:
    - species: name of the tree species (optional). If not provided, returns list of all available species.
    
    Returns:
    - carbon_fraction: carbon content as a fraction (0-1)
    - carbon_percent: carbon content as percentage
    - range: range of values from scientific studies
    - number_of_values: number of studies used to calculate the mean
    - species_name: standardized species name from dataset
    
    Example uses:
    - "What is the carbon content for Maple?"
    - "Carbon fraction for Acer platanoides"
    - "Stock di carbonio per Quercus robur"
    - "List all species with carbon data"
    """
    args_schema: Type[BaseModel] = CarbonContentInput

    _service: CarbonContentService
    _csv_path: Path

    def __init__(self, csv_path: Optional[Path] = None, **kwargs):
        super().__init__(**kwargs)
        
        if csv_path is None:
            # Default to dataset/carbon_content.csv
            csv_path = Path(__file__).parent.parent.parent / "dataset" / "carbon_content.csv"
        
        object.__setattr__(self, "_csv_path", csv_path)
        object.__setattr__(self, "_service", CarbonContentService(csv_path))

    def _read_full_dataset(self) -> List[Dict[str, Any]]:
        """Read full dataset with all fields (not just the mean fraction)."""
        if not self._csv_path.exists():
            return []
        
        species_data = []
        with self._csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                species = row.get("Species", "").strip()
                mean_str = row.get("Mean", "").strip()
                range_str = row.get("Range", "").strip()
                num_values_str = row.get("Number of Values", "").strip()
                
                if species and mean_str:
                    try:
                        mean_percent = float(mean_str)
                        mean_fraction = mean_percent / 100.0
                        
                        species_data.append({
                            "species": species,
                            "carbon_fraction": round(mean_fraction, 4),
                            "carbon_percent": round(mean_percent, 2),
                            "range": range_str if range_str else "N/A",
                            "number_of_values": int(num_values_str) if num_values_str else 1,
                        })
                    except (ValueError, TypeError):
                        continue
        
        return species_data

    def _find_species(self, species_name: str) -> Optional[Dict[str, Any]]:
        """Find species data by name (case-insensitive, flexible matching)."""
        species_name_lower = species_name.lower().strip()
        all_data = self._read_full_dataset()
        
        # First try exact match
        for data in all_data:
            if data["species"].lower() == species_name_lower:
                return data
        
        # Then try partial match
        for data in all_data:
            if species_name_lower in data["species"].lower() or data["species"].lower() in species_name_lower:
                return data
        
        return None

    def _run(self, species: Optional[str] = None) -> Dict[str, Any]:
        """Execute carbon content lookup."""
        
        # If no species provided, return list of all available species
        if not species:
            all_data = self._read_full_dataset()
            return {
                "available_species": [item["species"] for item in all_data],
                "total_count": len(all_data),
                "dataset_source": str(self._csv_path),
                "note": "Use the species name to get detailed carbon content data for a specific species.",
                "scientific_source": {
                    "title": "Carbon Content of Tree Tissues: A Synthesis",
                    "author": "Martin et al. (2018)",
                    "url": "https://www.researchgate.net/publication/259443596_Carbon_Content_of_Tree_Tissues_A_Synthesis",
                    "description": "Comprehensive review of carbon content values across tree species"
                }
            }
        
        # Look up specific species
        species_data = self._find_species(species)
        
        if not species_data:
            # Species not found - provide helpful response
            all_data = self._read_full_dataset()
            similar_species = [
                item["species"] for item in all_data 
                if any(word in item["species"].lower() for word in species.lower().split())
            ]
            
            return {
                "found": False,
                "requested_species": species,
                "message": f"Specie '{species}' non trovata nel dataset carbon_content.csv",
                "similar_species": similar_species[:5] if similar_species else [],
                "available_species_count": len(all_data),
                "suggestion": "Usa il tool senza parametro 'species' per vedere tutte le specie disponibili."
            }
        
        # Species found - return detailed data
        return {
            "found": True,
            "species_name": species_data["species"],
            "carbon_fraction": species_data["carbon_fraction"],
            "carbon_percent": species_data["carbon_percent"],
            "range_percent": species_data["range"],
            "number_of_studies": species_data["number_of_values"],
            "dataset_source": str(self._csv_path),
            "interpretation": f"La specie {species_data['species']} ha un contenuto medio di carbonio del {species_data['carbon_percent']}% della biomassa secca (frazione {species_data['carbon_fraction']}).",
            "usage_note": "Questo valore può essere utilizzato nel calcolo di CO2 sequestration insieme a misure di diametro e altezza dell'albero.",
            "scientific_source": {
                "title": "Carbon Content of Tree Tissues: A Synthesis",
                "author": "Martin et al. (2018)",
                "url": "https://www.researchgate.net/publication/259443596_Carbon_Content_of_Tree_Tissues_A_Synthesis",
                "description": f"Basato su {species_data['number_of_values']} studi scientifici"
            }
        }

