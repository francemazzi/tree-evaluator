"""Carbon Sequestration Tool - Annual carbon storage rates by species (Paoletti et al.)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from streamlit_app.constants import CO2_C_RATIO, DEFAULT_SEQUESTRATION_MATURITA_KG_YEAR, DEFAULT_SEQUESTRATION_MEDIO_KG_YEAR


class CarbonSequestrationInput(BaseModel):
    """Input schema for carbon sequestration lookup tool."""

    species: Optional[str] = Field(
        default=None,
        description="Tree species name to lookup (e.g., 'Acer platanoides', 'Tilia'). If None, returns all available species.",
    )
    n_trees: int = Field(
        default=1,
        description="Number of trees to calculate total annual carbon sequestration.",
    )
    maturity: str = Field(
        default="medio",
        description="Tree maturity stage: 'medio' (average/young) or 'maturita' (mature). Affects sequestration rate.",
    )


class CarbonSequestrationTool(BaseTool):
    """Tool to lookup annual carbon sequestration rates for tree species from Paoletti et al. dataset."""

    name: str = "lookup_carbon_sequestration"
    description: str = """
    Lookup annual carbon sequestration rates for tree species from Paoletti et al. research.

    Use this tool when the user asks about:
    - Annual carbon sequestration/storage for a specific species
    - How much carbon a tree absorbs per year
    - Carbon uptake rates (kg C/year)
    - "Stoccaggio annuale di carbonio" or "sequestro di carbonio annuo"
    - Comparing carbon sequestration between species
    - Total carbon sequestration for multiple trees

    Input:
    - species: name of the tree species (optional). If not provided, returns all available species.
    - n_trees: number of trees (default 1) to calculate total sequestration.
    - maturity: 'medio' for average/young trees, 'maturita' for mature trees.

    Returns:
    - c_medio_kg_year: average annual carbon sequestration (kg C/year)
    - c_maturita_kg_year: carbon sequestration at maturity (kg C/year)
    - total_c_kg_year: total carbon for n trees
    - co2_equivalent_kg_year: CO2 equivalent (C × 3.67)

    Example uses:
    - "Quanto carbonio sequestra un Acer platanoides all'anno?"
    - "Stoccaggio annuale di carbonio per Tilia"
    - "Carbon sequestration for 10 Populus trees"
    - "Confronta il sequestro di carbonio tra le specie"
    """
    args_schema: Type[BaseModel] = CarbonSequestrationInput

    _csv_path: Path
    _species_data: List[Dict[str, Any]]

    def __init__(self, csv_path: Optional[Path] = None, **kwargs):
        super().__init__(**kwargs)

        if csv_path is None:
            csv_path = Path(__file__).parent.parent.parent / "dataset" / "c_sequestration.csv"

        object.__setattr__(self, "_csv_path", csv_path)
        object.__setattr__(self, "_species_data", self._load_data(csv_path))

    def _load_data(self, csv_path: Path) -> List[Dict[str, Any]]:
        """Load carbon sequestration data from CSV."""
        if not csv_path.exists():
            return []

        species_data = []
        with csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                species = row.get("Specie", "").strip()
                c_medio_str = row.get("C medio (kg/anno)", "").strip()
                c_maturita_str = row.get("C a maturità (kg/anno)", "").strip()

                if species and c_medio_str:
                    try:
                        c_medio = float(c_medio_str)
                        c_maturita = float(c_maturita_str) if c_maturita_str else c_medio

                        species_data.append({
                            "species": species,
                            "c_medio_kg_year": c_medio,
                            "c_maturita_kg_year": c_maturita,
                        })
                    except (ValueError, TypeError):
                        continue

        return species_data

    def _find_species(self, species_name: str) -> Optional[Dict[str, Any]]:
        """Find species data by name (case-insensitive, flexible matching)."""
        species_name_lower = species_name.lower().strip()

        # First try exact match
        for data in self._species_data:
            if data["species"].lower() == species_name_lower:
                return data

        # Then try partial match
        for data in self._species_data:
            if species_name_lower in data["species"].lower() or data["species"].lower() in species_name_lower:
                return data

        # Try matching genus only (first word)
        genus = species_name_lower.split()[0] if species_name_lower else ""
        if genus:
            for data in self._species_data:
                if data["species"].lower().startswith(genus):
                    return data

        return None

    def _get_mean_values(self) -> Dict[str, float]:
        """Get mean values for unknown species."""
        for data in self._species_data:
            if data["species"].upper() == "MEDIA":
                return {
                    "c_medio_kg_year": data["c_medio_kg_year"],
                    "c_maturita_kg_year": data["c_maturita_kg_year"],
                }
        # Fallback if MEDIA row not found
        return {"c_medio_kg_year": DEFAULT_SEQUESTRATION_MEDIO_KG_YEAR, "c_maturita_kg_year": DEFAULT_SEQUESTRATION_MATURITA_KG_YEAR}

    def _run(
        self,
        species: Optional[str] = None,
        n_trees: int = 1,
        maturity: str = "medio",
    ) -> Dict[str, Any]:
        """Execute carbon sequestration lookup."""

        # If no species provided, return list of all available species
        if not species:
            available_species = [
                {
                    "species": item["species"],
                    "c_medio_kg_year": item["c_medio_kg_year"],
                    "c_maturita_kg_year": item["c_maturita_kg_year"],
                }
                for item in self._species_data
                if item["species"].upper() != "MEDIA"
            ]
            mean_values = self._get_mean_values()

            return {
                "available_species": available_species,
                "total_count": len(available_species),
                "mean_values": mean_values,
                "dataset_source": "Paoletti et al.",
                "note": "Valori in kg C/anno. Per CO2 equivalente, moltiplicare per 3.67.",
                "usage": "Specifica una specie per ottenere i valori dettagliati di sequestro.",
            }

        # Look up specific species
        species_data = self._find_species(species)
        use_mean = False

        if not species_data:
            # Species not found - use mean values
            mean_values = self._get_mean_values()
            species_data = {
                "species": species,
                "c_medio_kg_year": mean_values["c_medio_kg_year"],
                "c_maturita_kg_year": mean_values["c_maturita_kg_year"],
            }
            use_mean = True

        # Calculate based on maturity stage
        if maturity.lower() in ["maturita", "maturità", "mature", "maturo"]:
            rate = species_data["c_maturita_kg_year"]
            stage = "maturità"
        else:
            rate = species_data["c_medio_kg_year"]
            stage = "medio"

        # Calculate totals
        total_c = rate * n_trees
        total_co2 = total_c * CO2_C_RATIO

        result = {
            "found": not use_mean,
            "species_name": species_data["species"],
            "maturity_stage": stage,
            "c_medio_kg_year": species_data["c_medio_kg_year"],
            "c_maturita_kg_year": species_data["c_maturita_kg_year"],
            "selected_rate_kg_year": rate,
            "n_trees": n_trees,
            "total_c_kg_year": round(total_c, 2),
            "total_co2_kg_year": round(total_co2, 2),
            "dataset_source": "Paoletti et al.",
        }

        if use_mean:
            result["note"] = f"Specie '{species}' non trovata. Utilizzati valori medi generali (MEDIA)."
            result["available_species"] = [
                item["species"] for item in self._species_data
                if item["species"].upper() != "MEDIA"
            ]
        else:
            result["interpretation"] = (
                f"Un albero di {species_data['species']} (stadio: {stage}) sequestra "
                f"{rate} kg C/anno, equivalenti a {round(rate * CO2_C_RATIO, 2)} kg CO2/anno."
            )
            if n_trees > 1:
                result["interpretation"] += (
                    f" Per {n_trees} alberi: {round(total_c, 2)} kg C/anno "
                    f"({round(total_co2, 2)} kg CO2/anno)."
                )

        return result
