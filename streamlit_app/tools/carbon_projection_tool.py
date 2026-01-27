"""Carbon Projection Tool - Future sequestration trends and woody biomass conversion."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class CarbonProjectionInput(BaseModel):
    """Input schema for carbon projection tool."""

    # Biomass conversion inputs
    biomass_kg: Optional[float] = Field(
        default=None,
        description="Total woody biomass in kg (dry weight). Use this OR volume_m3.",
    )
    volume_m3: Optional[float] = Field(
        default=None,
        description="Wood volume in cubic meters. Use this OR biomass_kg.",
    )
    wood_density_kg_m3: float = Field(
        default=600.0,
        description="Wood density in kg/m³ (default 600). Used only when volume_m3 is provided.",
    )
    carbon_fraction: float = Field(
        default=0.47,
        description="Carbon fraction of dry biomass (default 0.47 = 47%).",
    )

    # Projection inputs
    species: Optional[str] = Field(
        default=None,
        description="Tree species for growth rate lookup (e.g., 'Acer platanoides', 'Tilia').",
    )
    current_age_years: int = Field(
        default=20,
        description="Current estimated age of the tree in years.",
    )
    projection_years: int = Field(
        default=30,
        description="Number of years to project into the future (default 30).",
    )
    n_trees: int = Field(
        default=1,
        description="Number of trees for aggregate calculation.",
    )
    maturity_age_years: int = Field(
        default=50,
        description="Age at which tree reaches maturity (affects sequestration rate).",
    )


class CarbonProjectionTool(BaseTool):
    """Tool to convert woody biomass to carbon and project future sequestration trends."""

    name: str = "project_carbon_sequestration"
    description: str = """
    Convert woody biomass to carbon and project future carbon sequestration trends.

    Use this tool when the user asks about:
    - Converting wood volume or biomass to carbon/CO2
    - Future carbon sequestration projections
    - Carbon stock trends over time
    - "Quanto carbonio avrà tra X anni"
    - "Proiezione del sequestro futuro"
    - "Trend di sequestro"
    - "Capitale legnoso in carbonio"
    - Dynamic carbon sequestration modeling

    Input options:
    1. Biomass conversion: provide biomass_kg OR (volume_m3 + wood_density)
    2. Future projection: provide species, current_age_years, projection_years

    Returns:
    - current_carbon_t: current carbon stock in tonnes
    - current_co2_t: current CO2 equivalent in tonnes
    - yearly_projection: list of {year, carbon_t, co2_t, cumulative_sequestration_t}
    - total_future_sequestration_t: total additional carbon over projection period
    - final_carbon_stock_t: projected carbon stock at end of projection

    Example uses:
    - "Converti 500 kg di biomassa in carbonio"
    - "Proiezione a 30 anni per un Tilia di 20 anni"
    - "Trend di sequestro futuro per 10 aceri"
    """
    args_schema: Type[BaseModel] = CarbonProjectionInput

    _csv_path: Path
    _species_rates: Dict[str, Dict[str, float]]

    def __init__(self, csv_path: Optional[Path] = None, **kwargs):
        super().__init__(**kwargs)

        if csv_path is None:
            csv_path = Path(__file__).parent.parent.parent / "dataset" / "c_sequestration.csv"

        object.__setattr__(self, "_csv_path", csv_path)
        object.__setattr__(self, "_species_rates", self._load_rates(csv_path))

    def _load_rates(self, csv_path: Path) -> Dict[str, Dict[str, float]]:
        """Load carbon sequestration rates from CSV."""
        rates = {}
        if not csv_path.exists():
            return rates

        with csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                species = row.get("Specie", "").strip()
                c_medio_str = row.get("C medio (kg/anno)", "").strip()
                c_maturita_str = row.get("C a maturità (kg/anno)", "").strip()

                if species and c_medio_str:
                    try:
                        rates[species.lower()] = {
                            "c_medio": float(c_medio_str),
                            "c_maturita": float(c_maturita_str) if c_maturita_str else float(c_medio_str),
                        }
                    except (ValueError, TypeError):
                        continue

        return rates

    def _get_species_rate(self, species: Optional[str], is_mature: bool) -> float:
        """Get sequestration rate for species, or default if not found."""
        default_medio = 4.6  # kg C/year
        default_maturita = 11.4  # kg C/year

        if not species:
            return default_maturita if is_mature else default_medio

        species_lower = species.lower().strip()

        # Try exact match
        if species_lower in self._species_rates:
            rates = self._species_rates[species_lower]
            return rates["c_maturita"] if is_mature else rates["c_medio"]

        # Try partial match
        for key, rates in self._species_rates.items():
            if species_lower in key or key in species_lower:
                return rates["c_maturita"] if is_mature else rates["c_medio"]

        # Try genus match
        genus = species_lower.split()[0] if species_lower else ""
        if genus:
            for key, rates in self._species_rates.items():
                if key.startswith(genus):
                    return rates["c_maturita"] if is_mature else rates["c_medio"]

        return default_maturita if is_mature else default_medio

    def _biomass_to_carbon(
        self,
        biomass_kg: Optional[float],
        volume_m3: Optional[float],
        wood_density_kg_m3: float,
        carbon_fraction: float,
    ) -> Dict[str, float]:
        """Convert biomass or volume to carbon and CO2."""
        # Calculate biomass if volume provided
        if biomass_kg is None and volume_m3 is not None:
            biomass_kg = volume_m3 * wood_density_kg_m3

        if biomass_kg is None:
            return {"biomass_kg": 0, "carbon_kg": 0, "carbon_t": 0, "co2_kg": 0, "co2_t": 0}

        carbon_kg = biomass_kg * carbon_fraction
        co2_kg = carbon_kg * 3.67  # CO2/C molecular weight ratio

        return {
            "biomass_kg": round(biomass_kg, 2),
            "carbon_kg": round(carbon_kg, 2),
            "carbon_t": round(carbon_kg / 1000, 4),
            "co2_kg": round(co2_kg, 2),
            "co2_t": round(co2_kg / 1000, 4),
        }

    def _project_sequestration(
        self,
        species: Optional[str],
        current_age_years: int,
        projection_years: int,
        n_trees: int,
        maturity_age_years: int,
        current_carbon_kg: float,
    ) -> Dict[str, Any]:
        """Project future carbon sequestration year by year."""
        yearly_data = []
        cumulative_c = 0.0

        for year_offset in range(1, projection_years + 1):
            age = current_age_years + year_offset
            is_mature = age >= maturity_age_years

            # Get rate for this year
            rate_per_tree = self._get_species_rate(species, is_mature)
            annual_c_kg = rate_per_tree * n_trees
            cumulative_c += annual_c_kg

            yearly_data.append({
                "year": year_offset,
                "age": age,
                "is_mature": is_mature,
                "annual_c_kg": round(annual_c_kg, 2),
                "annual_co2_kg": round(annual_c_kg * 3.67, 2),
                "cumulative_c_kg": round(cumulative_c, 2),
                "cumulative_co2_kg": round(cumulative_c * 3.67, 2),
                "total_c_stock_kg": round(current_carbon_kg + cumulative_c, 2),
            })

        # Summary statistics
        total_sequestration_kg = cumulative_c
        final_stock_kg = current_carbon_kg + cumulative_c

        # Decade summaries
        decades = {}
        for i in range(0, projection_years, 10):
            decade_start = i + 1
            decade_end = min(i + 10, projection_years)
            decade_data = [d for d in yearly_data if decade_start <= d["year"] <= decade_end]
            if decade_data:
                decades[f"anni_{decade_start}-{decade_end}"] = {
                    "sequestro_c_kg": round(sum(d["annual_c_kg"] for d in decade_data), 2),
                    "sequestro_co2_kg": round(sum(d["annual_co2_kg"] for d in decade_data), 2),
                }

        return {
            "yearly_projection": yearly_data,
            "decades_summary": decades,
            "total_sequestration_c_kg": round(total_sequestration_kg, 2),
            "total_sequestration_c_t": round(total_sequestration_kg / 1000, 4),
            "total_sequestration_co2_kg": round(total_sequestration_kg * 3.67, 2),
            "total_sequestration_co2_t": round(total_sequestration_kg * 3.67 / 1000, 4),
            "final_c_stock_kg": round(final_stock_kg, 2),
            "final_c_stock_t": round(final_stock_kg / 1000, 4),
            "final_co2_stock_t": round(final_stock_kg * 3.67 / 1000, 4),
        }

    def _run(
        self,
        biomass_kg: Optional[float] = None,
        volume_m3: Optional[float] = None,
        wood_density_kg_m3: float = 600.0,
        carbon_fraction: float = 0.47,
        species: Optional[str] = None,
        current_age_years: int = 20,
        projection_years: int = 30,
        n_trees: int = 1,
        maturity_age_years: int = 50,
    ) -> Dict[str, Any]:
        """Execute carbon projection calculation."""

        # Step 1: Convert current biomass to carbon
        current_stock = self._biomass_to_carbon(
            biomass_kg, volume_m3, wood_density_kg_m3, carbon_fraction
        )

        # Step 2: Project future sequestration
        projection = self._project_sequestration(
            species=species,
            current_age_years=current_age_years,
            projection_years=projection_years,
            n_trees=n_trees,
            maturity_age_years=maturity_age_years,
            current_carbon_kg=current_stock["carbon_kg"],
        )

        # Get rate info for context
        is_currently_mature = current_age_years >= maturity_age_years
        current_rate = self._get_species_rate(species, is_currently_mature)
        mature_rate = self._get_species_rate(species, True)
        young_rate = self._get_species_rate(species, False)

        result = {
            "input_summary": {
                "biomass_kg": biomass_kg,
                "volume_m3": volume_m3,
                "wood_density_kg_m3": wood_density_kg_m3,
                "carbon_fraction": carbon_fraction,
                "species": species or "generico (valori medi)",
                "current_age_years": current_age_years,
                "projection_years": projection_years,
                "n_trees": n_trees,
                "maturity_age_years": maturity_age_years,
            },
            "current_stock": current_stock,
            "sequestration_rates": {
                "giovane_medio_kg_year": young_rate,
                "maturo_kg_year": mature_rate,
                "current_rate_kg_year": current_rate,
                "is_currently_mature": is_currently_mature,
            },
            "projection": projection,
            "interpretation": self._generate_interpretation(
                species, n_trees, current_stock, projection, projection_years
            ),
            "methodology": {
                "biomass_to_carbon": "C = Biomassa × CF (carbon fraction)",
                "carbon_to_co2": "CO2 = C × 3.67 (rapporto pesi molecolari)",
                "projection_model": "Modello dinamico con tasso variabile in base all'età",
                "data_source": "Paoletti et al. per tassi di sequestro per specie",
            },
        }

        return result

    def _generate_interpretation(
        self,
        species: Optional[str],
        n_trees: int,
        current_stock: Dict[str, float],
        projection: Dict[str, Any],
        projection_years: int,
    ) -> str:
        """Generate natural language interpretation of results."""
        species_name = species or "specie generica"
        tree_word = "albero" if n_trees == 1 else f"{n_trees} alberi"

        parts = []

        if current_stock["carbon_kg"] > 0:
            parts.append(
                f"Lo stock attuale di {tree_word} ({species_name}) è di "
                f"{current_stock['carbon_t']:.3f} t C ({current_stock['co2_t']:.3f} t CO2)."
            )

        parts.append(
            f"Nei prossimi {projection_years} anni, si prevede un sequestro aggiuntivo di "
            f"{projection['total_sequestration_c_t']:.3f} t C "
            f"({projection['total_sequestration_co2_t']:.3f} t CO2)."
        )

        parts.append(
            f"Lo stock finale proiettato sarà di {projection['final_c_stock_t']:.3f} t C "
            f"({projection['final_co2_stock_t']:.3f} t CO2)."
        )

        return " ".join(parts)
