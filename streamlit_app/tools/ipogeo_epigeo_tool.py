from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, field_validator


class IpogeoEpigeoInput(BaseModel):
    """Input schema per il calcolo biomassa epigea/ipogea."""

    species_group: str = Field(
        description="Gruppo della specie: hardwood/latifoglie oppure softwood/conifere"
    )
    above_ground_biomass_kg: Optional[float] = Field(
        default=None,
        description="Biomassa epigea (AGB) in kg. Se omessa, viene restituito solo il rapporto radici/chioma",
        gt=0,
    )

    @field_validator("species_group")
    @classmethod
    def normalize_group(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized in {"hardwood", "latifoglie"}:
            return "hardwood"
        if normalized in {"softwood", "conifere"}:
            return "softwood"
        raise ValueError("Use hardwood/latifoglie oppure softwood/conifere")


class IpogeoEpigeoDataset:
    """Gestisce il caricamento e l'accesso ai valori del dataset ipogeo/epigeo."""

    def __init__(self, dataset_path: Path):
        self._dataset_path = dataset_path
        self._data: Dict[str, Dict[str, float]] = self._load()

    @property
    def dataset_path(self) -> Path:
        return self._dataset_path

    def _load(self) -> Dict[str, Dict[str, float]]:
        if not self._dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset non trovato in {self._dataset_path}. "
                "Assicurati che ipogeo_epigeo.csv sia presente."
            )

        lines = self._dataset_path.read_text(encoding="utf-8").splitlines()
        # Le prime tre righe contengono l'header spezzato, i dati iniziano dalla quarta
        records = lines[3:]

        data: Dict[str, Dict[str, float]] = {}
        for raw in records:
            parts = [part.strip().strip('"') for part in raw.split(",")]
            if len(parts) < 3:
                continue
            component, hardwood_val, softwood_val = parts[0], parts[1], parts[2]
            try:
                data[component] = {
                    "hardwood": float(hardwood_val),
                    "softwood": float(softwood_val),
                }
            except ValueError:
                continue

        required_keys = {"Above-ground biomass", "Roots"}
        if not required_keys.issubset(data.keys()):
            missing = ", ".join(sorted(required_keys - set(data.keys())))
            raise ValueError(
                f"Dati mancanti nel dataset ipogeo_epigeo.csv: {missing}"
            )

        return data

    def get_ratio(self, group: Literal["hardwood", "softwood"]) -> float:
        agb = self._data["Above-ground biomass"][group]
        roots = self._data["Roots"][group]
        return roots / agb

    def get_components(self, group: Literal["hardwood", "softwood"]) -> Dict[str, float]:
        return {component: values[group] for component, values in self._data.items()}


class IpogeoEpigeoTool(BaseTool):
    """Tool per calcolare biomassa ipogea/epigea usando il dataset ipogeo_epigeo.csv."""

    name: str = "calculate_ipogeo_epigeo_biomass"
    description: str = """
    Calcola biomassa ipogea (BGB) da biomassa epigea (AGB) usando i rapporti root/shot
    forniti nel dataset ipogeo_epigeo.csv. Funziona solo per Hardwood/Latifoglie
    oppure Softwood/Conifere: se il gruppo non appartiene a questi insiemi il calcolo
    non è disponibile.
    """
    args_schema: Type[BaseModel] = IpogeoEpigeoInput

    _dataset: IpogeoEpigeoDataset

    def __init__(self, dataset_path: Optional[Path] = None, **kwargs):
        super().__init__(**kwargs)
        if dataset_path is None:
            dataset_path = Path(__file__).parent.parent / "dataset" / "ipogeo_epigeo.csv"
        object.__setattr__(self, "_dataset", IpogeoEpigeoDataset(dataset_path))

    def _run(self, species_group: str, above_ground_biomass_kg: Optional[float] = None) -> dict:
        group = species_group  # già normalizzato dal validator
        ratio = self._dataset.get_ratio(group)  # BGB / AGB

        result = {
            "group": group,
            "root_to_shoot_ratio": ratio,
            "dataset_source": str(self._dataset.dataset_path),
            "components_percent": self._dataset.get_components(group),
            "note": (
                "Calcolo disponibile solo per hardwood/latifoglie e softwood/conifere. "
                "I valori derivano dal dataset ipogeo_epigeo.csv."
            ),
        }

        if above_ground_biomass_kg is not None:
            below_ground = above_ground_biomass_kg * ratio
            result.update(
                {
                    "above_ground_biomass_kg": above_ground_biomass_kg,
                    "below_ground_biomass_kg": below_ground,
                    "total_biomass_kg": above_ground_biomass_kg + below_ground,
                    "method": "BGB = AGB * (roots / above-ground biomass)",
                }
            )

        return result

