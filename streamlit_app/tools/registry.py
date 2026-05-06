"""Tool Registry — central metadata for all agent tools.

Provides per-tool configuration (max calls, category, descriptions) in one place,
eliminating hardcoded per-tool thresholds scattered across loop detection code.

Tools self-describe their capabilities and limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional


@dataclass(frozen=True)
class ToolMetadata:
    """Metadata for a registered tool."""

    name: str
    category: str  # "co2", "dataset", "visualization", "biomass", "research", "utility"
    description: Dict[str, str] = field(default_factory=dict)  # {"it": "...", "en": "..."}
    max_calls_per_session: int = 5  # Unified threshold for loop detection


# Central registry of all tool metadata.
# Tool names must match the `name` attribute on each BaseTool subclass.
_REGISTRY: Dict[str, ToolMetadata] = {}


def _register(meta: ToolMetadata) -> None:
    _REGISTRY[meta.name] = meta


# ── CO2 & Carbon tools ─────────────────────────────────────────────
_register(ToolMetadata(
    name="calculate_co2_sequestration",
    category="co2",
    description={"it": "Calcolo CO2 per singolo albero", "en": "CO2 calculation for a single tree"},
    max_calls_per_session=5,
))
_register(ToolMetadata(
    name="calculate_co2_aggregate",
    category="co2",
    description={"it": "Stock CO2 aggregato per gruppo di alberi", "en": "Aggregate CO2 stock for tree groups"},
    max_calls_per_session=4,
))
_register(ToolMetadata(
    name="lookup_carbon_content",
    category="co2",
    description={"it": "Frazione di carbonio per specie", "en": "Carbon fraction per species"},
    max_calls_per_session=5,
))
_register(ToolMetadata(
    name="lookup_carbon_sequestration",
    category="co2",
    description={"it": "Tasso annuale sequestro carbonio", "en": "Annual carbon sequestration rate"},
    max_calls_per_session=5,
))
_register(ToolMetadata(
    name="project_carbon_sequestration",
    category="co2",
    description={"it": "Proiezione sequestro futuro", "en": "Future sequestration projection"},
    max_calls_per_session=5,
))
_register(ToolMetadata(
    name="calculate_environmental_estimates",
    category="co2",
    description={"it": "Stime ambientali (volume, biomassa, carbonio)", "en": "Environmental estimates (volume, biomass, carbon)"},
    max_calls_per_session=5,
))

# ── Dataset tools ──────────────────────────────────────────────────
_register(ToolMetadata(
    name="query_tree_dataset",
    category="dataset",
    description={"it": "Query dataset alberi", "en": "Query tree dataset"},
    max_calls_per_session=4,
))
_register(ToolMetadata(
    name="query_species_list",
    category="dataset",
    description={"it": "Query lista specie botaniche", "en": "Query botanical species list"},
    max_calls_per_session=5,
))

# ── Visualization tools ───────────────────────────────────────────
_register(ToolMetadata(
    name="generate_chart",
    category="visualization",
    description={"it": "Genera grafici interattivi", "en": "Generate interactive charts"},
    max_calls_per_session=3,
))
_register(ToolMetadata(
    name="generate_map",
    category="visualization",
    description={"it": "Genera mappe interattive", "en": "Generate interactive maps"},
    max_calls_per_session=3,
))

# ── Biomass & Volume tools ─────────────────────────────────────────
_register(ToolMetadata(
    name="calculate_heyer_volume",
    category="biomass",
    description={"it": "Volume con formula di Heyer", "en": "Heyer volume formula"},
))
_register(ToolMetadata(
    name="calculate_general_volume",
    category="biomass",
    description={"it": "Volume allometrico generalizzato", "en": "General allometric volume"},
))
_register(ToolMetadata(
    name="calculate_simplified_volume",
    category="biomass",
    description={"it": "Volume semplificato", "en": "Simplified volume"},
))
_register(ToolMetadata(
    name="calculate_allometric_relation",
    category="biomass",
    description={"it": "Relazione allometrica Y=aX^b", "en": "Allometric relation Y=aX^b"},
))
_register(ToolMetadata(
    name="calculate_log_allometric",
    category="biomass",
    description={"it": "Allometria logaritmica", "en": "Log allometric"},
))
_register(ToolMetadata(
    name="calculate_model_error",
    category="biomass",
    description={"it": "Errore del modello", "en": "Model error"},
))
_register(ToolMetadata(
    name="calculate_log_fuel_biomass",
    category="biomass",
    description={"it": "Biomassa combustibile logaritmica", "en": "Log fuel biomass"},
))
_register(ToolMetadata(
    name="calculate_leaf_biomass",
    category="biomass",
    description={"it": "Biomassa fogliare", "en": "Leaf biomass"},
))
_register(ToolMetadata(
    name="calculate_stem_biomass",
    category="biomass",
    description={"it": "Biomassa del fusto", "en": "Stem biomass"},
))
_register(ToolMetadata(
    name="calculate_root_biomass",
    category="biomass",
    description={"it": "Biomassa radicale", "en": "Root biomass"},
))
_register(ToolMetadata(
    name="calculate_total_biomass",
    category="biomass",
    description={"it": "Biomassa totale", "en": "Total biomass"},
))

# ── Research tools ─────────────────────────────────────────────────
_register(ToolMetadata(
    name="search_scientific_papers",
    category="research",
    description={"it": "Cerca paper scientifici", "en": "Search scientific papers"},
    max_calls_per_session=5,
))

# ── Utility tools ──────────────────────────────────────────────────
_register(ToolMetadata(
    name="export_data",
    category="utility",
    description={"it": "Esporta dati", "en": "Export data"},
    max_calls_per_session=3,
))
_register(ToolMetadata(
    name="detect_language",
    category="utility",
    description={"it": "Rileva lingua", "en": "Detect language"},
))
_register(ToolMetadata(
    name="translate_text",
    category="utility",
    description={"it": "Traduci testo", "en": "Translate text"},
))


class ToolRegistry:
    """Read-only access to tool metadata."""

    @staticmethod
    def get(tool_name: str) -> Optional[ToolMetadata]:
        """Get metadata for a tool by name. Returns None if not registered."""
        return _REGISTRY.get(tool_name)

    @staticmethod
    def get_max_calls(tool_name: str, default: int = 5) -> int:
        """Get the max_calls_per_session for a tool, with a fallback default."""
        meta = _REGISTRY.get(tool_name)
        return meta.max_calls_per_session if meta else default

    @staticmethod
    def all_tools() -> Dict[str, ToolMetadata]:
        """Get all registered tool metadata."""
        return dict(_REGISTRY)

    @staticmethod
    def by_category(category: str) -> List[ToolMetadata]:
        """Get all tools in a category."""
        return [m for m in _REGISTRY.values() if m.category == category]

    @staticmethod
    def get_summary(language: Literal["it", "en"] = "it") -> str:
        """Generate a tools summary string for prompt injection."""
        lines: list[str] = []
        current_cat = ""
        for meta in _REGISTRY.values():
            if meta.category != current_cat:
                current_cat = meta.category
                lines.append(f"\n### {current_cat.title()}")
            desc = meta.description.get(language, meta.description.get("en", meta.name))
            lines.append(f"- **{meta.name}**: {desc}")
        return "\n".join(lines)
