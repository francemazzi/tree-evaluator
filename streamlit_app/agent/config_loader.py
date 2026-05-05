"""Dynamic configuration loader for the Tree Evaluator Agent.

This module loads configuration from JSON files to avoid hardcoding:
- Dataset presets (Vienna, Milano, etc.)
- Static tools metadata (names, descriptions, keywords)
- Simple query patterns for optimization
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolMatch:
    """Scored keyword match for a tool candidate."""

    name: str
    score: float
    matched_keywords: List[str]
    reason: str


class ConfigLoader:
    """Loads agent configuration dynamically from JSON files."""

    _instance: Optional["ConfigLoader"] = None
    _presets_data: Optional[Dict[str, Any]] = None
    _GENERIC_KEYWORDS = {
        "alberi", "trees", "specie", "species", "query", "cerca", "search",
        "trova", "find", "totale", "total", "tutti", "all", "media",
        "average", "carbonio", "carbon", "biomassa", "biomass", "volume",
    }

    def __new__(cls) -> "ConfigLoader":
        """Singleton pattern to avoid reloading config multiple times."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the config loader."""
        if self._presets_data is None:
            self._load_presets()

    def _load_presets(self) -> None:
        """Load presets from JSON file."""
        presets_path = Path(__file__).parent.parent.parent / "dataset" / "presets.json"
        try:
            with open(presets_path, "r", encoding="utf-8") as f:
                self._presets_data = json.load(f)
        except FileNotFoundError:
            logger.warning(f"Presets file not found: {presets_path}")
            self._presets_data = {}
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in presets file: {e}")
            self._presets_data = {}

    def get_dataset_presets(self) -> Dict[str, Dict[str, Any]]:
        """Get all dataset presets (excluding internal keys starting with _).

        Returns:
            Dict of preset_name -> preset_config
        """
        if not self._presets_data:
            return {}

        return {
            k: v for k, v in self._presets_data.items()
            if not k.startswith("_")
        }

    def get_dataset_preset(self, name: str, language: str = "it") -> Optional[Dict[str, Any]]:
        """Get a specific dataset preset with localized description.

        Args:
            name: Preset name (e.g., "vienna", "milano")
            language: Language for description ("it" or "en")

        Returns:
            Preset config dict or None if not found
        """
        presets = self.get_dataset_presets()
        preset = presets.get(name)

        if not preset:
            return None

        # Resolve localized description
        result = dict(preset)
        if isinstance(result.get("description"), dict):
            result["description"] = result["description"].get(language, result["description"].get("it", ""))

        return result

    def get_static_tools_metadata(self, language: str = "it") -> List[Dict[str, Any]]:
        """Get static tools metadata with localized descriptions.

        Args:
            language: Language for description ("it" or "en")

        Returns:
            List of tool metadata dicts with name, description, keywords
        """
        if not self._presets_data:
            return []

        tools = self._presets_data.get("_static_tools", [])
        result = []

        for tool in tools:
            tool_data = {
                "name": tool.get("name", ""),
                "keywords": tool.get("keywords", []),
            }

            # Resolve localized description
            desc = tool.get("description", {})
            if isinstance(desc, dict):
                tool_data["description"] = desc.get(language, desc.get("it", ""))
            else:
                tool_data["description"] = desc

            result.append(tool_data)

        return result

    def get_simple_query_patterns(self, language: str = "it") -> List[str]:
        """Get simple query patterns for a language.

        Args:
            language: Language code ("it" or "en")

        Returns:
            List of pattern strings
        """
        if not self._presets_data:
            return []

        patterns = self._presets_data.get("_simple_query_patterns", {})
        return patterns.get(language, patterns.get("it", []))

    def get_all_keywords(self, language: str = "it") -> Dict[str, List[str]]:
        """Get all keywords mapped to tool names.

        Args:
            language: Language for filtering (not currently used, keywords are multilingual)

        Returns:
            Dict mapping tool_name -> list of keywords
        """
        tools = self.get_static_tools_metadata(language)
        return {tool["name"]: tool["keywords"] for tool in tools if tool.get("keywords")}

    def match_tools_with_scores(
        self, query: str, max_tools: Optional[int] = 3
    ) -> List[Dict[str, Any]]:
        """Match query keywords to ranked tool candidates.

        Args:
            query: User query string
            max_tools: Maximum number of candidates to return. Use None for all.

        Returns:
            List of dicts with name, score, matched_keywords, and reason.
        """
        query_lower = query.lower()
        scored_matches: List[ToolMatch] = []

        for tool in self.get_static_tools_metadata():
            tool_name = tool.get("name", "")
            keywords = tool.get("keywords", [])
            matched_keywords: List[str] = []
            score = 0.0

            for keyword in keywords:
                if self._keyword_matches(query_lower, keyword):
                    matched_keywords.append(keyword)
                    score += self._keyword_score(keyword)

            score += self._intent_score(query_lower, tool_name)

            if score > 0:
                scored_matches.append(
                    ToolMatch(
                        name=tool_name,
                        score=score,
                        matched_keywords=matched_keywords,
                        reason=self._match_reason(matched_keywords, score),
                    )
                )

        ranked = sorted(
            scored_matches,
            key=lambda m: (-m.score, self._tool_order_index(m.name), m.name),
        )
        if ranked:
            top_score = ranked[0].score
            ranked = [m for m in ranked if m.score >= max(1.5, top_score * 0.45)]
        if max_tools is not None:
            ranked = ranked[:max_tools]

        return [
            {
                "name": match.name,
                "score": round(match.score, 3),
                "matched_keywords": match.matched_keywords,
                "reason": match.reason,
            }
            for match in ranked
        ]

    def match_keywords_to_tools(self, query: str, max_tools: Optional[int] = 3) -> List[str]:
        """Match query keywords to ranked tool names.

        The matcher combines keyword hits with intent-specific boosts so that
        generic words such as "alberi" or "specie" do not overshadow explicit
        requests for maps, charts, taxonomy lookups, or CO2 calculations.
        """
        return [
            match["name"]
            for match in self.match_tools_with_scores(query, max_tools=max_tools)
        ]

    def _keyword_matches(self, query_lower: str, keyword: str) -> bool:
        keyword_lower = keyword.lower().strip()
        if not keyword_lower:
            return False
        if " " in keyword_lower:
            return keyword_lower in query_lower
        return re.search(rf"(?<!\w){re.escape(keyword_lower)}(?!\w)", query_lower) is not None

    def _keyword_score(self, keyword: str) -> float:
        keyword_lower = keyword.lower().strip()
        if keyword_lower in self._GENERIC_KEYWORDS:
            return 1.0
        if len(keyword_lower) <= 3:
            return 1.5
        return 2.5

    def _intent_score(self, query_lower: str, tool_name: str) -> float:
        score = 0.0

        map_intent = self._has_any(
            query_lower, ["mappa", "map", "gps", "coordinate", "heatmap", "markers"]
        )
        chart_intent = self._has_any(
            query_lower,
            ["grafico", "chart", "plot", "barre", "torta", "istogramma", "visualizza"],
        )
        export_intent = self._has_any(
            query_lower, ["esporta", "export", "csv", "excel", "xlsx", "scarica", "download"]
        )
        paper_intent = self._has_any(
            query_lower, ["paper", "articoli", "pubmed", "arxiv", "ricerca scientifica"]
        )
        carbon_content_intent = self._has_any(
            query_lower, ["contenuto", "content", "frazione", "fraction"]
        ) and self._has_any(query_lower, ["carbonio", "carbon"])
        taxonomy_intent = self._has_any(
            query_lower,
            [
                "tassonomia", "taxonomy", "famiglia", "family", "ordine", "order",
                "tratti", "traits", "leaf_type", "growth",
            ],
        )
        dataset_intent = self._has_any(
            query_lower,
            [
                "dataset", "quanti", "quante", "count", "statistiche", "statistics",
                "top", "piu comune", "più comune", "distretto", "municipio",
            ],
        )
        co2_intent = self._has_any(query_lower, ["co2", "sequestro", "sequestration"])
        aggregate_intent = self._has_any(
            query_lower,
            ["totale", "total", "media", "average", "aggregato", "aggregate", "tutti", "all", "gruppo", "group"],
        ) or (co2_intent and self._has_any(query_lower, ["alberi", "trees"]))
        single_tree_intent = self._has_any(
            query_lower,
            ["singolo", "single", "un albero", "una pianta", "dbh", "diametro", "diameter", "altezza", "height"],
        )
        environment_intent = self._has_any(
            query_lower,
            ["ambiente", "environment", "environmental", "stima ambientale", "estimates"],
        )
        biomass_intent = self._has_any(query_lower, ["biomassa", "biomass"])
        specific_formula_intent = self._has_any(
            query_lower,
            [
                "heyer", "semplificato", "simplified", "generale", "general",
                "allometria", "allometric", "logaritmo", "log", "errore", "error",
                "fogliare", "foglie", "leaf", "fusto", "tronco", "stem", "trunk",
                "radicale", "radici", "root",
            ],
        )

        if tool_name == "generate_map" and map_intent:
            score += 8.0
        if tool_name == "generate_chart" and chart_intent:
            score += 8.0
        if tool_name == "export_data" and export_intent:
            score += 8.0
        if tool_name == "search_scientific_papers" and paper_intent:
            score += 8.0

        if tool_name == "query_tree_dataset":
            if dataset_intent:
                score += 5.0
            if map_intent or chart_intent or taxonomy_intent or carbon_content_intent:
                score -= 3.0

        if tool_name == "query_species_list":
            if taxonomy_intent:
                score += 8.0
            elif self._has_any(query_lower, ["specie", "species"]) and not dataset_intent:
                score += 2.0

        if tool_name == "lookup_carbon_content" and carbon_content_intent:
            score += 9.0

        if tool_name == "calculate_co2_aggregate" and co2_intent:
            score += 4.0
            if aggregate_intent:
                score += 7.0
            if single_tree_intent and not aggregate_intent:
                score -= 3.0

        if tool_name == "calculate_co2_sequestration" and co2_intent:
            score += 4.0
            if single_tree_intent:
                score += 7.0
            if aggregate_intent:
                score -= 3.0

        if tool_name == "calculate_environmental_estimates":
            if environment_intent:
                score += 8.0
            elif (
                not specific_formula_intent
                and self._has_any(query_lower, ["volume", "biomassa", "biomass"])
                and self._has_any(query_lower, ["stima", "estimate", "calcola", "calculate"])
            ):
                score += 3.0

        if tool_name == "calculate_leaf_biomass" and biomass_intent and self._has_any(
            query_lower, ["foglie", "fogliare", "leaf", "chioma", "canopy"]
        ):
            score += 8.0
        if tool_name == "calculate_stem_biomass" and biomass_intent and self._has_any(
            query_lower, ["fusto", "tronco", "stem", "trunk"]
        ):
            score += 8.0
        if tool_name == "calculate_root_biomass" and biomass_intent and self._has_any(
            query_lower, ["radici", "radicale", "root", "sotterraneo"]
        ):
            score += 8.0
        if tool_name == "calculate_total_biomass" and biomass_intent and self._has_any(
            query_lower, ["totale", "total"]
        ):
            score += 8.0

        return score

    @staticmethod
    def _has_any(query_lower: str, terms: List[str]) -> bool:
        return any(term in query_lower for term in terms)

    def _tool_order_index(self, tool_name: str) -> int:
        for index, tool in enumerate(self.get_static_tools_metadata()):
            if tool.get("name") == tool_name:
                return index
        return 999

    @staticmethod
    def _match_reason(matched_keywords: List[str], score: float) -> str:
        if matched_keywords:
            return f"keywords: {', '.join(matched_keywords)}; score={score:.1f}"
        return f"intent score={score:.1f}"

    def build_tools_summary(self, language: str = "it") -> str:
        """Build a summary of all static tools for planning.

        Args:
            language: Language for descriptions

        Returns:
            Formatted string with tool summaries
        """
        tools = self.get_static_tools_metadata(language)
        lines = []

        for tool in tools:
            lines.append(f"- {tool['name']}: {tool['description']}")

        return "\n".join(lines)


# Global instance for easy access
_config_loader: Optional[ConfigLoader] = None


def get_config_loader() -> ConfigLoader:
    """Get the global config loader instance.

    Returns:
        ConfigLoader singleton instance
    """
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader


def get_dataset_presets() -> Dict[str, Dict[str, Any]]:
    """Convenience function to get dataset presets."""
    return get_config_loader().get_dataset_presets()


def get_static_tools_metadata(language: str = "it") -> List[Dict[str, Any]]:
    """Convenience function to get static tools metadata."""
    return get_config_loader().get_static_tools_metadata(language)


def match_query_to_tools(query: str) -> List[str]:
    """Convenience function to match query to tools."""
    return get_config_loader().match_keywords_to_tools(query)


def match_query_to_tools_with_scores(query: str, max_tools: Optional[int] = 3) -> List[Dict[str, Any]]:
    """Convenience function to match query to ranked tool candidates."""
    return get_config_loader().match_tools_with_scores(query, max_tools=max_tools)
