"""Dynamic configuration loader for the Tree Evaluator Agent.

This module loads configuration from JSON files to avoid hardcoding:
- Dataset presets (Vienna, Milano, etc.)
- Static tools metadata (names, descriptions, keywords)
- Simple query patterns for optimization
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Loads agent configuration dynamically from JSON files."""

    _instance: Optional["ConfigLoader"] = None
    _presets_data: Optional[Dict[str, Any]] = None

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

    def match_keywords_to_tools(self, query: str) -> List[str]:
        """Match query keywords to tool names.

        Args:
            query: User query string

        Returns:
            List of matching tool names
        """
        query_lower = query.lower()
        matching_tools = []

        for tool in self.get_static_tools_metadata():
            keywords = tool.get("keywords", [])
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    if tool["name"] not in matching_tools:
                        matching_tools.append(tool["name"])
                    break

        return matching_tools

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
