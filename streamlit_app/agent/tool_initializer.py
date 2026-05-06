"""Tool initialization for Tree Evaluator Agent.

This module handles the setup and configuration of all tools used by the agent,
including static tools and dynamically loaded tools from JSON configuration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, TYPE_CHECKING

from streamlit_app.agent.state import DATASET_PRESETS
from streamlit_app.tools.allometric_relation_tool import AllometricRelationTool
from streamlit_app.tools.carbon_content_tool import CarbonContentTool
from streamlit_app.tools.chart_tool import ChartGenerationTool
from streamlit_app.tools.co2_aggregate_tool import CO2AggregateTool
from streamlit_app.tools.co2_tool import CO2CalculationTool
from streamlit_app.tools.dataset_tool import DatasetQueryTool
from streamlit_app.tools.dynamic_tool_loader import DynamicToolLoader
from streamlit_app.tools.environment_tool import EnvironmentEstimationTool
from streamlit_app.tools.export_tool import ExportDataTool
from streamlit_app.tools.general_volume_tool import GeneralVolumeTool
from streamlit_app.tools.heyer_volume_tool import HeyerVolumeTool
from streamlit_app.tools.language_tool import LanguageDetectionTool, LanguageTranslationTool
from streamlit_app.tools.leaf_biomass_tool import LeafBiomassTool
from streamlit_app.tools.log_allometric_tool import LogAllometricTool
from streamlit_app.tools.log_fuel_biomass_tool import LogFuelBiomassTool
from streamlit_app.tools.map_tool import MapGenerationTool
from streamlit_app.tools.model_error_tool import ModelErrorTool
from streamlit_app.tools.paper_search_tool import PaperSearchTool
from streamlit_app.tools.root_biomass_tool import RootBiomassTool
from streamlit_app.tools.simplified_volume_tool import SimplifiedVolumeTool
from streamlit_app.tools.species_list_tool import SpeciesListQueryTool
from streamlit_app.tools.stem_biomass_tool import StemBiomassTool
from streamlit_app.tools.total_biomass_tool import TotalBiomassTool

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Get the project root directory.

    Returns:
        Path to project root (parent of streamlit_app)
    """
    return Path(__file__).parent.parent.parent


class ToolInitializer:
    """Handles initialization and configuration of agent tools."""

    def __init__(
        self,
        base_llm: "BaseChatModel",
        fallback_llm: "BaseChatModel",
        embeddings: "Embeddings",
        interface_language: str = "it",
    ) -> None:
        """Initialize the tool initializer.

        Args:
            base_llm: Primary LLM for tool operations
            fallback_llm: Fallback LLM for rate limit scenarios
            embeddings: Embeddings model for similarity operations
            interface_language: Language for tool responses ("it" or "en")
        """
        self._base_llm = base_llm
        self._fallback_llm = fallback_llm
        self._embeddings = embeddings
        self._interface_language = interface_language
        self._dynamic_tools_summary: List[dict] = []

    @property
    def dynamic_tools_summary(self) -> List[dict]:
        """Get the summary of dynamically loaded tools."""
        return self._dynamic_tools_summary

    def initialize_tools(
        self,
        custom_db_path: Optional[Path] = None,
        custom_table_name: Optional[str] = None,
        data_description: str = "",
        dataset_preset: str = "vienna",
        dataset_column_roles: Optional[dict] = None,
    ) -> List[Any]:
        """Initialize all available tools for the agent.

        Args:
            custom_db_path: Optional path to custom SQLite database
            custom_table_name: Optional custom table name in the database
            data_description: Optional description of the data for context
            dataset_preset: Preset dataset to use ("vienna", "milano")
            dataset_column_roles: Optional role hints inferred from uploaded dataset profiling

        Returns:
            List of initialized tool instances
        """
        project_root = get_project_root()
        dataset_column_roles = dataset_column_roles or {}

        # Initialize dataset-specific tools
        dataset_tool = self._create_dataset_tool(
            custom_db_path, custom_table_name, data_description, dataset_preset, project_root
        )
        species_list_tool = self._create_species_list_tool(project_root)
        co2_aggregate_tool = self._create_co2_aggregate_tool(
            custom_db_path, custom_table_name, dataset_preset, project_root
        )
        chart_tool = self._create_chart_tool(
            custom_db_path, custom_table_name, dataset_preset, project_root
        )
        map_tool = self._create_map_tool(
            custom_db_path, custom_table_name, dataset_preset, project_root, dataset_column_roles
        )

        # Static tools list
        static_tools = [
            CO2CalculationTool(),
            co2_aggregate_tool,
            CarbonContentTool(),
            EnvironmentEstimationTool(),
            dataset_tool,
            species_list_tool,
            chart_tool,
            map_tool,
            HeyerVolumeTool(),
            GeneralVolumeTool(),
            SimplifiedVolumeTool(),
            AllometricRelationTool(),
            LogAllometricTool(),
            ModelErrorTool(),
            LogFuelBiomassTool(),
            LeafBiomassTool(),
            StemBiomassTool(),
            RootBiomassTool(),
            TotalBiomassTool(),
            PaperSearchTool(),
            ExportDataTool(language=self._interface_language),
            LanguageDetectionTool(llm=self._base_llm),
            LanguageTranslationTool(llm=self._base_llm),
        ]

        # Load dynamic tools from JSON
        dynamic_tools = self._load_dynamic_tools()

        # Combine static and dynamic tools
        return static_tools + dynamic_tools

    def _create_dataset_tool(
        self,
        custom_db_path: Optional[Path],
        custom_table_name: Optional[str],
        data_description: str,
        dataset_preset: str,
        project_root: Path,
    ) -> DatasetQueryTool:
        """Create and configure the DatasetQueryTool."""
        if custom_db_path and custom_table_name:
            return DatasetQueryTool(
                db_path=custom_db_path,
                table_name=custom_table_name,
                user_description=data_description,
                llm=self._base_llm,
                fallback_llm=self._fallback_llm,
                embeddings=self._embeddings,
            )

        if dataset_preset in DATASET_PRESETS:
            preset = DATASET_PRESETS[dataset_preset]
            db_path = project_root / preset["db_path"]
            return DatasetQueryTool(
                db_path=db_path,
                table_name=preset["table_name"],
                user_description=preset["description"],
                llm=self._base_llm,
                fallback_llm=self._fallback_llm,
                embeddings=self._embeddings,
            )

        return DatasetQueryTool(
            llm=self._base_llm,
            fallback_llm=self._fallback_llm,
            embeddings=self._embeddings,
        )

    def _create_species_list_tool(self, project_root: Path) -> SpeciesListQueryTool:
        """Create and configure the SpeciesListQueryTool."""
        species_list_db_path = project_root / "dataset" / "species_list.db"
        return SpeciesListQueryTool(
            db_path=species_list_db_path,
            table_name="species_list",
            llm=self._base_llm,
            fallback_llm=self._fallback_llm,
            embeddings=self._embeddings,
        )

    def _create_co2_aggregate_tool(
        self,
        custom_db_path: Optional[Path],
        custom_table_name: Optional[str],
        dataset_preset: str,
        project_root: Path,
    ) -> CO2AggregateTool:
        """Create and configure the CO2AggregateTool."""
        if custom_db_path and custom_table_name:
            return CO2AggregateTool(
                db_path=custom_db_path,
                table_name=custom_table_name,
                dataset_type="custom",
                llm=self._base_llm,
            )

        if dataset_preset in DATASET_PRESETS:
            preset = DATASET_PRESETS[dataset_preset]
            db_path = project_root / preset["db_path"]
            return CO2AggregateTool(
                db_path=db_path,
                table_name=preset["table_name"],
                dataset_type=dataset_preset,
                llm=self._base_llm,
            )

        return CO2AggregateTool(
            dataset_type="vienna",
            llm=self._base_llm,
        )

    def _create_chart_tool(
        self,
        custom_db_path: Optional[Path],
        custom_table_name: Optional[str],
        dataset_preset: str,
        project_root: Path,
    ) -> ChartGenerationTool:
        """Create and configure the ChartGenerationTool for the selected dataset."""
        if custom_db_path and custom_table_name:
            return ChartGenerationTool(
                db_path=custom_db_path,
                table_name=custom_table_name,
                llm=self._base_llm,
                fallback_llm=self._fallback_llm,
            )

        if dataset_preset in DATASET_PRESETS:
            preset = DATASET_PRESETS[dataset_preset]
            return ChartGenerationTool(
                db_path=project_root / preset["db_path"],
                table_name=preset["table_name"],
                llm=self._base_llm,
                fallback_llm=self._fallback_llm,
            )

        return ChartGenerationTool(llm=self._base_llm, fallback_llm=self._fallback_llm)

    def _create_map_tool(
        self,
        custom_db_path: Optional[Path],
        custom_table_name: Optional[str],
        dataset_preset: str,
        project_root: Path,
        dataset_column_roles: Optional[dict] = None,
    ) -> MapGenerationTool:
        """Create and configure the MapGenerationTool."""
        dataset_column_roles = dataset_column_roles or {}
        latitude_candidates = dataset_column_roles.get("latitude_candidates") or []
        longitude_candidates = dataset_column_roles.get("longitude_candidates") or []
        lat_column = latitude_candidates[0] if latitude_candidates else "latitude"
        lon_column = longitude_candidates[0] if longitude_candidates else "longitude"

        if custom_db_path and custom_table_name:
            return MapGenerationTool(
                db_path=custom_db_path,
                table_name=custom_table_name,
                lat_column=lat_column,
                lon_column=lon_column,
                llm=self._base_llm,
                fallback_llm=self._fallback_llm,
            )

        if dataset_preset in DATASET_PRESETS:
            preset = DATASET_PRESETS[dataset_preset]
            return MapGenerationTool(
                db_path=project_root / preset["db_path"],
                table_name=preset["table_name"],
                llm=self._base_llm,
                fallback_llm=self._fallback_llm,
            )

        return MapGenerationTool(llm=self._base_llm, fallback_llm=self._fallback_llm)

    def _load_dynamic_tools(self) -> List[Any]:
        """Load dynamic tools from JSON configuration."""
        try:
            dynamic_loader = DynamicToolLoader()
            dynamic_tools = dynamic_loader.create_tools()
            self._dynamic_tools_summary = dynamic_loader.get_tools_summary()
            return dynamic_tools
        except Exception as e:
            logger.warning(f"Failed to load dynamic tools: {e}")
            self._dynamic_tools_summary = []
            return []
