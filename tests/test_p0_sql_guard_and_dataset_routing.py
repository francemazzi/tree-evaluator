from __future__ import annotations

from pathlib import Path

import pytest

from streamlit_app.tools.chart_tool import ChartGenerationTool
from streamlit_app.tools.co2_aggregate_tool import CO2AggregateTool
from streamlit_app.tools.map_tool import MapGenerationTool
from streamlit_app.tools.sql_validator import SQLValidator
from streamlit_app.agent.tool_initializer import ToolInitializer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeLLM:
    def __init__(self, content: str) -> None:
        self._content = content

    def invoke(self, _prompt: str):
        class Response:
            pass

        response = Response()
        response.content = self._content
        return response


def _tool_initializer() -> ToolInitializer:
    return ToolInitializer(
        base_llm=FakeLLM("{}"),
        fallback_llm=FakeLLM("{}"),
        embeddings=None,
    )


def _tool_by_name(tools: list, name: str):
    return next(tool for tool in tools if tool.name == name)


def test_sql_validator_accepts_readonly_cte_with_allowed_base_table() -> None:
    sql = """
    WITH species_counts AS (
        SELECT genus_species, COUNT(*) AS count
        FROM baumkatogd
        GROUP BY genus_species
    ),
    ranked AS (
        SELECT genus_species, count FROM species_counts
    )
    SELECT * FROM ranked
    """

    is_valid, sanitized, error = SQLValidator(
        allowed_tables=["baumkatogd"],
        default_limit=5000,
    ).validate(sql)

    assert is_valid is True
    assert error == ""
    assert "LIMIT 5000" in sanitized


def test_co2_aggregate_rejects_non_select_sql_from_llm() -> None:
    db_path = PROJECT_ROOT / "dataset" / "BAUMKATOGD.db"
    if not db_path.exists():
        pytest.skip("Vienna database not available")

    tool = CO2AggregateTool(
        db_path=db_path,
        table_name="baumkatogd",
        dataset_type="vienna",
        llm=FakeLLM("DROP TABLE baumkatogd"),
    )

    result = tool._run("calcola CO2 totale")

    assert "SQL validation failed" in result["error"]
    assert result["sql_attempted"] == "DROP TABLE baumkatogd"


def test_chart_uses_configured_milano_table_for_predefined_query() -> None:
    db_path = PROJECT_ROOT / "dataset" / "dataset_milano.db"
    if not db_path.exists():
        pytest.skip("Milano database not available")

    tool = ChartGenerationTool(db_path=db_path, table_name="milano_trees")
    result = tool._run("bar", "Numero di alberi per municipio")

    assert result["success"] is True
    assert "FROM milano_trees" in result["sql_executed"]
    assert "baumkatogd" not in result["sql_executed"]


def test_map_uses_configured_vienna_table_and_does_not_fallback_to_milano() -> None:
    db_path = PROJECT_ROOT / "dataset" / "BAUMKATOGD.db"
    if not db_path.exists():
        pytest.skip("Vienna database not available")

    tool = MapGenerationTool(db_path=db_path, table_name="baumkatogd")
    result = tool._run("markers", "Mostra gli alberi su mappa")

    assert result["success"] is False
    assert "coordinate GPS" in result["error"]


def test_agent_routes_custom_dataset_to_chart_and_map_tools(tmp_path: Path) -> None:
    custom_db_path = tmp_path / "custom.db"
    initializer = _tool_initializer()

    tools = initializer.initialize_tools(
        custom_db_path=custom_db_path,
        custom_table_name="uploaded_data",
        data_description="Dataset custom",
        dataset_preset="vienna",
        dataset_column_roles={
            "latitude_candidates": ["lat"],
            "longitude_candidates": ["lon"],
        },
    )

    dataset_tool = _tool_by_name(tools, "query_tree_dataset")
    chart_tool = _tool_by_name(tools, "generate_chart")
    map_tool = _tool_by_name(tools, "generate_map")

    assert dataset_tool._db_path == custom_db_path
    assert dataset_tool._table_name == "uploaded_data"
    assert chart_tool._db_path == custom_db_path
    assert chart_tool._table_name == "uploaded_data"
    assert map_tool._db_path == custom_db_path
    assert map_tool._table_name == "uploaded_data"
    assert map_tool._lat_column == "lat"
    assert map_tool._lon_column == "lon"


def test_agent_routes_vienna_chart_and_map_to_vienna_dataset() -> None:
    initializer = _tool_initializer()

    tools = initializer.initialize_tools(
        custom_db_path=None,
        custom_table_name=None,
        data_description="",
        dataset_preset="vienna",
    )

    chart_tool = _tool_by_name(tools, "generate_chart")
    map_tool = _tool_by_name(tools, "generate_map")

    assert chart_tool._db_path.name == "BAUMKATOGD.db"
    assert chart_tool._table_name == "baumkatogd"
    assert map_tool._db_path.name == "BAUMKATOGD.db"
    assert map_tool._table_name == "baumkatogd"


def test_agent_routes_milano_chart_and_map_to_milano_dataset() -> None:
    initializer = _tool_initializer()

    tools = initializer.initialize_tools(
        custom_db_path=None,
        custom_table_name=None,
        data_description="",
        dataset_preset="milano",
    )

    chart_tool = _tool_by_name(tools, "generate_chart")
    map_tool = _tool_by_name(tools, "generate_map")

    assert chart_tool._db_path.name == "dataset_milano.db"
    assert chart_tool._table_name == "milano_trees"
    assert map_tool._db_path.name == "dataset_milano.db"
    assert map_tool._table_name == "milano_trees"
