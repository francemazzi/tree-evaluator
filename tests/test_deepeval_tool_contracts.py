from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List

import pytest

pytest.importorskip("deepeval")

from streamlit_app.agent.config_loader import get_static_tools_metadata, match_query_to_tools  # noqa: E402
from streamlit_app.agent.tool_initializer import ToolInitializer  # noqa: E402
from streamlit_app.tools.chart_tool import ChartGenerationTool  # noqa: E402
from streamlit_app.tools.co2_aggregate_tool import CO2AggregateTool  # noqa: E402
from streamlit_app.tools.dataset_tool import DatasetQueryTool  # noqa: E402
from streamlit_app.tools.dynamic_tool_loader import DynamicToolLoader  # noqa: E402
from streamlit_app.tools.export_tool import ExportDataTool  # noqa: E402
from streamlit_app.tools.language_tool import LanguageTranslationTool  # noqa: E402
from streamlit_app.tools.map_tool import MapGenerationTool  # noqa: E402
from streamlit_app.tools.species_list_tool import SpeciesListQueryTool  # noqa: E402
from tests.deepeval_contract_helpers import (  # noqa: E402
    Check,
    FakeEmbeddings,
    FakeLLM,
    assert_deepeval_contract,
    output_has_no_error,
    output_success,
    path_exists,
    path_numeric_gt,
)
from tests.deepeval_tool_cases import DIRECT_TOOL_CONTRACT_CASES, ROUTER_CASES  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _real_tool_names() -> set[str]:
    initializer = ToolInitializer(
        base_llm=FakeLLM(""),
        fallback_llm=FakeLLM(""),
        embeddings=FakeEmbeddings(),
    )
    return {tool.name for tool in initializer.initialize_tools()}


@pytest.mark.deepeval
def test_deepeval_static_tool_metadata_matches_initialized_tool_names() -> None:
    real_tool_names = _real_tool_names()
    metadata_names = {tool["name"] for tool in get_static_tools_metadata()}
    missing = sorted(metadata_names - real_tool_names)

    assert_deepeval_contract(
        case_name="static_tool_metadata",
        user_input="Verifica che i metadata dei tool puntino ai BaseTool reali.",
        actual={
            "metadata_tool_count": len(metadata_names),
            "initialized_tool_count": len(real_tool_names),
            "missing_tool_names": missing,
        },
        checks=[
            lambda payload: (
                not payload["missing_tool_names"],
                f"Metadata tool names not initialized: {payload['missing_tool_names']}",
            )
        ],
    )


@pytest.mark.deepeval
@pytest.mark.parametrize(("query", "expected_tool"), ROUTER_CASES)
def test_deepeval_keyword_router_returns_real_tool_names(query: str, expected_tool: str) -> None:
    real_tool_names = _real_tool_names()
    matched = match_query_to_tools(query)
    unknown = sorted(set(matched) - real_tool_names)

    assert_deepeval_contract(
        case_name=f"router_{expected_tool}",
        user_input=query,
        actual={
            "query": query,
            "matched_tools": matched,
            "expected_tool": expected_tool,
            "unknown_tools": unknown,
        },
        checks=[
            lambda payload: (
                payload["expected_tool"] in payload["matched_tools"],
                f"Expected {payload['expected_tool']} in {payload['matched_tools']}",
            ),
            lambda payload: (
                payload["matched_tools"] and payload["matched_tools"][0] == payload["expected_tool"],
                f"Expected {payload['expected_tool']} as first match, got {payload['matched_tools']}",
            ),
            lambda payload: (
                not payload["unknown_tools"],
                f"Router returned unknown tools: {payload['unknown_tools']}",
            ),
        ],
    )


@pytest.mark.deepeval
@pytest.mark.parametrize(("case_name", "tool_factory", "tool_input", "checks"), DIRECT_TOOL_CONTRACT_CASES)
def test_deepeval_direct_tools_return_expected_contracts(
    case_name: str,
    tool_factory: Callable[[], Any],
    tool_input: Dict[str, Any],
    checks: List[Check],
) -> None:
    tool = tool_factory()
    output = tool.invoke(tool_input)

    assert_deepeval_contract(
        case_name=case_name,
        user_input=f"Run {tool.name}",
        actual={"tool_name": tool.name, "tool_input": tool_input, "tool_output": output},
        checks=[output_has_no_error, *checks],
    )


@pytest.mark.deepeval
def test_deepeval_translation_tool_contract_with_fake_llm() -> None:
    tool = LanguageTranslationTool(llm=FakeLLM("Hello tree"))
    tool_input = {"text": "Ciao albero", "source_language": "it", "target_language": "en"}
    output = tool.invoke(tool_input)

    assert_deepeval_contract(
        case_name="translation_tool",
        user_input="traduci ciao albero in inglese",
        actual={"tool_name": tool.name, "tool_input": tool_input, "tool_output": output},
        checks=[output_has_no_error, path_exists("tool_output.translated_text")],
    )


@pytest.mark.deepeval
def test_deepeval_export_tool_contract(tmp_path: Path) -> None:
    tool = ExportDataTool(output_dir=tmp_path)
    tool_input = {"data": [{"species": "Oak", "count": 2}], "format": "csv", "filename": "deepeval_export"}
    output = tool.invoke(tool_input)

    assert_deepeval_contract(
        case_name="export_data",
        user_input="esporta questi dati in csv",
        actual={"tool_name": tool.name, "tool_input": tool_input, "tool_output": output},
        checks=[
            output_has_no_error,
            output_success(),
            path_exists("tool_output.file_path"),
            lambda payload: (
                Path(payload["tool_output"]["file_path"]).exists(),
                f"Export file not found: {payload['tool_output'].get('file_path')}",
            ),
        ],
    )


@pytest.mark.deepeval
def test_deepeval_dataset_and_species_query_tools_execute_readonly_sql() -> None:
    vienna_db = PROJECT_ROOT / "dataset" / "BAUMKATOGD.db"
    species_db = PROJECT_ROOT / "dataset" / "species_list.db"
    if not vienna_db.exists() or not species_db.exists():
        pytest.skip("SQLite datasets not available")

    dataset_tool = DatasetQueryTool(
        db_path=vienna_db,
        table_name="baumkatogd",
        llm=FakeLLM("SELECT COUNT(*) AS total FROM baumkatogd"),
        embeddings=FakeEmbeddings(),
    )
    species_tool = SpeciesListQueryTool(
        db_path=species_db,
        table_name="species_list",
        llm=FakeLLM(
            "SELECT genus_species, family, taxonomic_order "
            "FROM species_list WHERE genus_species LIKE '%Abelia triflora%' LIMIT 5"
        ),
        embeddings=FakeEmbeddings(),
    )

    cases = [
        (
            "dataset_query",
            "quanti alberi ci sono nel dataset?",
            dataset_tool,
            {"natural_query": "quanti alberi ci sono nel dataset?"},
            [path_numeric_gt("tool_output.result", 0), path_exists("tool_output.sql_executed")],
        ),
        (
            "species_list_query",
            "dimmi famiglia e ordine di Abelia triflora",
            species_tool,
            {"natural_query": "dimmi famiglia e ordine di Abelia triflora"},
            [path_numeric_gt("tool_output.row_count", 0), path_exists("tool_output.sql_executed")],
        ),
    ]

    for case_name, user_input, tool, tool_input, checks in cases:
        output = tool.invoke(tool_input)
        assert_deepeval_contract(
            case_name=case_name,
            user_input=user_input,
            actual={"tool_name": tool.name, "tool_input": tool_input, "tool_output": output},
            checks=[output_has_no_error, *checks],
        )


@pytest.mark.deepeval
def test_deepeval_co2_aggregate_chart_and_map_tools_execute() -> None:
    vienna_db = PROJECT_ROOT / "dataset" / "BAUMKATOGD.db"
    milano_db = PROJECT_ROOT / "dataset" / "dataset_milano.db"
    if not vienna_db.exists() or not milano_db.exists():
        pytest.skip("SQLite datasets not available")

    map_sql_json = json.dumps(
        {
            "sql": (
                "SELECT latitude, longitude, genus_species "
                "FROM milano_trees "
                "WHERE latitude IS NOT NULL AND longitude IS NOT NULL "
                "LIMIT 10"
            ),
            "suggested_title": "Mappa test Milano",
            "popup_columns": ["genus_species"],
            "center_lat": 45.4642,
            "center_lon": 9.19,
            "zoom": 11,
        }
    )

    cases = [
        (
            "co2_aggregate",
            CO2AggregateTool(
                db_path=vienna_db,
                table_name="baumkatogd",
                dataset_type="vienna",
                llm=FakeLLM(
                    "SELECT objectid, trunk_circumference, tree_height, genus_species "
                    "FROM baumkatogd "
                    "WHERE trunk_circumference > 0 AND tree_height > 0 "
                    "LIMIT 25"
                ),
            ),
            {"natural_query": "calcola lo stock CO2 su un campione di alberi"},
            [path_numeric_gt("tool_output.tree_count", 0), path_numeric_gt("tool_output.co2_stock_t", 0)],
        ),
        (
            "chart_generation",
            ChartGenerationTool(db_path=vienna_db, table_name="baumkatogd"),
            {"chart_type": "bar", "data_query": "Numero di alberi per distretto"},
            [output_success(), path_exists("tool_output.chart_json"), path_numeric_gt("tool_output.data_points", 0)],
        ),
        (
            "map_generation",
            MapGenerationTool(db_path=milano_db, table_name="milano_trees", llm=FakeLLM(map_sql_json)),
            {"map_type": "markers", "data_query": "mostra 10 alberi sulla mappa", "max_points": 10},
            [output_success(), path_exists("tool_output.map_html"), path_numeric_gt("tool_output.data_points", 0)],
        ),
    ]

    for case_name, tool, tool_input, checks in cases:
        output = tool.invoke(tool_input)
        assert_deepeval_contract(
            case_name=case_name,
            user_input=f"Run {tool.name}",
            actual={"tool_name": tool.name, "tool_input": tool_input, "tool_output": output},
            checks=[output_has_no_error, *checks],
        )


@pytest.mark.deepeval
def test_deepeval_dynamic_formula_tools_load_and_execute() -> None:
    loader = DynamicToolLoader(PROJECT_ROOT / "dataset" / "tools.json")
    dynamic_tools = loader.create_tools()
    failures: list[dict[str, Any]] = []

    for tool in dynamic_tools:
        tool_input: dict[str, Any] = {}
        for field_name, field in tool.args_schema.model_fields.items():
            annotation = str(field.annotation)
            tool_input[field_name] = [1.0, 2.0, 3.0] if "List" in annotation or "list" in annotation else 2.0

        output = tool.invoke(tool_input)
        if not output.get("success"):
            failures.append({"tool": tool.name, "input": tool_input, "output": output})

    assert_deepeval_contract(
        case_name="dynamic_formula_tools",
        user_input="verifica caricamento ed esecuzione dei tool dinamici",
        actual={
            "dynamic_tool_count": len(dynamic_tools),
            "failed_tools": failures,
        },
        checks=[
            lambda payload: (
                payload["dynamic_tool_count"] >= 10,
                f"Expected at least 10 dynamic tools, got {payload['dynamic_tool_count']}",
            ),
            lambda payload: (
                not payload["failed_tools"],
                f"Dynamic tools failed: {payload['failed_tools']}",
            ),
        ],
    )


@pytest.mark.deepeval
@pytest.mark.slow
def test_deepeval_agent_e2e_smoke_optional() -> None:
    if os.getenv("RUN_DEEPEVAL_AGENT_E2E") != "1":
        pytest.skip("Set RUN_DEEPEVAL_AGENT_E2E=1 to run real LLM agent eval")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    from streamlit_app.agent import TreeEvaluatorAgent

    agent = TreeEvaluatorAgent(openai_api_key=api_key, interface_language="it")
    response = agent.chat("Calcola la CO2 per un albero con diametro 30 cm e altezza 12 m.")

    assert_deepeval_contract(
        case_name="agent_e2e_smoke",
        user_input="Calcola la CO2 per un albero con diametro 30 cm e altezza 12 m.",
        actual={"agent_output": response},
        checks=[
            lambda payload: (
                "co2" in payload["agent_output"].lower() or "carbon" in payload["agent_output"].lower(),
                f"Agent output does not mention CO2/carbon: {payload['agent_output']}",
            )
        ],
    )
