from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

import pytest

deepeval = pytest.importorskip("deepeval")
from deepeval import assert_test  # noqa: E402
from deepeval.metrics import BaseMetric  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402

from streamlit_app.agent.config_loader import get_static_tools_metadata, match_query_to_tools
from streamlit_app.agent.tool_initializer import ToolInitializer
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
from streamlit_app.tools.root_biomass_tool import RootBiomassTool
from streamlit_app.tools.simplified_volume_tool import SimplifiedVolumeTool
from streamlit_app.tools.species_list_tool import SpeciesListQueryTool
from streamlit_app.tools.stem_biomass_tool import StemBiomassTool
from streamlit_app.tools.total_biomass_tool import TotalBiomassTool


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeLLM:
    def __init__(self, content: str = "") -> None:
        self._content = content

    def bind_tools(self, _tools: list[Any]) -> "FakeLLM":
        return self

    def invoke(self, _prompt: Any) -> FakeResponse:
        return FakeResponse(self._content)


class FakeEmbeddings:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[float(len(text)), 0.0, 1.0] for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return [float(len(text)), 0.0, 1.0]


Check = Callable[[Dict[str, Any]], Tuple[bool, str]]


class ToolContractMetric(BaseMetric):
    """Small deterministic DeepEval metric for local tool contracts."""

    threshold = 1.0
    evaluation_model = None
    strict_mode = True
    async_mode = False
    verbose_mode = False
    include_reason = True

    def __init__(self, name: str, checks: Iterable[Check]) -> None:
        self.threshold = 1.0
        self.name = name
        self._checks = list(checks)
        self.score = 0.0
        self.success = False
        self.reason = "Not measured yet"
        self.error: str | None = None

    def measure(self, test_case: LLMTestCase, *_args: Any, **_kwargs: Any) -> float:
        try:
            payload = json.loads(test_case.actual_output)
        except json.JSONDecodeError as exc:
            self.success = False
            self.score = 0.0
            self.reason = f"actual_output is not valid JSON: {exc}"
            return self.score

        failures: list[str] = []
        for check in self._checks:
            ok, reason = check(payload)
            if not ok:
                failures.append(reason)

        self.success = not failures
        self.score = 1.0 if self.success else 0.0
        self.reason = "All contract checks passed" if self.success else "; ".join(failures)
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *_args: Any, **_kwargs: Any) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:
        return self.name


def _get_path(payload: Dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def path_exists(path: str) -> Check:
    def check(payload: Dict[str, Any]) -> Tuple[bool, str]:
        value = _get_path(payload, path)
        return value is not None, f"Missing required path: {path}"

    return check


def path_numeric_gt(path: str, minimum: float) -> Check:
    def check(payload: Dict[str, Any]) -> Tuple[bool, str]:
        value = _get_path(payload, path)
        ok = isinstance(value, (int, float)) and value > minimum
        return ok, f"Expected {path} to be numeric and > {minimum}, got {value!r}"

    return check


def output_has_no_error(payload: Dict[str, Any]) -> Tuple[bool, str]:
    output = payload.get("tool_output", {})
    has_error = isinstance(output, dict) and bool(output.get("error"))
    return not has_error, f"Tool returned error: {output.get('error') if isinstance(output, dict) else output!r}"


def output_success(path: str = "tool_output.success") -> Check:
    def check(payload: Dict[str, Any]) -> Tuple[bool, str]:
        value = _get_path(payload, path)
        return value is True, f"Expected {path} to be true, got {value!r}"

    return check


def _assert_deepeval_contract(
    *,
    case_name: str,
    user_input: str,
    actual: Dict[str, Any],
    checks: Iterable[Check],
) -> None:
    os.environ.setdefault("DEEPEVAL_RESULTS_FOLDER", ".deepeval-results")
    test_case = LLMTestCase(
        input=user_input,
        actual_output=json.dumps(actual, default=str, ensure_ascii=False),
        expected_output=json.dumps({"case": case_name}, ensure_ascii=False),
    )
    metric = ToolContractMetric(name=f"{case_name}_contract", checks=checks)
    assert_test(test_case, [metric], run_async=False)


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

    _assert_deepeval_contract(
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
@pytest.mark.parametrize(
    ("query", "expected_tool"),
    [
        ("calcola la CO2 per un albero con diametro 30 cm", "calculate_co2_sequestration"),
        ("qual e il contenuto di carbonio per Oak?", "lookup_carbon_content"),
        ("stima ambiente volume e biomassa con diametro 30", "calculate_environmental_estimates"),
        ("quanti alberi ci sono nel dataset?", "query_tree_dataset"),
        ("crea un grafico a barre per distretto", "generate_chart"),
        ("mostra una mappa degli alberi di Milano", "generate_map"),
    ],
)
def test_deepeval_keyword_router_returns_real_tool_names(query: str, expected_tool: str) -> None:
    real_tool_names = _real_tool_names()
    matched = match_query_to_tools(query)
    unknown = sorted(set(matched) - real_tool_names)

    _assert_deepeval_contract(
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
@pytest.mark.parametrize(
    ("case_name", "tool_factory", "tool_input", "checks"),
    [
        (
            "co2_single_tree",
            CO2CalculationTool,
            {"dbh_cm": 30.0, "height_m": 12.0},
            [path_numeric_gt("tool_output.total_biomass_t", 0), path_numeric_gt("tool_output.co2_stock_t", 0)],
        ),
        (
            "carbon_content_lookup",
            CarbonContentTool,
            {"species": "Oak"},
            [path_exists("tool_output.species_name"), path_numeric_gt("tool_output.carbon_fraction", 0)],
        ),
        (
            "environmental_estimates",
            EnvironmentEstimationTool,
            {"diameter_cm": 30.0, "height_m": 12.0},
            [
                path_numeric_gt("tool_output.results.volume_dm3", 0),
                path_numeric_gt("tool_output.results.biomass_kg", 0),
                path_numeric_gt("tool_output.results.carbon_stock_kg", 0),
            ],
        ),
        (
            "heyer_volume",
            HeyerVolumeTool,
            {"sections": [0.1, 0.2, 0.3]},
            [path_numeric_gt("tool_output.volume", 0)],
        ),
        (
            "general_volume",
            GeneralVolumeTool,
            {"diameter_m": 0.3, "height_m": 12.0, "coefficient_a": 0.5},
            [path_numeric_gt("tool_output.volume", 0)],
        ),
        (
            "simplified_volume",
            SimplifiedVolumeTool,
            {"diameter_m": 0.3},
            [path_numeric_gt("tool_output.volume", 0)],
        ),
        (
            "allometric_relation",
            AllometricRelationTool,
            {"variable_x": 10.0, "coeff_a": 2.0, "exponent_b": 0.5},
            [path_numeric_gt("tool_output.result_y", 0)],
        ),
        (
            "log_allometric",
            LogAllometricTool,
            {"variable_x": 10.0, "coeff_a": 2.0, "coeff_b": 0.5},
            [path_numeric_gt("tool_output.ln_y", 0)],
        ),
        (
            "model_error",
            ModelErrorTool,
            {"measured_value": 100.0, "estimated_value": 90.0},
            [path_numeric_gt("tool_output.percentage_error", 0)],
        ),
        (
            "log_fuel_biomass",
            LogFuelBiomassTool,
            {"variable_x": 10.0, "correction_factor": 1.1, "intercept_a": 0.2, "slope_b": 0.7},
            [path_numeric_gt("tool_output.result_y", 0)],
        ),
        (
            "leaf_biomass",
            LeafBiomassTool,
            {"diameter_cm": 30.0, "height_m": 12.0, "age_years": 10.0},
            [path_numeric_gt("tool_output.leaf_biomass", 0)],
        ),
        (
            "stem_biomass",
            StemBiomassTool,
            {"diameter": 0.3, "age_years": 5.0},
            [path_numeric_gt("tool_output.stem_biomass", 0)],
        ),
        (
            "root_biomass",
            RootBiomassTool,
            {"diameter_cm": 30.0, "height_m": 12.0, "age_years": 10.0, "root_shoot_ratio": 0.24},
            [path_numeric_gt("tool_output.root_biomass", 0)],
        ),
        (
            "total_biomass",
            TotalBiomassTool,
            {"diameter_cm": 30.0, "height_m": 12.0, "age_years": 10.0, "root_shoot_ratio": 0.24},
            [path_numeric_gt("tool_output.total_biomass", 0)],
        ),
        (
            "language_detection",
            LanguageDetectionTool,
            {"text": "This is an English sentence about urban trees."},
            [path_exists("tool_output.detected_language")],
        ),
    ],
)
def test_deepeval_direct_tools_return_expected_contracts(
    case_name: str,
    tool_factory: Callable[[], Any],
    tool_input: Dict[str, Any],
    checks: List[Check],
) -> None:
    tool = tool_factory()
    output = tool.invoke(tool_input)

    _assert_deepeval_contract(
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

    _assert_deepeval_contract(
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

    _assert_deepeval_contract(
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

    for case_name, user_input, tool, tool_input, checks in [
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
    ]:
        output = tool.invoke(tool_input)
        _assert_deepeval_contract(
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
        _assert_deepeval_contract(
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

    _assert_deepeval_contract(
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

    _assert_deepeval_contract(
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
