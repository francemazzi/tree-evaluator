from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

from streamlit_app.tools.allometric_relation_tool import AllometricRelationTool
from streamlit_app.tools.carbon_content_tool import CarbonContentTool
from streamlit_app.tools.co2_tool import CO2CalculationTool
from streamlit_app.tools.environment_tool import EnvironmentEstimationTool
from streamlit_app.tools.general_volume_tool import GeneralVolumeTool
from streamlit_app.tools.heyer_volume_tool import HeyerVolumeTool
from streamlit_app.tools.language_tool import LanguageDetectionTool
from streamlit_app.tools.leaf_biomass_tool import LeafBiomassTool
from streamlit_app.tools.log_allometric_tool import LogAllometricTool
from streamlit_app.tools.log_fuel_biomass_tool import LogFuelBiomassTool
from streamlit_app.tools.model_error_tool import ModelErrorTool
from streamlit_app.tools.root_biomass_tool import RootBiomassTool
from streamlit_app.tools.simplified_volume_tool import SimplifiedVolumeTool
from streamlit_app.tools.stem_biomass_tool import StemBiomassTool
from streamlit_app.tools.total_biomass_tool import TotalBiomassTool
from tests.deepeval_contract_helpers import Check, path_exists, path_numeric_gt


ROUTER_CASES = [
    ("calcola la CO2 per un albero con diametro 30 cm", "calculate_co2_sequestration"),
    ("qual e il contenuto di carbonio per Oak?", "lookup_carbon_content"),
    ("stima ambiente volume e biomassa con diametro 30", "calculate_environmental_estimates"),
    ("quanti alberi ci sono nel dataset?", "query_tree_dataset"),
    ("crea un grafico a barre per distretto", "generate_chart"),
    ("mostra una mappa degli alberi di Milano", "generate_map"),
]


DIRECT_TOOL_CONTRACT_CASES: list[Tuple[str, Callable[[], Any], Dict[str, Any], List[Check]]] = [
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
    ("heyer_volume", HeyerVolumeTool, {"sections": [0.1, 0.2, 0.3]}, [path_numeric_gt("tool_output.volume", 0)]),
    (
        "general_volume",
        GeneralVolumeTool,
        {"diameter_m": 0.3, "height_m": 12.0, "coefficient_a": 0.5},
        [path_numeric_gt("tool_output.volume", 0)],
    ),
    ("simplified_volume", SimplifiedVolumeTool, {"diameter_m": 0.3}, [path_numeric_gt("tool_output.volume", 0)]),
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
]
