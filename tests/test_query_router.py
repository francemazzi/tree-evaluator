from __future__ import annotations

from streamlit_app.agent.config_loader import match_query_to_tools, match_query_to_tools_with_scores
from streamlit_app.agent.query_optimizer import QueryOptimizer


def test_router_prefers_map_over_generic_tree_dataset_match() -> None:
    matches = match_query_to_tools("mostra una mappa degli alberi di Milano")

    assert matches == ["generate_map"]


def test_router_prefers_chart_over_dataset_statistics_terms() -> None:
    matches = match_query_to_tools("crea un grafico a barre per distretto")

    assert matches == ["generate_chart"]


def test_router_distinguishes_species_taxonomy_from_tree_dataset() -> None:
    matches = match_query_to_tools("qual e la famiglia di Abies alba?")

    assert matches == ["query_species_list"]


def test_router_distinguishes_co2_aggregate_from_single_tree() -> None:
    aggregate = match_query_to_tools("calcola CO2 totale degli alberi di Milano")
    single = match_query_to_tools("calcola la CO2 per un albero con diametro 30 cm")

    assert aggregate == ["calculate_co2_aggregate"]
    assert single == ["calculate_co2_sequestration"]


def test_router_prefers_carbon_content_lookup_over_co2_calculators() -> None:
    matches = match_query_to_tools("qual e il contenuto di carbonio per Oak?")

    assert matches == ["lookup_carbon_content"]


def test_router_prefers_specific_biomass_formula_tools() -> None:
    assert match_query_to_tools("calcola biomassa fogliare") == ["calculate_leaf_biomass"]
    assert match_query_to_tools("calcola biomassa radicale") == ["calculate_root_biomass"]
    assert match_query_to_tools("calcola biomassa totale") == ["calculate_total_biomass"]


def test_router_exposes_scores_for_debugging() -> None:
    scored = match_query_to_tools_with_scores("quanti alberi ci sono nel dataset?")

    assert scored[0]["name"] == "query_tree_dataset"
    assert scored[0]["score"] > 0
    assert "matched_keywords" in scored[0]
    assert "reason" in scored[0]


def test_simple_plan_fallback_uses_real_dataset_tool_name() -> None:
    optimizer = QueryOptimizer(
        interface_language="it",
        fallback_model="unused",
        create_chat_without_tools=lambda *_args, **_kwargs: None,
    )

    plan = optimizer._create_simple_tool_plan("ciao come stai", "it")

    assert plan[0]["tools"] == ["query_tree_dataset"]
