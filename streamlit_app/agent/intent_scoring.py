from __future__ import annotations


def intent_score(query_lower: str, tool_name: str) -> float:
    """Return intent-specific score adjustments for a tool."""
    score = 0.0
    map_intent = _has_any(query_lower, ["mappa", "map", "gps", "coordinate", "heatmap", "markers"])
    chart_intent = _has_any(query_lower, ["grafico", "chart", "plot", "barre", "torta", "istogramma", "visualizza"])
    export_intent = _has_any(query_lower, ["esporta", "export", "csv", "excel", "xlsx", "scarica", "download"])
    paper_intent = _has_any(query_lower, ["paper", "articoli", "pubmed", "arxiv", "ricerca scientifica"])
    carbon_content_intent = _has_any(query_lower, ["contenuto", "content", "frazione", "fraction"]) and _has_any(query_lower, ["carbonio", "carbon"])
    taxonomy_intent = _has_any(query_lower, ["tassonomia", "taxonomy", "famiglia", "family", "ordine", "order", "tratti", "traits", "leaf_type", "growth"])
    dataset_intent = _has_any(query_lower, ["dataset", "quanti", "quante", "count", "statistiche", "statistics", "top", "piu comune", "più comune", "distretto", "municipio"])
    co2_intent = _has_any(query_lower, ["co2", "sequestro", "sequestration"])
    aggregate_intent = _has_any(query_lower, ["totale", "total", "media", "average", "aggregato", "aggregate", "tutti", "all", "gruppo", "group"]) or (co2_intent and _has_any(query_lower, ["alberi", "trees"]))
    single_tree_intent = _has_any(query_lower, ["singolo", "single", "un albero", "una pianta", "dbh", "diametro", "diameter", "altezza", "height"])
    environment_intent = _has_any(query_lower, ["ambiente", "environment", "environmental", "stima ambientale", "estimates"])
    biomass_intent = _has_any(query_lower, ["biomassa", "biomass"])
    specific_formula_intent = _has_any(query_lower, ["heyer", "semplificato", "simplified", "generale", "general", "allometria", "allometric", "logaritmo", "log", "errore", "error", "fogliare", "foglie", "leaf", "fusto", "tronco", "stem", "trunk", "radicale", "radici", "root"])

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
        elif _has_any(query_lower, ["specie", "species"]) and not dataset_intent:
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
        elif not specific_formula_intent and _has_any(query_lower, ["volume", "biomassa", "biomass"]) and _has_any(query_lower, ["stima", "estimate", "calcola", "calculate"]):
            score += 3.0
    if tool_name == "calculate_leaf_biomass" and biomass_intent and _has_any(query_lower, ["foglie", "fogliare", "leaf", "chioma", "canopy"]):
        score += 8.0
    if tool_name == "calculate_stem_biomass" and biomass_intent and _has_any(query_lower, ["fusto", "tronco", "stem", "trunk"]):
        score += 8.0
    if tool_name == "calculate_root_biomass" and biomass_intent and _has_any(query_lower, ["radici", "radicale", "root", "sotterraneo"]):
        score += 8.0
    if tool_name == "calculate_total_biomass" and biomass_intent and _has_any(query_lower, ["totale", "total"]):
        score += 8.0
    return score


def _has_any(query_lower: str, terms: list[str]) -> bool:
    return any(term in query_lower for term in terms)
