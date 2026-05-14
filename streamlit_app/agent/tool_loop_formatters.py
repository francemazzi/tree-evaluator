from __future__ import annotations

import json
from typing import List, Sequence

from langchain_core.messages import BaseMessage


def format_dataset_results(results: List[dict], messages: Sequence[BaseMessage], language: str = "it") -> str:
    """Format dataset results as user-friendly response."""
    from streamlit_app.agent.response_builder import ResponseBuilder
    return ResponseBuilder.format_dataset_results(results, messages, language)

def format_chart_results(chart_results: List[dict], messages: Sequence[BaseMessage]) -> str:
    """Format chart results as user-friendly response."""
    if not chart_results:
        return "Non sono riuscito a generare il grafico richiesto."
    
    # Get the most recent successful chart
    chart = chart_results[0]
    
    # Build response with chart markers for UI parsing
    chart_type = chart.get("chart_type", "grafico")
    data_points = chart.get("data_points", 0)
    title = chart.get("title", "Grafico")
    description = chart.get("description", f"Grafico {chart_type} generato con successo")
    
    response = f"Ecco il {chart_type} che hai richiesto: **{title}**\n\n"
    response += f"{description} con {data_points} punti dati.\n\n"
    
    # Add chart data markers for UI
    chart_json_str = json.dumps(chart, ensure_ascii=False, indent=2)
    response += f"\nCHART_DATA_START\n{chart_json_str}\nCHART_DATA_END\n"
    
    return response

def format_map_results(map_results: List[dict], messages: Sequence[BaseMessage]) -> str:
    """Format map results as user-friendly response."""
    if not map_results:
        return "Non sono riuscito a generare la mappa richiesta."
    
    # Get the most recent successful map
    map_data = map_results[0]
    
    # Build response with map markers for UI parsing
    map_type = map_data.get("map_type", "mappa")
    data_points = map_data.get("data_points", 0)
    title = map_data.get("title", "Mappa")
    description = map_data.get("description", f"Mappa {map_type} generata con successo")
    
    response = f"Ecco la {map_type} che hai richiesto: **{title}**\n\n"
    response += f"{description} con {data_points} punti visualizzati.\n\n"
    
    # Add map data markers for UI
    map_json_str = json.dumps(map_data, ensure_ascii=False, indent=2)
    response += f"\nMAP_DATA_START\n{map_json_str}\nMAP_DATA_END\n"
    
    return response

def format_co2_aggregate_results(result: dict, language: str = "it") -> str:
    """Format CO2 aggregate results as user-friendly response with CARBON value prominently.
    
    Args:
        result: CO2 aggregate result dictionary
        language: Response language
        
    Returns:
        Formatted response string
    """
    # If answer_hint is present, use it directly
    if "answer_hint" in result:
        return result["answer_hint"]
    
    # Otherwise build response manually
    carbon_stock = result.get("carbon_stock_t", 0)
    co2_stock = result.get("co2_stock_t", 0)
    tree_count = result.get("tree_count", 0)
    total_biomass = result.get("total_biomass_t", 0)
    agb = result.get("above_ground_biomass_t", 0)
    bgb = result.get("below_ground_biomass_t", 0)
    species = result.get("dominant_species", "")
    
    # Get parameters
    params = result.get("parameters", {})
    cf = params.get("carbon_fraction", {}).get("value", 0.47)
    rs = params.get("root_shoot_ratio", {}).get("value", 0.24)
    
    if language == "en":
        response = f"""The carbon stock of {species} is **{carbon_stock:,.2f} t C** (tonnes of carbon).

Details:
- Trees analyzed: {tree_count:,}
- Carbon stock: {carbon_stock:,.2f} t C
- CO2 equivalent: {co2_stock:,.2f} t CO2
- Total biomass: {total_biomass:,.2f} t
  - Above-ground biomass (AGB): {agb:,.2f} t
  - Below-ground biomass (BGB): {bgb:,.2f} t

**Formulas used:**
- AGB = 0.0673 × (WD × DBH² × H)^0.976 (Chave et al., 2014)
- BGB = AGB × R/S
- C = Biomass × CF
- CO2 = C × (44/12)

**Parameters:**
- Wood density (WD): 0.6 g/cm³
- Carbon fraction (CF): {cf} ({cf*100:.1f}%)
- Root-to-shoot ratio (R/S): {rs}

Tools used: calculate_co2_aggregate"""
    else:
        response = f"""Lo stock di carbonio di {species} è di **{carbon_stock:,.2f} t C** (tonnellate di carbonio).

Dettagli:
- Alberi analizzati: {tree_count:,}
- Stock di carbonio: {carbon_stock:,.2f} t C
- CO2 equivalente: {co2_stock:,.2f} t CO2
- Biomassa totale: {total_biomass:,.2f} t
  - Biomassa epigea (AGB): {agb:,.2f} t
  - Biomassa ipogea (BGB): {bgb:,.2f} t

**Formule utilizzate:**
- AGB = 0.0673 × (WD × DBH² × H)^0.976 (Chave et al., 2014)
- BGB = AGB × R/S
- C = Biomassa × CF
- CO2 = C × (44/12)

**Parametri:**
- Densità legno (WD): 0.6 g/cm³
- Frazione carbonio (CF): {cf} ({cf*100:.1f}%)
- Rapporto R/S: {rs}

Tool utilizzati: calculate_co2_aggregate"""
    
    return response

