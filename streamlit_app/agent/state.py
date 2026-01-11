"""State definitions and dataset configurations for the Tree Evaluator Agent."""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State for the LangGraph agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    optimized_query: Optional[str]
    tasks: Optional[List[str]]
    validation_result: Optional[dict]
    context_summary: Optional[str]  # Summary of important context
    message_count: Optional[int]  # Track conversation length
    chart_data: Optional[dict]  # Store chart data when chart tool is called
    retry_count: Optional[int]  # Number of validator retries attempted in this run
    tool_last_fingerprint: Optional[str]  # Last tool-call fingerprint (for loop detection)
    tool_repeat_count: Optional[int]  # Consecutive repeats of the same fingerprint
    tool_loop_detected: Optional[bool]  # True if loop guard stops the graph
    tool_loop_action: Optional[Literal["continue", "replan", "stop"]]  # Next step after tool loop guard
    tool_loop_details: Optional[dict]  # Details about repeated tool calls / last SQL, etc.
    tool_loop_replan_count: Optional[int]  # Number of replans attempted after loop detection
    total_tool_calls: Optional[int]  # Total number of tool calls in this run (for global limit)
    tool_call_counts: Optional[Dict[str, int]]  # Count of calls per tool name (for detecting repeated tool abuse)
    # Budget tracking for preventing infinite loops
    budget: Optional[Dict[str, Any]]  # Serialized AgentBudget state
    budget_exceeded: Optional[bool]  # True if budget limits were hit
    budget_status: Optional[Dict[str, Any]]  # Current budget status for debugging
    detected_language: Optional[Literal["it", "en"]]  # Detected language from user message ("it" for Italian, "en" for English)


# Dataset presets configuration
DATASET_PRESETS = {
    "vienna": {
        "db_path": "dataset/BAUMKATOGD.db",
        "table_name": "baumkatogd",
        "description": """Dataset degli alberi di Vienna (BAUMKATOGD) con 229.298 alberi.
Colonne principali:
- objectid: ID univoco albero
- district: Numero distretto (1-23)
- genus_species: Nome specie (es. "Acer platanoides")
- plant_year: Anno di piantumazione
- trunk_circumference: Circonferenza tronco in cm
- tree_height: Categoria altezza (codificata)
- crown_diameter: Categoria diametro chioma (codificata)
- object_street: Nome via
Calcoli derivati: DBH = trunk_circumference / π, Età = anno_corrente - plant_year"""
    },
    "milano": {
        "db_path": "dataset/dataset_milano.db",
        "table_name": "milano_trees",
        "description": """Dataset degli alberi di Milano con 251.165 alberi.
Colonne principali:
- _id: ID univoco albero
- district: Numero municipio (1-9)
- genere: Genere botanico (es. "Prunus", "Acer")
- specie: Specie botanica (es. "cerasifera", "platanoides")
- varieta: Varietà (es. "Pissardii")
- genus_species: Nome completo specie (genere + specie)
- trunk_diameter_cm: Diametro tronco in cm (NON circonferenza!)
- crown_diameter_m: Diametro chioma in metri
- height_m: Altezza in metri
- street: Nome via/località
- plant_year: Anno di piantumazione
- longitude, latitude: Coordinate GPS
Nota: trunk_diameter_cm è già il diametro, NON la circonferenza (a differenza di Vienna)"""
    }
}

