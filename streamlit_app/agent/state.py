"""State definitions and dataset configurations for the Tree Evaluator Agent."""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# ============================================================================
# Grouped State TypedDicts for better organization
# ============================================================================


class QueryPlanState(TypedDict, total=False):
    """Query planning state - consolidates query optimization fields."""

    optimized_query: str  # Query after optimization
    tasks: List[str]  # List of tasks extracted from query
    tool_plan: List[Dict[str, Any]]  # Planned tools: [{"task": str, "tools": [str], "reason": str}]
    available_tools_summary: str  # Summary of available tools for planning


class LoopState(TypedDict, total=False):
    """Tool loop detection state - consolidates loop tracking fields."""

    last_fingerprint: str  # Last tool-call fingerprint
    repeat_count: int  # Consecutive repeats of the same fingerprint
    detected: bool  # True if loop guard stops the graph
    action: Literal["continue", "replan", "stop"]  # Next step after detection
    details: Dict[str, Any]  # Details about repeated tool calls
    replan_count: int  # Number of replans attempted
    total_calls: int  # Total number of tool calls
    call_counts: Dict[str, int]  # Count of calls per tool name


class BudgetState(TypedDict, total=False):
    """Budget tracking state - consolidates budget fields."""

    data: Dict[str, Any]  # Serialized AgentBudget state
    exceeded: bool  # True if budget limits were hit
    status: Dict[str, Any]  # Current budget status for debugging


class ContextState(TypedDict, total=False):
    """Context management state - consolidates context fields."""

    summary: str  # Summary of important context
    message_count: int  # Track conversation length
    retry_count: int  # Number of validator retries attempted


# ============================================================================
# Main AgentState - supports both legacy flat fields and new grouped fields
# ============================================================================


class AgentState(TypedDict, total=False):
    """State for the LangGraph agent.

    This state supports both:
    - Legacy flat fields (for backwards compatibility)
    - New grouped fields (for cleaner organization)

    New code should prefer grouped fields when possible.
    """

    # Core required fields
    messages: Annotated[Sequence[BaseMessage], add_messages]
    detected_language: Literal["it", "en"]

    # Output fields
    validation_result: Dict[str, Any]
    chart_data: Dict[str, Any]

    # ========== GROUPED FIELDS (preferred for new code) ==========
    query_plan: QueryPlanState
    loop_state: LoopState
    budget_state: BudgetState
    context_state: ContextState

    # ========== LEGACY FLAT FIELDS (for backwards compatibility) ==========
    # Query planning (use query_plan instead)
    optimized_query: str
    tasks: List[str]
    tool_plan: List[Dict[str, Any]]
    available_tools_summary: str

    # Context (use context_state instead)
    context_summary: str
    message_count: int
    retry_count: int

    # Loop detection (use loop_state instead)
    tool_last_fingerprint: str
    tool_repeat_count: int
    tool_loop_detected: bool
    tool_loop_action: Literal["continue", "replan", "stop"]
    tool_loop_details: Dict[str, Any]
    tool_loop_replan_count: int
    total_tool_calls: int
    tool_call_counts: Dict[str, int]

    # Budget (use budget_state instead)
    budget: Dict[str, Any]
    budget_exceeded: bool
    budget_status: Dict[str, Any]


# ============================================================================
# Helper functions for state access
# ============================================================================


def get_loop_state(state: AgentState) -> LoopState:
    """Get loop state from AgentState, supporting both grouped and legacy fields.

    Args:
        state: The agent state.

    Returns:
        LoopState with values from grouped or legacy fields.
    """
    # Prefer grouped field if present
    if "loop_state" in state and state["loop_state"]:
        return state["loop_state"]

    # Fall back to legacy fields
    return LoopState(
        last_fingerprint=state.get("tool_last_fingerprint", ""),
        repeat_count=state.get("tool_repeat_count", 0),
        detected=state.get("tool_loop_detected", False),
        action=state.get("tool_loop_action", "continue"),
        details=state.get("tool_loop_details", {}),
        replan_count=state.get("tool_loop_replan_count", 0),
        total_calls=state.get("total_tool_calls", 0),
        call_counts=state.get("tool_call_counts", {}),
    )


def get_budget_state(state: AgentState) -> BudgetState:
    """Get budget state from AgentState, supporting both grouped and legacy fields.

    Args:
        state: The agent state.

    Returns:
        BudgetState with values from grouped or legacy fields.
    """
    # Prefer grouped field if present
    if "budget_state" in state and state["budget_state"]:
        return state["budget_state"]

    # Fall back to legacy fields
    return BudgetState(
        data=state.get("budget", {}),
        exceeded=state.get("budget_exceeded", False),
        status=state.get("budget_status", {}),
    )


def get_query_plan_state(state: AgentState) -> QueryPlanState:
    """Get query plan state from AgentState, supporting both grouped and legacy fields.

    Args:
        state: The agent state.

    Returns:
        QueryPlanState with values from grouped or legacy fields.
    """
    # Prefer grouped field if present
    if "query_plan" in state and state["query_plan"]:
        return state["query_plan"]

    # Fall back to legacy fields
    return QueryPlanState(
        optimized_query=state.get("optimized_query", ""),
        tasks=state.get("tasks", []),
        tool_plan=state.get("tool_plan", []),
        available_tools_summary=state.get("available_tools_summary", ""),
    )


# Dataset presets configuration - loaded dynamically from presets.json
def _load_dataset_presets() -> Dict[str, Any]:
    """Load dataset presets from JSON file.

    Falls back to hardcoded defaults if file not found.
    """
    import json
    from pathlib import Path

    presets_path = Path(__file__).parent.parent.parent / "dataset" / "presets.json"

    try:
        with open(presets_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Filter out internal keys (starting with _) and flatten descriptions
        presets = {}
        for key, value in data.items():
            if not key.startswith("_"):
                preset = dict(value)
                # Use Italian description by default for backward compatibility
                if isinstance(preset.get("description"), dict):
                    preset["description"] = preset["description"].get("it", "")
                presets[key] = preset

        return presets

    except (FileNotFoundError, json.JSONDecodeError):
        # Fallback to hardcoded defaults
        return {
            "vienna": {
                "db_path": "dataset/BAUMKATOGD.db",
                "table_name": "baumkatogd",
                "description": "Dataset degli alberi di Vienna (BAUMKATOGD)",
            },
            "milano": {
                "db_path": "dataset/dataset_milano.db",
                "table_name": "milano_trees",
                "description": "Dataset degli alberi di Milano",
            }
        }


# Load presets at module import (cached)
DATASET_PRESETS = _load_dataset_presets()

