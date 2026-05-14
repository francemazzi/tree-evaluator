from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import BaseMessage

from streamlit_app.agent.budget import AgentBudget


def create_initial_state(
    messages: List[BaseMessage],
    interface_language: str,
) -> Dict[str, Any]:
    """Create the initial state for graph invocation."""
    initial_budget = AgentBudget()
    return {
        "messages": messages,
        "retry_count": 0,
        "tool_last_fingerprint": None,
        "tool_repeat_count": 0,
        "tool_loop_detected": False,
        "tool_loop_action": "continue",
        "tool_loop_details": None,
        "tool_loop_replan_count": 0,
        "total_tool_calls": 0,
        "tool_call_counts": {},
        "budget": initial_budget.to_dict(),
        "budget_exceeded": False,
        "budget_status": None,
        "detected_language": interface_language,
        "tool_plan": None,
        "available_tools_summary": None,
    }
