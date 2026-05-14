from __future__ import annotations

from typing import Any, Dict, Optional

from streamlit_app.agent.streaming_handler import StreamingHandler
from streamlit_app.agent.translations import get_translation


def process_stream_event(
    node_name: str,
    node_output: Any,
    language: str,
    msg_count: int,
) -> Optional[Dict]:
    """Convert a LangGraph node update into a UI streaming event."""
    if node_name == "language_detector":
        return StreamingHandler.handle_language_detector_event(node_output, language)
    if node_name == "context_manager":
        return StreamingHandler.handle_context_manager_event(node_output, msg_count, language)
    if node_name == "query_optimizer":
        return StreamingHandler.handle_query_optimizer_event(node_output, language)
    if node_name == "agent":
        return StreamingHandler.handle_agent_event(node_output, language)
    if node_name == "tools":
        result, chart_json, map_json = StreamingHandler.handle_tools_event(node_output, language)
        if chart_json:
            return {"type": "chart_data", "data": chart_json}
        if map_json:
            return {"type": "map_data", "data": map_json}
        return result
    if node_name == "budget_check":
        return StreamingHandler.handle_budget_check_event(node_output, language)
    if node_name == "tool_loop_guard":
        return StreamingHandler.handle_tool_loop_guard_event(node_output, language)
    if node_name == "tool_loop_replanner":
        return {
            "type": "reasoning",
            "content": (
                f"{get_translation('replanning', language)}\n\n"
                f"{get_translation('reformulating_step', language)}\n"
            ),
        }
    if node_name == "validator":
        result, _ = StreamingHandler.handle_validator_event(node_output, 0, 1, language)
        return result

    return None
