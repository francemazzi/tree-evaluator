"""Adapter: bridges FailureTracker into the LangGraph tool_loop_guard node.

Maintains the same interface as ToolLoopManager (check_for_loops, create_replan_prompt)
so core.py requires only an import swap to switch implementations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Sequence

from langchain_core.messages import AIMessage, BaseMessage

from streamlit_app.agent.failure_tracker import FailureTracker, LoopDecision

logger = logging.getLogger(__name__)


class FailureTrackerAdapter:
    """Adapts FailureTracker's API to produce AgentState-compatible dicts.

    Drop-in replacement for ToolLoopManager. The graph builder reads
    ``state["tool_loop_action"]`` to decide the next edge — this adapter
    writes exactly the same keys.
    """

    def __init__(self) -> None:
        self._tracker = FailureTracker()

    def check_for_loops(
        self, messages: Sequence[BaseMessage], state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check for loops using FailureTracker, returning AgentState-shaped dict."""
        # Extract the most recent tool calls from the last AIMessage
        pending_tool_calls: list[dict] = []
        for msg in reversed(list(messages)):
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                pending_tool_calls = msg.tool_calls
                break

        # Feed each pending tool call to FailureTracker
        decision = LoopDecision(action="continue")
        for tc in pending_tool_calls:
            tool_name = tc.get("name", "unknown")
            args = tc.get("args", {})
            decision = self._tracker.record_and_check(tool_name, args)
            if decision.action != "continue":
                logger.info(
                    "FailureTracker: %s on tool=%s (count=%d, reason=%s)",
                    decision.action, decision.tool_name,
                    decision.call_count, decision.reason,
                )
                break

        # Map LoopDecision → AgentState dict (same shape as ToolLoopManager)
        is_loop = decision.action != "continue"
        return {
            "tool_loop_detected": is_loop,
            "tool_loop_action": decision.action,
            "tool_loop_details": {
                "reason": decision.reason,
                "tool_name": decision.tool_name,
                "call_count": decision.call_count,
            } if is_loop else None,
            "tool_call_counts": self._tracker.get_counts(),
            "total_tool_calls": self._tracker.get_total_calls(),
        }

    def create_replan_prompt(
        self, state: Dict[str, Any], messages: Sequence[BaseMessage]
    ) -> Any:
        """Delegate replan prompt generation to ToolLoopManager.

        ToolLoopManager.create_replan_prompt() only reads state and messages
        to build a self-reflection prompt — it doesn't depend on the loop
        detection internals, so it's safe to reuse.
        """
        from streamlit_app.agent.tool_guard import ToolLoopManager

        return ToolLoopManager().create_replan_prompt(state, messages)
