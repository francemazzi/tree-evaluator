"""LangGraph workflow builder for Tree Evaluator Agent.

This module handles the construction of the LangGraph workflow,
including node definitions and edge routing logic.
"""

from __future__ import annotations

from typing import Any, Callable, List, Literal, Optional, TYPE_CHECKING

from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_node import ToolNode

from streamlit_app.agent.state import AgentState

if TYPE_CHECKING:
    pass


class GraphBuilder:
    """Builds the LangGraph workflow for the Tree Evaluator Agent."""

    def __init__(self, tools: List[Any]) -> None:
        """Initialize the graph builder.

        Args:
            tools: List of tool instances to be used in the workflow
        """
        self._tools = tools

    def build(
        self,
        manage_context: Callable[[AgentState], dict],
        optimize_query: Callable[[AgentState], dict],
        check_budget: Callable[[AgentState], dict],
        call_model: Callable[[AgentState], dict],
        guard_tool_loop: Callable[[AgentState], dict],
        replan_after_tool_loop: Callable[[AgentState], dict],
        validate_response: Callable[[AgentState], dict],
        increment_retry_count: Callable[[AgentState], dict],
        run_tools: Optional[Callable[[AgentState], dict]] = None,
    ) -> StateGraph:
        """Build the LangGraph workflow with all nodes and edges.

        Args:
            manage_context: Node function for context management
            optimize_query: Node function for query optimization
            check_budget: Node function for budget checking
            call_model: Node function for LLM calls
            guard_tool_loop: Node function for loop detection
            replan_after_tool_loop: Node function for replanning
            validate_response: Node function for response validation
            increment_retry_count: Node function for retry counting
            run_tools: Optional replacement for LangGraph ToolNode, used for tracing

        Returns:
            Compiled StateGraph workflow
        """
        workflow = StateGraph(AgentState)

        # Define nodes
        workflow.add_node("context_manager", manage_context)
        workflow.add_node("query_optimizer", optimize_query)
        workflow.add_node("budget_check", check_budget)
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", run_tools or ToolNode(self._tools))
        workflow.add_node("tool_loop_guard", guard_tool_loop)
        workflow.add_node("tool_loop_replanner", replan_after_tool_loop)
        workflow.add_node("validator", validate_response)
        workflow.add_node("retry_counter", increment_retry_count)

        # Set entry point
        workflow.set_entry_point("context_manager")

        # Define edges
        workflow.add_edge("context_manager", "query_optimizer")
        workflow.add_edge("query_optimizer", "agent")

        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {"continue": "budget_check", "validate": "validator"},
        )

        workflow.add_conditional_edges(
            "budget_check",
            self._should_continue_after_budget,
            {"continue": "tools", "stop": END},
        )

        workflow.add_edge("tools", "tool_loop_guard")
        workflow.add_conditional_edges(
            "tool_loop_guard",
            self._should_continue_after_tool_guard,
            {"continue": "agent", "replan": "tool_loop_replanner", "stop": END},
        )
        workflow.add_edge("tool_loop_replanner", "agent")

        workflow.add_conditional_edges(
            "validator",
            self._should_retry,
            {"complete": END, "retry": "retry_counter"},
        )

        workflow.add_edge("retry_counter", "agent")

        return workflow.compile()

    # ===== Conditional Edge Methods =====

    def _should_continue(self, state: AgentState) -> Literal["continue", "validate"]:
        """Determine if we should continue to tools or validate response."""
        messages = state["messages"]
        last_message = messages[-1]

        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "continue"
        return "validate"

    def _should_continue_after_budget(self, state: AgentState) -> Literal["continue", "stop"]:
        """Determine if we should continue to tools or stop due to budget."""
        if state.get("budget_exceeded", False):
            return "stop"
        return "continue"

    def _should_continue_after_tool_guard(
        self, state: AgentState
    ) -> Literal["continue", "replan", "stop"]:
        """Determine action after tool loop guard."""
        action = state.get("tool_loop_action") or "continue"
        if action in ("continue", "replan", "stop"):
            return action
        return "continue"

    def _should_retry(self, state: AgentState) -> Literal["complete", "retry"]:
        """Determine if we should retry or complete based on validation."""
        validation_result = state.get("validation_result", {})
        retry_count = int(state.get("retry_count") or 0)
        max_retries = 1

        is_complete = validation_result.get("is_complete", True)

        if is_complete:
            return "complete"
        if retry_count >= max_retries:
            return "complete"
        return "retry"
