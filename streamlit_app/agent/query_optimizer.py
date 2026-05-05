"""Query optimization and planning for Tree Evaluator Agent.

This module handles query analysis, task planning, and tool selection
to optimize agent responses.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from streamlit_app.agent.config_loader import get_config_loader
from streamlit_app.agent.state import AgentState

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


class QueryOptimizer:
    """Optimizes user queries and plans tool execution."""

    def __init__(
        self,
        interface_language: str,
        fallback_model: str,
        create_chat_without_tools: Callable[[str, float], "BaseChatModel"],
        dynamic_tools_summary: Optional[List[dict]] = None,
    ) -> None:
        """Initialize the query optimizer.

        Args:
            interface_language: Language for optimization ("it" or "en")
            fallback_model: Model name for optimization calls
            create_chat_without_tools: Factory function to create plain chat model
            dynamic_tools_summary: Optional summary of dynamically loaded tools
        """
        self._interface_language = interface_language
        self._fallback_model = fallback_model
        self._create_chat_without_tools = create_chat_without_tools
        self._dynamic_tools_summary = dynamic_tools_summary or []
        self._cached_tools_summary: Optional[str] = None

    def set_dynamic_tools_summary(self, summary: List[dict]) -> None:
        """Update the dynamic tools summary."""
        self._dynamic_tools_summary = summary
        self._cached_tools_summary = None  # Invalidate cache

    def optimize_query(self, state: AgentState) -> dict:
        """Optimize user query, break it into tasks, and plan which tools to use."""
        messages = state["messages"]

        # Get the last user message
        last_user_msg = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_user_msg = msg.content
                break

        if not last_user_msg:
            return {
                "optimized_query": None,
                "tasks": [],
                "tool_plan": [],
                "detected_language": self._interface_language,
            }

        # Use the configured interface language from user settings
        detected_language = self._interface_language

        # Build available tools summary for planning
        tools_summary = self._build_tools_summary_for_planning()

        # For simple/short queries, skip LLM optimization and do basic planning
        if len(last_user_msg) < 100 or self._is_simple_query(last_user_msg):
            simple_plan = self._create_simple_tool_plan(last_user_msg, detected_language)
            return {
                "optimized_query": last_user_msg,
                "tasks": [last_user_msg],
                "tool_plan": simple_plan,
                "available_tools_summary": tools_summary,
                "detected_language": detected_language,
            }

        # Use LLM to optimize query and plan tools
        return self._optimize_with_llm(
            last_user_msg, detected_language, tools_summary
        )

    def _optimize_with_llm(
        self, query: str, language: str, tools_summary: str
    ) -> dict:
        """Use LLM to optimize query and create execution plan."""
        if language == "en":
            optimizer_prompt = self._get_english_optimizer_prompt(query, tools_summary)
            optimization_msg_template = """Optimized query: {optimized_query}

Execution Plan:
{plan_text}"""
        else:
            optimizer_prompt = self._get_italian_optimizer_prompt(query, tools_summary)
            optimization_msg_template = """Query ottimizzata: {optimized_query}

Piano di esecuzione:
{plan_text}"""

        try:
            optimizer_llm = self._create_chat_without_tools(
                model=self._fallback_model, temperature=0.7
            )
            response = optimizer_llm.invoke([HumanMessage(content=optimizer_prompt)])
            response_text = response.content.strip()

            # Extract JSON from markdown code blocks if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            optimization_result = json.loads(response_text)
            optimized_query = optimization_result.get("optimized_query", query)
            tasks = optimization_result.get("tasks", [])
            tool_plan = optimization_result.get("tool_plan", [])

            # Format plan for display
            plan_text = self._format_tool_plan(tasks, tool_plan, language)
            optimization_msg = SystemMessage(
                content=optimization_msg_template.format(
                    optimized_query=optimized_query, plan_text=plan_text
                )
            )

            return {
                "messages": [optimization_msg],
                "optimized_query": optimized_query,
                "tasks": tasks,
                "tool_plan": tool_plan,
                "available_tools_summary": tools_summary,
                "detected_language": language,
            }

        except Exception as e:
            logger.warning(f"Query optimizer failed: {e}, falling back to simple plan")
            simple_plan = self._create_simple_tool_plan(query, language)
            return {
                "optimized_query": query,
                "tasks": [query],
                "tool_plan": simple_plan,
                "available_tools_summary": tools_summary,
                "detected_language": language,
            }

    def _get_english_optimizer_prompt(self, query: str, tools_summary: str) -> str:
        """Get the optimization prompt in English."""
        return f"""Analyze the following question and create a task list with tool selection.

Question: {query}

AVAILABLE TOOLS:
{tools_summary}

Respond ONLY in JSON format:
{{
    "optimized_query": "briefly rephrased question",
    "tasks": ["task 1", "task 2"],
    "tool_plan": [
        {{"task": "task 1", "tools": ["tool_name_1"], "reason": "why this tool"}},
        {{"task": "task 2", "tools": ["tool_name_2", "tool_name_3"], "reason": "why these tools"}}
    ]
}}

RULES:
- Maximum 3 tasks
- Short and direct tasks
- For each task, select 1-3 most appropriate tools from the available list
- Provide a brief reason for tool selection
- If the question is already clear, return the original question with 1 task"""

    def _get_italian_optimizer_prompt(self, query: str, tools_summary: str) -> str:
        """Get the optimization prompt in Italian."""
        return f"""Analizza la seguente domanda e crea una lista di task con selezione dei tool.

Domanda: {query}

TOOL DISPONIBILI:
{tools_summary}

Rispondi SOLO in formato JSON:
{{
    "optimized_query": "domanda riformulata brevemente",
    "tasks": ["task 1", "task 2"],
    "tool_plan": [
        {{"task": "task 1", "tools": ["nome_tool_1"], "reason": "perché questo tool"}},
        {{"task": "task 2", "tools": ["nome_tool_2", "nome_tool_3"], "reason": "perché questi tool"}}
    ]
}}

REGOLE:
- Massimo 3 task
- Task brevi e diretti
- Per ogni task, seleziona 1-3 tool più appropriati dalla lista disponibile
- Fornisci una breve motivazione per la selezione del tool
- Se la domanda è già chiara, restituisci la domanda originale con 1 task"""

    def _is_simple_query(self, query: str) -> bool:
        """Detect simple queries that don't require LLM optimization.

        Uses patterns loaded dynamically from presets.json.
        """
        query_lower = query.lower()
        config = get_config_loader()

        # Load patterns dynamically from config for both languages
        patterns_it = config.get_simple_query_patterns("it")
        patterns_en = config.get_simple_query_patterns("en")
        all_patterns = patterns_it + patterns_en

        # Also check if query matches any tool keywords
        matching_tools = config.match_keywords_to_tools(query)

        return any(pattern in query_lower for pattern in all_patterns) or len(matching_tools) > 0

    def _build_tools_summary_for_planning(self) -> str:
        """Build a summary of all available tools for task planning.

        Uses caching to avoid rebuilding the summary on every request.
        """
        if self._cached_tools_summary:
            return self._cached_tools_summary

        config = get_config_loader()
        language = self._interface_language

        summary_lines = []

        # Static tools summary - loaded dynamically from config
        static_tools = config.get_static_tools_metadata(language)
        for tool in static_tools:
            summary_lines.append(f"- {tool['name']}: {tool['description']}")

        # Dynamic tools summary (from tools.json)
        if self._dynamic_tools_summary:
            summary_lines.append("\nDynamic Formula Tools:")
            for tool_info in self._dynamic_tools_summary:
                summary_lines.append(
                    f"- {tool_info['name']}: {tool_info['title']} ({tool_info['formula']})"
                )

        self._cached_tools_summary = "\n".join(summary_lines)
        return self._cached_tools_summary

    def _create_simple_tool_plan(self, query: str, language: str) -> List[Dict]:
        """Create a simple tool plan for short queries.

        Uses dynamic keyword matching from presets.json.
        """
        config = get_config_loader()

        # Match keywords to tools dynamically
        tools = config.match_keywords_to_tools(query)

        # Also check dynamic tools keywords
        if self._dynamic_tools_summary:
            query_lower = query.lower()
            for tool_info in self._dynamic_tools_summary:
                keywords = tool_info.get("keywords", [])
                for kw in keywords:
                    if kw.lower() in query_lower and tool_info["name"] not in tools:
                        tools.append(tool_info["name"])
                        break

        # Default to the main tree dataset query if no specific match
        if not tools:
            tools.append("query_tree_dataset")

        reason_it = "Tool suggerito in base alla domanda"
        reason_en = "Suggested tool based on the query"

        return [
            {
                "task": query,
                "tools": tools,
                "reason": reason_en if language == "en" else reason_it,
            }
        ]

    def _format_tool_plan(
        self, tasks: List[str], tool_plan: List[Dict], language: str
    ) -> str:
        """Format the tool plan for display."""
        lines = []

        for i, task in enumerate(tasks):
            plan_item = tool_plan[i] if i < len(tool_plan) else {}
            tools = plan_item.get("tools", [])
            reason = plan_item.get("reason", "")

            if language == "en":
                lines.append(f"{i+1}. Task: {task}")
                lines.append(f"   Tools: {', '.join(tools) if tools else 'auto-select'}")
                if reason:
                    lines.append(f"   Reason: {reason}")
            else:
                lines.append(f"{i+1}. Task: {task}")
                lines.append(
                    f"   Tool: {', '.join(tools) if tools else 'selezione automatica'}"
                )
                if reason:
                    lines.append(f"   Motivo: {reason}")

        return "\n".join(lines)
