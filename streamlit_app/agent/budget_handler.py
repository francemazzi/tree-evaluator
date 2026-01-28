"""Budget handling and summary generation for Tree Evaluator Agent.

This module manages budget constraints and generates conversational summaries
when budget limits are reached.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, TYPE_CHECKING

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from streamlit_app.agent.budget import AgentBudget, BudgetAwareToolGuard
from streamlit_app.agent.extraction import DataExtractor
from streamlit_app.agent.state import AgentState

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


def get_detected_language(state: AgentState, default: str = "it") -> str:
    """Get the detected language from state with validation.

    Args:
        state: Agent state dictionary
        default: Default language if not found or invalid

    Returns:
        Valid language code ("it" or "en")
    """
    detected = state.get("detected_language", default)
    if detected not in ["it", "en"]:
        return default
    return detected


class BudgetHandler:
    """Handles budget checking and conversational summary generation."""

    def __init__(
        self,
        fallback_model: str,
        create_chat_without_tools: Callable[[str, float], "BaseChatModel"],
    ) -> None:
        """Initialize the budget handler.

        Args:
            fallback_model: Model name for summary generation
            create_chat_without_tools: Factory function to create plain chat model
        """
        self._fallback_model = fallback_model
        self._create_chat_without_tools = create_chat_without_tools
        self._extractor = DataExtractor()

    def check_budget(self, state: AgentState) -> dict:
        """Check budget constraints before tool execution."""
        messages = state.get("messages") or []
        detected_language = get_detected_language(state)

        # Restore or create budget
        budget_data = state.get("budget")
        if budget_data:
            budget = AgentBudget.from_dict(budget_data)
        else:
            budget = AgentBudget()

        # Create guard and check
        guard = BudgetAwareToolGuard(budget)
        can_proceed, error_msg, status = guard.check_before_tools(messages, detected_language)

        if not can_proceed:
            # Budget exceeded - generate conversational response
            conversational_response = self._generate_conversational_summary(
                messages, status, detected_language
            )
            return {
                "messages": [AIMessage(content=conversational_response)],
                "budget": budget.to_dict(),
                "budget_exceeded": True,
                "budget_status": status,
            }

        return {
            "budget": budget.to_dict(),
            "budget_exceeded": False,
            "budget_status": status,
        }

    def _generate_conversational_summary(
        self,
        messages: Sequence[BaseMessage],
        budget_status: Dict[str, Any],
        language: Literal["it", "en"] = "it",
    ) -> str:
        """Generate a conversational response summarizing results collected so far."""
        # Extract user's original question
        user_question = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                user_question = msg.content

        # Extract all tool results
        tool_results = self._extractor.extract_tool_results(messages)

        if not tool_results:
            return self._get_no_data_message(language)

        tools_used = list(set(r["tool"] for r in tool_results))

        # Try to generate a conversational summary using LLM
        try:
            return self._generate_llm_summary(
                user_question, tool_results, tools_used, language
            )
        except Exception as e:
            logger.warning(f"Failed to generate LLM summary: {e}")
            return self._generate_fallback_summary(
                user_question, tool_results, tools_used, language
            )

    def _get_no_data_message(self, language: str) -> str:
        """Get message when no data has been collected."""
        if language == "en":
            return (
                "I haven't collected enough data to answer completely yet.\n\n"
                "**Suggestion:** Try rephrasing the question more specifically.\n\n"
                "Tools used: None"
            )
        return (
            "Non ho ancora raccolto abbastanza dati per rispondere completamente.\n\n"
            "**Suggerimento:** Prova a riformulare la domanda in modo più specifico.\n\n"
            "Tool utilizzati: Nessuno"
        )

    def _generate_llm_summary(
        self,
        user_question: str,
        tool_results: List[Dict],
        tools_used: List[str],
        language: str,
    ) -> str:
        """Generate summary using LLM."""
        summary_llm = self._create_chat_without_tools(
            model=self._fallback_model, temperature=0.7
        )

        # Format results for summary
        results_text = self._format_tool_results_for_summary(tool_results)

        if language == "en":
            summary_prompt = self._get_english_summary_prompt(
                user_question, results_text, tools_used
            )
            tool_citation_prefix = "Tools used:"
        else:
            summary_prompt = self._get_italian_summary_prompt(
                user_question, results_text, tools_used
            )
            tool_citation_prefix = "Tool utilizzati:"

        response = summary_llm.invoke([HumanMessage(content=summary_prompt)])
        summary = response.content.strip()

        # Ensure tool citation is present
        if tool_citation_prefix not in summary:
            summary += f"\n\n{tool_citation_prefix} {', '.join(tools_used)}"

        return summary

    def _get_english_summary_prompt(
        self, user_question: str, results_text: str, tools_used: List[str]
    ) -> str:
        """Get the summary generation prompt in English."""
        return f"""Generate a conversational and friendly response in English.

User's question: {user_question}

Results collected from tools:
{results_text}

Tools used: {', '.join(tools_used)}

INSTRUCTIONS:
1. Respond naturally and conversationally
2. Use the provided data to answer the question
3. If data is incomplete, explain what you found and what's missing
4. Include appropriate units of measurement
5. End with "Tools used: [tool list]"
6. DO NOT invent data that is not in the results

Response:"""

    def _get_italian_summary_prompt(
        self, user_question: str, results_text: str, tools_used: List[str]
    ) -> str:
        """Get the summary generation prompt in Italian."""
        return f"""Genera una risposta conversazionale e amichevole in italiano.

Domanda dell'utente: {user_question}

Risultati raccolti dai tool:
{results_text}

Tool utilizzati: {', '.join(tools_used)}

ISTRUZIONI:
1. Rispondi in modo naturale e conversazionale
2. Usa i dati forniti per rispondere alla domanda
3. Se i dati sono incompleti, spiega cosa hai trovato e cosa manca
4. Includi le unità di misura appropriate
5. Termina con "Tool utilizzati: [lista tool]"
6. NON inventare dati che non sono nei risultati

Risposta:"""

    def _generate_fallback_summary(
        self,
        user_question: str,
        tool_results: List[Dict],
        tools_used: List[str],
        language: str,
    ) -> str:
        """Generate a basic summary without LLM."""
        if language == "en":
            summary = f"**Results collected for:** {user_question}\n\n"
            for tr in tool_results[:3]:
                result = tr.get("result", {})
                if isinstance(result, dict):
                    if "co2_stock_t" in result:
                        summary += f"- **CO2 stock:** {result.get('co2_stock_t', 'N/A')} t CO2\n"
                    if "total_biomass_t" in result:
                        summary += f"- **Total biomass:** {result.get('total_biomass_t', 'N/A')} t\n"
            summary += f"\n\nTools used: {', '.join(tools_used)}"
        else:
            summary = f"**Risultati raccolti per:** {user_question}\n\n"
            for tr in tool_results[:3]:
                result = tr.get("result", {})
                if isinstance(result, dict):
                    if "co2_stock_t" in result:
                        summary += f"- **CO2 stock:** {result.get('co2_stock_t', 'N/A')} t CO2\n"
                    if "total_biomass_t" in result:
                        summary += f"- **Biomassa totale:** {result.get('total_biomass_t', 'N/A')} t\n"
            summary += f"\n\nTool utilizzati: {', '.join(tools_used)}"

        return summary

    def _format_tool_results_for_summary(self, tool_results: List[Dict]) -> str:
        """Format tool results for summary generation."""
        results_text = ""
        for tr in tool_results[:5]:
            result = tr.get("result", {})
            if isinstance(result, dict):
                if "co2_stock_t" in result:
                    results_text += f"- CO2 stock: {result.get('co2_stock_t', 'N/A')} tonnellate\n"
                if "total_biomass_t" in result:
                    results_text += f"- Biomassa totale: {result.get('total_biomass_t', 'N/A')} tonnellate\n"
                if "agb_t" in result:
                    results_text += f"- Biomassa epigea: {result.get('agb_t', 'N/A')} tonnellate\n"
                if "results" in result and isinstance(result["results"], list):
                    results_text += f"- Trovati {len(result['results'])} risultati nel dataset\n"
                if "result" in result:
                    results_text += f"- Valore: {result.get('result', 'N/A')}\n"

        if not results_text:
            results_text = json.dumps(
                tool_results[0].get("result", {}), indent=2, ensure_ascii=False
            )[:500]

        return results_text
