"""Tree Evaluator Agent - Main orchestrator for tree analysis tools.

This module provides the main TreeEvaluatorAgent class that coordinates
multiple specialized tools for tree analysis, CO2 calculations, and dataset queries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from streamlit_app.agent.budget import AgentBudget, BudgetAwareToolGuard
from streamlit_app.agent.config_loader import get_config_loader
from streamlit_app.agent.context_manager import ConversationContextManager
from streamlit_app.agent.extraction import DataExtractor
from streamlit_app.agent.formatting import ItalianNumberFormatter
from streamlit_app.agent.prompts import SystemPrompts
from streamlit_app.agent.response_builder import ResponseBuilder
from streamlit_app.agent.state import AgentState, DATASET_PRESETS
from streamlit_app.agent.streaming_handler import StreamingHandler
from streamlit_app.agent.tool_guard import ToolLoopManager
from streamlit_app.llm.factory import LlmFactory, LlmProvider, LlmSettings, LlmSettingsReader
from streamlit_app.tools.allometric_relation_tool import AllometricRelationTool
from streamlit_app.tools.carbon_content_tool import CarbonContentTool
from streamlit_app.tools.chart_tool import ChartGenerationTool
from streamlit_app.tools.co2_aggregate_tool import CO2AggregateTool
from streamlit_app.tools.co2_tool import CO2CalculationTool
from streamlit_app.tools.dataset_tool import DatasetQueryTool
from streamlit_app.tools.environment_tool import EnvironmentEstimationTool
from streamlit_app.tools.export_tool import ExportDataTool
from streamlit_app.tools.general_volume_tool import GeneralVolumeTool
from streamlit_app.tools.heyer_volume_tool import HeyerVolumeTool
from streamlit_app.tools.language_tool import LanguageDetectionTool, LanguageTranslationTool
from streamlit_app.tools.leaf_biomass_tool import LeafBiomassTool
from streamlit_app.tools.log_allometric_tool import LogAllometricTool
from streamlit_app.tools.log_fuel_biomass_tool import LogFuelBiomassTool
from streamlit_app.tools.map_tool import MapGenerationTool
from streamlit_app.tools.model_error_tool import ModelErrorTool
from streamlit_app.tools.paper_search_tool import PaperSearchTool
from streamlit_app.tools.root_biomass_tool import RootBiomassTool
from streamlit_app.tools.simplified_volume_tool import SimplifiedVolumeTool
from streamlit_app.tools.species_list_tool import SpeciesListQueryTool
from streamlit_app.tools.stem_biomass_tool import StemBiomassTool
from streamlit_app.tools.total_biomass_tool import TotalBiomassTool
from streamlit_app.tools.dynamic_tool_loader import DynamicToolLoader, get_tools_for_planning

# Load environment variables
load_dotenv()


class TreeEvaluatorAgent:
    """LangGraph-based agent that orchestrates tree evaluation tools."""

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        provider: Optional[str] = None,
        openai_chat_model: Optional[str] = None,
        openai_embedding_model: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
        ollama_chat_model: Optional[str] = None,
        ollama_embedding_model: Optional[str] = None,
        custom_db_path: Optional[Path] = None,
        custom_table_name: Optional[str] = None,
        data_description: str = "",
        dataset_preset: str = "vienna",
        interface_language: str = "it"
    ) -> None:
        """Initialize the agent with tools and LLM.

        Args:
            openai_api_key: OpenAI API key
            provider: LLM provider (openai/ollama)
            openai_chat_model: OpenAI chat model name
            openai_embedding_model: OpenAI embedding model name
            ollama_base_url: Ollama base URL
            ollama_chat_model: Ollama chat model name
            ollama_embedding_model: Ollama embedding model name
            custom_db_path: Optional path to custom SQLite database
            custom_table_name: Optional custom table name in the database
            data_description: Optional description of the data for context
            dataset_preset: Preset dataset to use ("vienna", "milano")
            interface_language: Language for agent responses ("it" for Italian, "en" for English)
        """
        # Initialize LLM settings and factory
        self._llm_settings: LlmSettings = LlmSettingsReader().read(
            openai_api_key_override=openai_api_key,
            provider_override=provider,
            openai_chat_model_override=openai_chat_model,
            openai_embedding_model_override=openai_embedding_model,
            ollama_base_url_override=ollama_base_url,
            ollama_chat_model_override=ollama_chat_model,
            ollama_embedding_model_override=ollama_embedding_model,
        )
        self._llm_factory = LlmFactory(self._llm_settings)
        self._llm_factory.validate()

        # Initialize LLMs
        if self._llm_settings.provider == LlmProvider.OLLAMA:
            self._primary_model = self._llm_settings.ollama_chat_model
            self._fallback_model = (self._llm_settings.ollama_fallback_model or "").strip() or self._llm_settings.ollama_chat_model
        else:
            self._primary_model = self._llm_settings.openai_chat_model
            self._fallback_model = self._llm_settings.openai_fallback_model

        self._base_llm = self._llm_factory.create_chat_model()
        self._fallback_llm = self._llm_factory.create_fallback_chat_model()
        self._embeddings = self._llm_factory.create_embeddings()

        # Store interface language (user preference from settings)
        self._interface_language = interface_language if interface_language in ["it", "en"] else "it"

        # Initialize tools
        self._tools = self._initialize_tools(custom_db_path, custom_table_name, data_description, dataset_preset)

        # Initialize LLM with tools bound
        self._llm = self._base_llm.bind_tools(self._tools)

        # Initialize utility classes
        self._context_manager = ConversationContextManager(self._embeddings)
        self._extractor = DataExtractor()
        self._formatter = ItalianNumberFormatter()
        self._tool_loop_manager = ToolLoopManager()

        # Cache for tool summary (optimization: avoid rebuilding every request)
        self._cached_tools_summary: Optional[str] = None

        # Build graph
        self._graph = self._build_graph()

    def _initialize_tools(self, custom_db_path, custom_table_name, data_description, dataset_preset) -> List:
        """Initialize all available tools for the agent."""
        # Initialize DatasetQueryTool
        if custom_db_path and custom_table_name:
            dataset_tool = DatasetQueryTool(
                db_path=custom_db_path,
                table_name=custom_table_name,
                user_description=data_description,
                llm=self._base_llm,
                fallback_llm=self._fallback_llm,
                embeddings=self._embeddings,
            )
        elif dataset_preset in DATASET_PRESETS:
            preset = DATASET_PRESETS[dataset_preset]
            # Path: core.py is in /streamlit_app/agent/, need to go up 2 levels to reach project root
            db_path = Path(__file__).parent.parent.parent / preset["db_path"]
            dataset_tool = DatasetQueryTool(
                db_path=db_path,
                table_name=preset["table_name"],
                user_description=preset["description"],
                llm=self._base_llm,
                fallback_llm=self._fallback_llm,
                embeddings=self._embeddings,
            )
        else:
            dataset_tool = DatasetQueryTool(
                llm=self._base_llm,
                fallback_llm=self._fallback_llm,
                embeddings=self._embeddings,
            )

        # Initialize SpeciesListQueryTool
        # Path: core.py is in /streamlit_app/agent/, need to go up 2 levels to reach project root
        species_list_db_path = Path(__file__).parent.parent.parent / "dataset" / "species_list.db"
        species_list_tool = SpeciesListQueryTool(
            db_path=species_list_db_path,
            table_name="species_list",
            llm=self._base_llm,
            fallback_llm=self._fallback_llm,
            embeddings=self._embeddings,
        )

        # Initialize CO2AggregateTool
        if custom_db_path and custom_table_name:
            co2_aggregate_tool = CO2AggregateTool(
                db_path=custom_db_path,
                table_name=custom_table_name,
                dataset_type="custom",
                llm=self._base_llm
            )
        elif dataset_preset in DATASET_PRESETS:
            preset = DATASET_PRESETS[dataset_preset]
            # Path: core.py is in /streamlit_app/agent/, need to go up 2 levels to reach project root
            db_path = Path(__file__).parent.parent.parent / preset["db_path"]
            co2_aggregate_tool = CO2AggregateTool(
                db_path=db_path,
                table_name=preset["table_name"],
                dataset_type=dataset_preset,
                llm=self._base_llm
            )
        else:
            co2_aggregate_tool = CO2AggregateTool(
                dataset_type="vienna",
                llm=self._base_llm
            )

        # Initialize MapGenerationTool
        if custom_db_path and custom_table_name:
            map_tool = MapGenerationTool(
                db_path=custom_db_path,
                table_name=custom_table_name,
                llm=self._base_llm,
                fallback_llm=self._fallback_llm,
            )
        elif dataset_preset == "milano":
            map_tool = MapGenerationTool(llm=self._base_llm, fallback_llm=self._fallback_llm)
        else:
            map_tool = MapGenerationTool(llm=self._base_llm, fallback_llm=self._fallback_llm)

        # Static tools list
        static_tools = [
            CO2CalculationTool(),
            co2_aggregate_tool,
            CarbonContentTool(),
            EnvironmentEstimationTool(),
            dataset_tool,
            species_list_tool,
            ChartGenerationTool(llm=self._base_llm, fallback_llm=self._fallback_llm),
            map_tool,
            HeyerVolumeTool(),
            GeneralVolumeTool(),
            SimplifiedVolumeTool(),
            AllometricRelationTool(),
            LogAllometricTool(),
            ModelErrorTool(),
            LogFuelBiomassTool(),
            LeafBiomassTool(),
            StemBiomassTool(),
            RootBiomassTool(),
            TotalBiomassTool(),
            PaperSearchTool(),
            ExportDataTool(language=interface_language),
            LanguageDetectionTool(llm=self._base_llm),
            LanguageTranslationTool(llm=self._base_llm),
        ]

        # Load dynamic tools from JSON
        try:
            dynamic_loader = DynamicToolLoader()
            dynamic_tools = dynamic_loader.create_tools()
            # Store tools summary for planning
            self._dynamic_tools_summary = dynamic_loader.get_tools_summary()
        except Exception:
            dynamic_tools = []
            self._dynamic_tools_summary = []

        # Combine static and dynamic tools
        return static_tools + dynamic_tools

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow with all nodes and edges."""
        workflow = StateGraph(AgentState)

        # Define nodes
        workflow.add_node("context_manager", self._manage_context)
        workflow.add_node("query_optimizer", self._optimize_query)
        workflow.add_node("budget_check", self._check_budget)
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", ToolNode(self._tools))
        workflow.add_node("tool_loop_guard", self._guard_tool_loop)
        workflow.add_node("tool_loop_replanner", self._replan_after_tool_loop)
        workflow.add_node("validator", self._validate_response)
        workflow.add_node("retry_counter", self._increment_retry_count)

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

    # ===== Node Methods =====

    def _manage_context(self, state: AgentState) -> dict:
        """Manage conversation context to avoid token limit issues."""
        messages = state.get("messages") or []
        return self._context_manager.manage_context(messages)

    def _optimize_query(self, state: AgentState) -> dict:
        """Optimize user query, break it into tasks, and plan which tools to use."""
        messages = state["messages"]

        # Get the last user message
        last_user_msg = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_user_msg = msg.content
                break

        if not last_user_msg:
            return {"optimized_query": None, "tasks": [], "tool_plan": [], "detected_language": self._interface_language}

        # Use the configured interface language from user settings
        detected_language = self._interface_language

        # Build available tools summary for planning
        tools_summary = self._build_tools_summary_for_planning()

        # For simple/short queries, skip LLM optimization and do basic planning
        # This saves 1 LLM call per simple request (~30% overhead reduction)
        if len(last_user_msg) < 100 or self._is_simple_query(last_user_msg):
            simple_plan = self._create_simple_tool_plan(last_user_msg, detected_language)
            return {
                "optimized_query": last_user_msg,
                "tasks": [last_user_msg],
                "tool_plan": simple_plan,
                "available_tools_summary": tools_summary,
                "detected_language": detected_language
            }

        # Use LLM to optimize query and plan tools
        if detected_language == "en":
            optimizer_prompt = f"""Analyze the following question and create a task list with tool selection.

Question: {last_user_msg}

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
            optimization_msg_template = """Optimized query: {optimized_query}

Execution Plan:
{plan_text}"""
        else:
            optimizer_prompt = f"""Analizza la seguente domanda e crea una lista di task con selezione dei tool.

Domanda: {last_user_msg}

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
            optimization_msg_template = """Query ottimizzata: {optimized_query}

Piano di esecuzione:
{plan_text}"""

        try:
            optimizer_llm = self._create_chat_without_tools(model=self._fallback_model, temperature=0.7)
            response = optimizer_llm.invoke([HumanMessage(content=optimizer_prompt)])
            response_text = response.content.strip()

            # Extract JSON from markdown code blocks if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            optimization_result = json.loads(response_text)
            optimized_query = optimization_result.get("optimized_query", last_user_msg)
            tasks = optimization_result.get("tasks", [])
            tool_plan = optimization_result.get("tool_plan", [])

            # Format plan for display
            plan_text = self._format_tool_plan(tasks, tool_plan, detected_language)
            optimization_msg = SystemMessage(
                content=optimization_msg_template.format(optimized_query=optimized_query, plan_text=plan_text)
            )

            return {
                "messages": [optimization_msg],
                "optimized_query": optimized_query,
                "tasks": tasks,
                "tool_plan": tool_plan,
                "available_tools_summary": tools_summary,
                "detected_language": detected_language
            }

        except Exception as e:
            # Log the error instead of failing silently (visibility for debugging)
            import logging
            logging.warning(f"Query optimizer failed: {e}, falling back to simple plan")
            simple_plan = self._create_simple_tool_plan(last_user_msg, detected_language)
            return {
                "optimized_query": last_user_msg,
                "tasks": [last_user_msg],
                "tool_plan": simple_plan,
                "available_tools_summary": tools_summary,
                "detected_language": detected_language
            }

    def _is_simple_query(self, query: str) -> bool:
        """Detect simple queries that don't require LLM optimization.

        Simple queries are direct requests like "quanti alberi?" that can be
        handled with keyword-based tool selection, saving an LLM call.

        Uses patterns loaded dynamically from presets.json.

        Args:
            query: The user's query string

        Returns:
            True if the query is simple and can skip LLM optimization
        """
        query_lower = query.lower()
        config = get_config_loader()

        # Load patterns dynamically from config for both languages
        patterns_it = config.get_simple_query_patterns("it")
        patterns_en = config.get_simple_query_patterns("en")
        all_patterns = patterns_it + patterns_en

        # Also check if query matches any tool keywords (indicates direct tool request)
        matching_tools = config.match_keywords_to_tools(query)

        return any(pattern in query_lower for pattern in all_patterns) or len(matching_tools) > 0

    def _build_tools_summary_for_planning(self) -> str:
        """Build a summary of all available tools for task planning.

        Uses caching to avoid rebuilding the summary on every request (~2000+ chars saved).
        Loads tool descriptions dynamically from presets.json.
        """
        # Return cached summary if available (optimization)
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
        if hasattr(self, '_dynamic_tools_summary') and self._dynamic_tools_summary:
            summary_lines.append("\nDynamic Formula Tools:")
            for tool_info in self._dynamic_tools_summary:
                summary_lines.append(f"- {tool_info['name']}: {tool_info['title']} ({tool_info['formula']})")

        # Cache the result
        self._cached_tools_summary = "\n".join(summary_lines)
        return self._cached_tools_summary

    def _create_simple_tool_plan(self, query: str, language: str) -> List[Dict]:
        """Create a simple tool plan for short queries.

        Uses dynamic keyword matching from presets.json instead of hardcoded mappings.
        """
        config = get_config_loader()

        # Match keywords to tools dynamically
        tools = config.match_keywords_to_tools(query)

        # Also check dynamic tools keywords
        if hasattr(self, '_dynamic_tools_summary') and self._dynamic_tools_summary:
            query_lower = query.lower()
            for tool_info in self._dynamic_tools_summary:
                keywords = tool_info.get("keywords", [])
                for kw in keywords:
                    if kw.lower() in query_lower and tool_info["name"] not in tools:
                        tools.append(tool_info["name"])
                        break

        # Default to dataset query if no specific match
        if not tools:
            tools.append("query_dataset")

        reason_it = "Tool suggerito in base alla domanda"
        reason_en = "Suggested tool based on the query"

        return [{
            "task": query,
            "tools": tools,
            "reason": reason_en if language == "en" else reason_it
        }]

    def _format_tool_plan(self, tasks: List[str], tool_plan: List[Dict], language: str) -> str:
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
                lines.append(f"   Tool: {', '.join(tools) if tools else 'selezione automatica'}")
                if reason:
                    lines.append(f"   Motivo: {reason}")

        return "\n".join(lines)

    def _check_budget(self, state: AgentState) -> dict:
        """Check budget constraints before tool execution."""
        messages = state.get("messages") or []

        # Get detected language early for use in error messages
        detected_language = state.get("detected_language", "it")
        if detected_language not in ["it", "en"]:
            detected_language = "it"

        # Restore or create budget
        budget_data = state.get("budget")
        if budget_data:
            budget = AgentBudget.from_dict(budget_data)
        else:
            budget = AgentBudget()

        # Create guard and check (pass language for proper error messages)
        guard = BudgetAwareToolGuard(budget)
        can_proceed, error_msg, status = guard.check_before_tools(messages, detected_language)

        if not can_proceed:
            # Budget exceeded - generate conversational response
            conversational_response = self._generate_conversational_summary(messages, status, detected_language)
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

    def _generate_conversational_summary(self, messages: Sequence[BaseMessage], budget_status: Dict[str, Any], language: Literal["it", "en"] = "it") -> str:
        """Generate a conversational response summarizing results collected so far."""
        # Extract user's original question
        user_question = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                user_question = msg.content

        # Extract all tool results
        tool_results = self._extractor.extract_tool_results(messages)

        if not tool_results:
            if language == "en":
                return (
                    "I haven't collected enough data to answer completely yet.\n\n"
                    "**Suggestion:** Try rephrasing the question more specifically.\n\n"
                    "Tools used: None"
                )
            else:
                return (
                    "Non ho ancora raccolto abbastanza dati per rispondere completamente.\n\n"
                    "**Suggerimento:** Prova a riformulare la domanda in modo più specifico.\n\n"
                    "Tool utilizzati: Nessuno"
                )

        tools_used = list(set(r["tool"] for r in tool_results))

        # Try to generate a conversational summary using LLM
        try:
            summary_llm = self._create_chat_without_tools(model=self._fallback_model, temperature=0.7)

            # Format results for summary
            results_text = self._format_tool_results_for_summary(tool_results)

            if language == "en":
                summary_prompt = f"""Generate a conversational and friendly response in English.

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
                tool_citation_prefix = "Tools used:"
            else:
                summary_prompt = f"""Genera una risposta conversazionale e amichevole in italiano.

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
                tool_citation_prefix = "Tool utilizzati:"

            response = summary_llm.invoke([HumanMessage(content=summary_prompt)])
            summary = response.content.strip()

            # Ensure tool citation is present
            if tool_citation_prefix not in summary:
                summary += f"\n\n{tool_citation_prefix} {', '.join(tools_used)}"

            return summary

        except Exception:
            # Fallback to basic summary
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
            results_text = json.dumps(tool_results[0].get("result", {}), indent=2, ensure_ascii=False)[:500]

        return results_text

    def _call_model(self, state: AgentState) -> dict:
        """Call the LLM model."""
        messages = state["messages"]

        # Get detected language from state (default to Italian)
        detected_language = state.get("detected_language", "it")
        if detected_language not in ["it", "en"]:
            detected_language = "it"

        # ALWAYS replace/add system message with correct language
        # Remove any existing system messages first
        messages = [m for m in messages if not isinstance(m, SystemMessage)]
        # Add system message with correct language
        system_prompt = SystemPrompts.get_system_prompt(detected_language)
        system_msg = SystemMessage(content=system_prompt)
        messages = [system_msg] + list(messages)

        # Truncate messages to save tokens
        truncate_label = "... [truncated]" if detected_language == "en" else "... [troncato]"
        def _truncate(msg: BaseMessage, max_len: int = 800) -> BaseMessage:
            content = (msg.content or "") if hasattr(msg, "content") else ""
            if len(content) > max_len:
                content = content[:max_len] + truncate_label
            if isinstance(msg, HumanMessage):
                return HumanMessage(content=content)
            if isinstance(msg, AIMessage):
                # Preserve tool_calls when truncating AIMessage
                return AIMessage(content=content, tool_calls=getattr(msg, 'tool_calls', None) or [])
            if isinstance(msg, SystemMessage):
                return SystemMessage(content=content)
            if isinstance(msg, ToolMessage):
                # ToolMessage requires tool_call_id and name
                return ToolMessage(
                    content=content,
                    tool_call_id=getattr(msg, 'tool_call_id', ''),
                    name=getattr(msg, 'name', '')
                )
            return msg

        # Build minimal message list - MUST include ToolMessages for agent to see results
        system_messages = [m for m in messages if isinstance(m, SystemMessage)]
        last_user_msg = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_user_msg = msg
                break

        # Find the last AIMessage with tool_calls and its corresponding ToolMessages
        last_ai_with_tools = None
        tool_messages_after_ai = []

        for i, msg in enumerate(messages):
            if isinstance(msg, AIMessage) and getattr(msg, 'tool_calls', None):
                last_ai_with_tools = msg
                # Collect all ToolMessages that follow this AIMessage
                tool_messages_after_ai = []
                for j in range(i + 1, len(messages)):
                    if isinstance(messages[j], ToolMessage):
                        tool_messages_after_ai.append(messages[j])
                    elif isinstance(messages[j], AIMessage):
                        # Stop when we hit another AIMessage
                        break

        minimal_messages = []
        minimal_messages.extend(system_messages[:1])
        if last_user_msg:
            minimal_messages.append(last_user_msg)

        # Include AIMessage with tool_calls AND its ToolMessages so agent can see results
        if last_ai_with_tools:
            minimal_messages.append(last_ai_with_tools)
            for tm in tool_messages_after_ai:
                minimal_messages.append(tm)

        # Truncate only non-ToolMessages (ToolMessages contain important results)
        def _truncate_if_needed(msg: BaseMessage) -> BaseMessage:
            if isinstance(msg, ToolMessage):
                # Truncate ToolMessages more conservatively to preserve results
                return _truncate(msg, max_len=1500)
            return _truncate(msg, max_len=400)

        minimal_messages = [_truncate_if_needed(m) for m in minimal_messages]

        try:
            response = self._llm.invoke(minimal_messages)
        except Exception as e:
            # Fallback to lighter model on error
            # NOTE: We intentionally DO NOT modify tool._llm to avoid state mutation
            # which can cause unpredictable behavior in subsequent requests
            if "rate_limit" in str(e).lower() or "429" in str(e) or "request too large" in str(e).lower():
                import logging
                logging.warning(f"Rate limit hit, switching to fallback model: {e}")
                try:
                    # Create a temporary LLM binding with fallback model
                    # Tools maintain their original LLM reference - only agent uses fallback
                    fallback_llm_with_tools = self._fallback_llm.bind_tools(self._tools)
                    minimal_fallback = []
                    minimal_fallback.extend(system_messages[:1])
                    if last_user_msg:
                        minimal_fallback.append(last_user_msg)
                    # Include ToolMessages in fallback too
                    if last_ai_with_tools:
                        minimal_fallback.append(last_ai_with_tools)
                        for tm in tool_messages_after_ai:
                            minimal_fallback.append(tm)
                    minimal_fallback = [_truncate(m, max_len=300) for m in minimal_fallback]
                    response = fallback_llm_with_tools.invoke(minimal_fallback)
                except Exception:
                    raise
            else:
                raise

        return {"messages": [response]}

    def _guard_tool_loop(self, state: AgentState) -> dict:
        """Guard against tool call loops."""
        messages = state.get("messages") or []
        return self._tool_loop_manager.check_for_loops(messages, state)

    def _replan_after_tool_loop(self, state: AgentState) -> dict:
        """Recover from repeated tool calls by asking the agent to self-reflect."""
        messages = state.get("messages") or []
        replan_msg = self._tool_loop_manager.create_replan_prompt(state, messages)

        return {
            "messages": [replan_msg],
            "tool_loop_action": "continue",
            "tool_loop_detected": False,
            "tool_loop_replan_count": int(state.get("tool_loop_replan_count") or 0) + 1,
        }

    def _validate_response(self, state: AgentState) -> dict:
        """Validate if the response adequately answers the user's question."""
        messages = state["messages"]

        # Placeholder - simplified validation
        return {"validation_result": {"is_complete": True, "feedback": ""}}

    def _increment_retry_count(self, state: AgentState) -> dict:
        """Increment retry counter."""
        current = int(state.get("retry_count") or 0)
        return {"retry_count": current + 1}

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

    def _should_continue_after_tool_guard(self, state: AgentState) -> Literal["continue", "replan", "stop"]:
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

    # ===== Public Methods =====

    def chat(self, message: str, history: Optional[List[dict]] = None) -> str:
        """Chat with the agent.

        Args:
            message: User message
            history: Optional chat history as list of dicts with 'role' and 'content' keys

        Returns:
            Agent response as string
        """
        # Convert history to messages
        messages: List[BaseMessage] = []
        if history:
            for msg in history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))

        messages.append(HumanMessage(content=message))

        # Initialize fresh budget
        initial_budget = AgentBudget()

        # Run graph
        result = self._graph.invoke(
            {
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
                "detected_language": self._interface_language,
                "tool_plan": None,
                "available_tools_summary": None,
            },
            config={"recursion_limit": 30},
        )

        # Extract final response
        final_message = result["messages"][-1]
        if isinstance(final_message, AIMessage):
            return final_message.content

        return str(final_message.content)

    def stream_chat(self, message: str, history: Optional[List[dict]] = None):
        """Stream chat response with LangGraph streaming, including reasoning steps.

        Args:
            message: User message
            history: Optional chat history

        Yields:
            Dict with 'type' and 'content' keys:
            - type: 'reasoning' (internal step) or 'response' (final answer)
            - content: the actual content to display
        """
        # Convert history to messages
        messages: List[BaseMessage] = []
        if history:
            for msg in history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))

        messages.append(HumanMessage(content=message))

        # Track execution
        final_response = None
        retry_count = 0
        max_retries = 1
        chart_data_json = None
        map_data_json = None

        # Initialize fresh budget
        initial_budget = AgentBudget()

        # Stream from graph
        for event in self._graph.stream(
            {
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
                "detected_language": self._interface_language,
                "tool_plan": None,
                "available_tools_summary": None,
            },
            config={"recursion_limit": 30},
            stream_mode="updates",
        ):
            # Get language from state or use interface language
            current_language = self._interface_language
            if "detected_language" in event.get("query_optimizer", {}):
                current_language = event["query_optimizer"].get("detected_language", self._interface_language)
            
            # Process each node event
            for node_name, node_output in event.items():
                result = None

                if node_name == "language_detector":
                    result = StreamingHandler.handle_language_detector_event(node_output, current_language)
                elif node_name == "context_manager":
                    result = StreamingHandler.handle_context_manager_event(node_output, len(messages), current_language)
                elif node_name == "query_optimizer":
                    # Update language from optimizer output
                    if "detected_language" in node_output:
                        current_language = node_output.get("detected_language", self._interface_language)
                    result = StreamingHandler.handle_query_optimizer_event(node_output, current_language)
                elif node_name == "agent":
                    result = StreamingHandler.handle_agent_event(node_output, current_language)
                    if result and result.get("type") == "final_response":
                        final_response = result.get("content")
                        result = None  # Don't yield yet
                elif node_name == "tools":
                    result, chart_json, map_json = StreamingHandler.handle_tools_event(node_output, current_language)
                    if chart_json:
                        chart_data_json = chart_json
                    if map_json:
                        map_data_json = map_json
                elif node_name == "budget_check":
                    result = StreamingHandler.handle_budget_check_event(node_output, current_language)
                    if result and "final_response" in result:
                        final_response = result.pop("final_response")
                elif node_name == "tool_loop_guard":
                    result = StreamingHandler.handle_tool_loop_guard_event(node_output, current_language)
                    if result and "final_response" in result:
                        final_response = result.pop("final_response")
                elif node_name == "tool_loop_replanner":
                    from streamlit_app.agent.translations import get_translation
                    result = {
                        "type": "reasoning", 
                        "content": f"{get_translation('replanning', current_language)}\n\n{get_translation('reformulating_step', current_language)}\n"
                    }
                elif node_name == "validator":
                    result, retry_count = StreamingHandler.handle_validator_event(node_output, retry_count, max_retries, current_language)

                if result:
                    yield result

        # Yield final response
        if final_response:
            # Add chart/map data if present
            if chart_data_json and "CHART_DATA_START" not in final_response:
                final_response = f"{final_response}\n\nCHART_DATA_START\n{chart_data_json}\nCHART_DATA_END"

            if map_data_json and "MAP_DATA_START" not in final_response:
                final_response = f"{final_response}\n\nMAP_DATA_START\n{map_data_json}\nMAP_DATA_END"

            yield {"type": "response", "content": final_response}

    # ===== Utility Methods =====

    def _create_chat_without_tools(self, model: str, temperature: float) -> Any:
        """Create a plain chat model (no tool binding) for internal prompts."""
        if self._llm_settings.provider == LlmProvider.OLLAMA:
            from langchain_ollama import ChatOllama
            return ChatOllama(model=model, temperature=temperature, base_url=self._llm_settings.ollama_base_url)

        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature, api_key=self._llm_settings.openai_api_key)
