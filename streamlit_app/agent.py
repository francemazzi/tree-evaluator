from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Annotated, List, Literal, Optional, Sequence, TypedDict, Any, Dict, Tuple
from pathlib import Path
import re
from decimal import Decimal, InvalidOperation

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from streamlit_app.llm.factory import LlmFactory, LlmProvider, LlmSettings, LlmSettingsReader
from streamlit_app.llm.tool_loop_guard import ToolLoopGuard
from streamlit_app.tools.chart_tool import ChartGenerationTool
from streamlit_app.tools.co2_tool import CO2CalculationTool
from streamlit_app.tools.co2_aggregate_tool import CO2AggregateTool
from streamlit_app.tools.dataset_tool import DatasetQueryTool
from streamlit_app.tools.environment_tool import EnvironmentEstimationTool
from streamlit_app.tools.heyer_volume_tool import HeyerVolumeTool
from streamlit_app.tools.general_volume_tool import GeneralVolumeTool
from streamlit_app.tools.simplified_volume_tool import SimplifiedVolumeTool
from streamlit_app.tools.allometric_relation_tool import AllometricRelationTool
from streamlit_app.tools.log_allometric_tool import LogAllometricTool
from streamlit_app.tools.model_error_tool import ModelErrorTool
from streamlit_app.tools.log_fuel_biomass_tool import LogFuelBiomassTool
from streamlit_app.tools.leaf_biomass_tool import LeafBiomassTool
from streamlit_app.tools.stem_biomass_tool import StemBiomassTool
from streamlit_app.tools.root_biomass_tool import RootBiomassTool
from streamlit_app.tools.total_biomass_tool import TotalBiomassTool
from streamlit_app.tools.map_tool import MapGenerationTool
from streamlit_app.tools.species_list_tool import SpeciesListQueryTool
from streamlit_app.tools.paper_search_tool import PaperSearchTool

# Load environment variables
load_dotenv()


@dataclass
class AgentBudget:
    """Budget constraints for agent execution to prevent infinite loops and runaway costs.
    
    This class tracks resource usage during agent execution and enforces limits on:
    - Total tool calls across all tools
    - Calls per individual tool
    - LLM invocations
    - Execution time
    - Replan attempts
    """
    
    # Budget limits (configurable)
    max_total_tool_calls: int = 15          # Hard limit on total tool calls
    max_calls_per_tool: int = 3             # Max calls to same tool
    max_llm_calls: int = 10                 # Max LLM invocations
    max_execution_time_seconds: int = 120   # Timeout in seconds
    max_replans: int = 2                    # Max replan attempts
    
    # Runtime counters
    tool_calls: Dict[str, int] = field(default_factory=dict)
    total_tool_calls: int = 0
    llm_calls: int = 0
    start_time: float = field(default_factory=time.time)
    replans: int = 0
    
    def can_call_tool(self, tool_name: str) -> Tuple[bool, str]:
        """Check if we can call a specific tool.
        
        Args:
            tool_name: Name of the tool to check.
            
        Returns:
            Tuple of (can_call, reason_if_blocked).
        """
        # Check timeout
        elapsed = time.time() - self.start_time
        if elapsed > self.max_execution_time_seconds:
            return False, f"Timeout: {elapsed:.1f}s exceeded limit of {self.max_execution_time_seconds}s"
        
        # Check total tool calls
        if self.total_tool_calls >= self.max_total_tool_calls:
            return False, f"Total tool call limit reached: {self.total_tool_calls}/{self.max_total_tool_calls}"
        
        # Check per-tool limit
        current_count = self.tool_calls.get(tool_name, 0)
        if current_count >= self.max_calls_per_tool:
            return False, f"Tool '{tool_name}' limit reached: {current_count}/{self.max_calls_per_tool} calls"
        
        return True, ""
    
    def record_tool_call(self, tool_name: str) -> None:
        """Record a tool call."""
        self.tool_calls[tool_name] = self.tool_calls.get(tool_name, 0) + 1
        self.total_tool_calls += 1
    
    def can_call_llm(self) -> Tuple[bool, str]:
        """Check if we can make another LLM call."""
        if self.llm_calls >= self.max_llm_calls:
            return False, f"LLM call limit reached: {self.llm_calls}/{self.max_llm_calls}"
        return True, ""
    
    def record_llm_call(self) -> None:
        """Record an LLM call."""
        self.llm_calls += 1
    
    def can_replan(self) -> Tuple[bool, str]:
        """Check if we can attempt another replan."""
        if self.replans >= self.max_replans:
            return False, f"Replan limit reached: {self.replans}/{self.max_replans}"
        return True, ""
    
    def record_replan(self) -> None:
        """Record a replan attempt."""
        self.replans += 1
    
    def get_status(self) -> Dict[str, Any]:
        """Get current budget status for debugging/logging."""
        elapsed = time.time() - self.start_time
        return {
            "total_tool_calls": f"{self.total_tool_calls}/{self.max_total_tool_calls}",
            "llm_calls": f"{self.llm_calls}/{self.max_llm_calls}",
            "replans": f"{self.replans}/{self.max_replans}",
            "elapsed_time": f"{elapsed:.1f}s/{self.max_execution_time_seconds}s",
            "per_tool_calls": dict(self.tool_calls),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize budget to dict for state storage."""
        return {
            "max_total_tool_calls": self.max_total_tool_calls,
            "max_calls_per_tool": self.max_calls_per_tool,
            "max_llm_calls": self.max_llm_calls,
            "max_execution_time_seconds": self.max_execution_time_seconds,
            "max_replans": self.max_replans,
            "tool_calls": dict(self.tool_calls),
            "total_tool_calls": self.total_tool_calls,
            "llm_calls": self.llm_calls,
            "start_time": self.start_time,
            "replans": self.replans,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentBudget":
        """Deserialize budget from dict."""
        budget = cls(
            max_total_tool_calls=data.get("max_total_tool_calls", 15),
            max_calls_per_tool=data.get("max_calls_per_tool", 3),
            max_llm_calls=data.get("max_llm_calls", 10),
            max_execution_time_seconds=data.get("max_execution_time_seconds", 120),
            max_replans=data.get("max_replans", 2),
        )
        budget.tool_calls = dict(data.get("tool_calls", {}))
        budget.total_tool_calls = data.get("total_tool_calls", 0)
        budget.llm_calls = data.get("llm_calls", 0)
        budget.start_time = data.get("start_time", time.time())
        budget.replans = data.get("replans", 0)
        return budget


class BudgetAwareToolGuard:
    """Circuit breaker that enforces budget limits before tool execution.
    
    This guard checks budget constraints before each tool call and prevents
    execution if limits are exceeded, returning a user-friendly error message.
    """
    
    def __init__(self, budget: AgentBudget):
        self._budget = budget
    
    def check_before_tools(self, messages: Sequence[BaseMessage]) -> Tuple[bool, str, Dict[str, Any]]:
        """Check budget before tool execution.
        
        Args:
            messages: Current conversation messages.
            
        Returns:
            Tuple of (can_proceed, error_message, budget_status).
        """
        # Extract pending tool calls from last AI message
        last_ai = None
        for msg in reversed(list(messages)):
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                last_ai = msg
                break
        
        if not last_ai or not last_ai.tool_calls:
            return True, "", self._budget.get_status()
        
        # Check each tool call against budget
        for tc in last_ai.tool_calls:
            tool_name = tc.get("name", "unknown")
            can_call, reason = self._budget.can_call_tool(tool_name)
            
            if not can_call:
                error_msg = self._build_budget_error_response(reason, tool_name)
                return False, error_msg, self._budget.get_status()
            
            # Record the call (pre-emptively)
            self._budget.record_tool_call(tool_name)
        
        return True, "", self._budget.get_status()
    
    def check_before_llm(self) -> Tuple[bool, str]:
        """Check budget before LLM call."""
        can_call, reason = self._budget.can_call_llm()
        if can_call:
            self._budget.record_llm_call()
        return can_call, reason
    
    def _build_budget_error_response(self, reason: str, tool_name: str) -> str:
        """Build user-friendly error message when budget is exceeded."""
        status = self._budget.get_status()
        tools_used = ', '.join(status['per_tool_calls'].keys()) or 'Nessuno'
        
        return f"""⚠️ **Limite di esecuzione raggiunto**

{reason}

**Stato attuale:**
- Tool calls totali: {status['total_tool_calls']}
- Chiamate LLM: {status['llm_calls']}
- Tempo trascorso: {status['elapsed_time']}
- Tool più usato: {tool_name}

**Suggerimenti:**
- Riformula la domanda in modo più specifico
- Suddividi la richiesta in domande più semplici
- Chiedi direttamente il risultato senza elaborazioni complesse

Tool utilizzati: {tools_used}"""


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


class TreeEvaluatorAgent:
    """LangGraph-based agent that orchestrates tree evaluation tools."""

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
        dataset_preset: str = "vienna"
    ) -> None:
        """Initialize the agent with tools and LLM.

        Args:
            openai_api_key: OpenAI API key. If not provided, tries OPENAI_API_KEY env var.
            custom_db_path: Optional path to custom SQLite database
            custom_table_name: Optional custom table name in the database
            data_description: Optional description of the data for context
            dataset_preset: Preset dataset to use ("vienna", "milano")
        """
        # LLM provider settings (OpenAI/Ollama)
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

        # Initialize LLMs (primary + fallback)
        if self._llm_settings.provider == LlmProvider.OLLAMA:
            self._primary_model = self._llm_settings.ollama_chat_model
            self._fallback_model = (self._llm_settings.ollama_fallback_model or "").strip() or self._llm_settings.ollama_chat_model
        else:
            self._primary_model = self._llm_settings.openai_chat_model
            self._fallback_model = self._llm_settings.openai_fallback_model

        self._base_llm = self._llm_factory.create_chat_model()
        self._fallback_llm = self._llm_factory.create_fallback_chat_model()
        self._embeddings = self._llm_factory.create_embeddings()

        # Initialize DatasetQueryTool with appropriate database
        if custom_db_path and custom_table_name:
            # Custom uploaded CSV
            dataset_tool = DatasetQueryTool(
                db_path=custom_db_path,
                table_name=custom_table_name,
                user_description=data_description,
                llm=self._base_llm,
                fallback_llm=self._fallback_llm,
                embeddings=self._embeddings,
            )
        elif dataset_preset in self.DATASET_PRESETS:
            # Preset dataset (Vienna or Milano)
            preset = self.DATASET_PRESETS[dataset_preset]
            db_path = Path(__file__).parent.parent / preset["db_path"]
            dataset_tool = DatasetQueryTool(
                db_path=db_path,
                table_name=preset["table_name"],
                user_description=preset["description"],
                llm=self._base_llm,
                fallback_llm=self._fallback_llm,
                embeddings=self._embeddings,
            )
        else:
            # Default: Vienna
            dataset_tool = DatasetQueryTool(
                llm=self._base_llm,
                fallback_llm=self._fallback_llm,
                embeddings=self._embeddings,
            )

        # Initialize SpeciesListQueryTool (botanical taxonomy/traits context)
        species_list_db_path = Path(__file__).parent.parent / "dataset" / "species_list.db"
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
        elif dataset_preset in self.DATASET_PRESETS:
            preset = self.DATASET_PRESETS[dataset_preset]
            db_path = Path(__file__).parent.parent / preset["db_path"]
            co2_aggregate_tool = CO2AggregateTool(
                db_path=db_path,
                table_name=preset["table_name"],
                dataset_type=dataset_preset,
                llm=self._base_llm
            )
        else:
            # Default Vienna
            co2_aggregate_tool = CO2AggregateTool(
                dataset_type="vienna",
                llm=self._base_llm
            )

        # Initialize tools with LLM
        # Initialize MapGenerationTool with appropriate database for dataset preset
        if custom_db_path and custom_table_name:
            # Custom uploaded CSV - may or may not have coordinates
            map_tool = MapGenerationTool(
                db_path=custom_db_path,
                table_name=custom_table_name,
                llm=self._base_llm,
                fallback_llm=self._fallback_llm,
            )
        elif dataset_preset == "milano":
            # Milano has GPS coordinates
            map_tool = MapGenerationTool(llm=self._base_llm, fallback_llm=self._fallback_llm)
        else:
            # Vienna doesn't have GPS - still create tool but it will show error message
            map_tool = MapGenerationTool(llm=self._base_llm, fallback_llm=self._fallback_llm)
        
        self._tools = [
            CO2CalculationTool(),
            co2_aggregate_tool,
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
        ]

        # Initialize LLM with tools bound
        self._llm = self._base_llm.bind_tools(self._tools)

        # Build graph
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow with query optimization, validation, and budget enforcement."""
        workflow = StateGraph(AgentState)

        # Define nodes
        workflow.add_node("context_manager", self._manage_context)
        workflow.add_node("query_optimizer", self._optimize_query)
        workflow.add_node("budget_check", self._check_budget)  # Budget enforcement before tools
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", ToolNode(self._tools))
        workflow.add_node("tool_loop_guard", self._guard_tool_loop)
        workflow.add_node("tool_loop_replanner", self._replan_after_tool_loop)
        workflow.add_node("validator", self._validate_response)
        workflow.add_node("retry_counter", self._increment_retry_count)

        # Set entry point - start with context management
        workflow.set_entry_point("context_manager")
        
        # Context manager -> query optimizer
        workflow.add_edge("context_manager", "query_optimizer")

        # Query optimizer -> agent
        workflow.add_edge("query_optimizer", "agent")

        # Agent decides: continue to budget check (before tools) or validate
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "budget_check",  # Check budget before tools
                "validate": "validator",
            },
        )

        # Budget check decides: proceed to tools or stop
        workflow.add_conditional_edges(
            "budget_check",
            self._should_continue_after_budget,
            {
                "continue": "tools",
                "stop": END,
            },
        )

        # After tool execution, run loop guard before returning to agent
        workflow.add_edge("tools", "tool_loop_guard")
        workflow.add_conditional_edges(
            "tool_loop_guard",
            self._should_continue_after_tool_guard,
            {
                "continue": "agent",
                "replan": "tool_loop_replanner",
                "stop": END,
            },
        )
        workflow.add_edge("tool_loop_replanner", "agent")

        # Validator decides: complete or retry
        workflow.add_conditional_edges(
            "validator",
            self._should_retry,
            {
                "complete": END,
                "retry": "retry_counter",
            },
        )

        # After incrementing retry counter, go back to agent
        workflow.add_edge("retry_counter", "agent")

        return workflow.compile()

    def _check_budget(self, state: AgentState) -> dict:
        """Check budget constraints before tool execution.
        
        This node acts as a circuit breaker, preventing tool execution
        if budget limits are exceeded. When limit is reached, it generates
        a conversational response using the results already obtained.
        """
        messages = state.get("messages") or []
        
        # Restore or create budget
        budget_data = state.get("budget")
        if budget_data:
            budget = AgentBudget.from_dict(budget_data)
        else:
            budget = AgentBudget()
        
        # Create guard and check
        guard = BudgetAwareToolGuard(budget)
        can_proceed, error_msg, status = guard.check_before_tools(messages)
        
        if not can_proceed:
            # Budget exceeded - generate conversational response with collected results
            conversational_response = self._generate_conversational_summary(messages, status)
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
    
    def _generate_conversational_summary(self, messages: Sequence[BaseMessage], budget_status: Dict[str, Any]) -> str:
        """Generate a conversational response summarizing the results collected so far.
        
        Instead of just showing "budget exceeded", this method extracts all tool results
        and creates a human-friendly summary.
        """
        from langchain_core.messages import ToolMessage
        
        # Extract user's original question
        user_question = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                user_question = msg.content
        
        # Extract all tool results
        tool_results = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                tool_name = getattr(msg, "name", "unknown")
                content = msg.content
                
                # Try to parse JSON content
                try:
                    if isinstance(content, str):
                        import ast
                        try:
                            parsed = json.loads(content)
                        except json.JSONDecodeError:
                            parsed = ast.literal_eval(content)
                        tool_results.append({"tool": tool_name, "result": parsed})
                    elif isinstance(content, dict):
                        tool_results.append({"tool": tool_name, "result": content})
                except Exception:
                    tool_results.append({"tool": tool_name, "result": str(content)[:500]})
        
        if not tool_results:
            # No results yet, return simple message
            return (
                "Non ho ancora raccolto abbastanza dati per rispondere completamente.\n\n"
                "**Suggerimento:** Prova a riformulare la domanda in modo più specifico.\n\n"
                "Tool utilizzati: Nessuno"
            )
        
        # Build summary based on tool type
        tools_used = list(set(r["tool"] for r in tool_results))
        
        # Try to generate a conversational summary using LLM
        try:
            summary_llm = self._create_chat_without_tools(model=self._fallback_model, temperature=0.7)
            
            # Format results for summary
            results_text = ""
            for tr in tool_results[:5]:  # Limit to 5 results
                tool = tr["tool"]
                result = tr["result"]
                if isinstance(result, dict):
                    # Extract key values
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
            
            response = summary_llm.invoke([HumanMessage(content=summary_prompt)])
            summary = response.content.strip()
            
            # Ensure tool citation is present
            if "Tool utilizzati" not in summary:
                summary += f"\n\nTool utilizzati: {', '.join(tools_used)}"
            
            return summary
            
        except Exception as e:
            # Fallback to basic summary if LLM fails
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

    def _should_continue_after_budget(self, state: AgentState) -> Literal["continue", "stop"]:
        """Determine if we should continue to tools or stop due to budget."""
        if state.get("budget_exceeded", False):
            return "stop"
        return "continue"

    def _guard_tool_loop(self, state: AgentState) -> dict:
        messages = state.get("messages") or []
        from langchain_core.messages import ToolMessage
        
        # Count ALL ToolMessages in the conversation by tool name
        # This is the definitive count - each ToolMessage = one tool execution
        tool_call_counts: Dict[str, int] = {}
        for msg in messages:
            if isinstance(msg, ToolMessage):
                # ToolMessage has a 'name' attribute with the tool name
                tool_name = getattr(msg, "name", None)
                if tool_name:
                    tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
        
        # Check if any single tool has been called too many times (even with different args)
        MAX_CALLS_PER_TOOL = 5  # If same tool called 5+ times, force replan
        abused_tool = None
        for tool_name, count in tool_call_counts.items():
            if count >= MAX_CALLS_PER_TOOL:
                abused_tool = tool_name
                break
        
        if abused_tool:
            call_count = tool_call_counts[abused_tool]
            replan_count = int(state.get("tool_loop_replan_count") or 0)
            
            details = {
                "tool_calls": [{"name": abused_tool}],
                "fingerprint": f"{abused_tool}:repeated_{call_count}_times",
                "abuse_detected": True,
                "call_count": call_count,
            }
            
            # If tool called 10+ times OR we've already tried 3+ replans, FORCE STOP
            if call_count >= 10 or replan_count >= 3:
                # Build dynamic fallback response based on which tool was abused
                fallback_response = self._build_dynamic_fallback_response(
                    abused_tool=abused_tool,
                    call_count=call_count,
                    messages=messages
                )
                return {
                    "messages": [AIMessage(content=fallback_response)],
                    "tool_loop_detected": True,
                    "tool_loop_action": "stop",
                    "tool_loop_details": details,
                    "tool_call_counts": tool_call_counts,
                }
            
            # Otherwise, force replan
            return {
                "tool_loop_detected": True,
                "tool_loop_action": "replan",
                "tool_loop_details": details,
                "tool_call_counts": tool_call_counts,
            }
        
        # Also check for exact fingerprint repeats (same tool + same args)
        guard = ToolLoopGuard(max_consecutive_repeats=2)
        last_fp = state.get("tool_last_fingerprint")
        repeat = int(state.get("tool_repeat_count") or 0)

        decision, new_fp, new_repeat = guard.evaluate(messages=messages, last_fingerprint=last_fp, repeat_count=repeat)
        
        if decision.action == "stop" and decision.user_message:
            return {
                "messages": [AIMessage(content=decision.user_message)],
                "tool_last_fingerprint": new_fp,
                "tool_repeat_count": new_repeat,
                "tool_loop_detected": True,
                "tool_loop_action": "stop",
                "tool_loop_details": decision.details,
                "tool_call_counts": tool_call_counts,
            }
        if decision.action == "replan":
            return {
                "tool_last_fingerprint": new_fp,
                "tool_repeat_count": new_repeat,
                "tool_loop_detected": True,
                "tool_loop_action": "replan",
                "tool_loop_details": decision.details,
                "tool_call_counts": tool_call_counts,
            }
        return {
            "tool_last_fingerprint": new_fp,
            "tool_repeat_count": new_repeat,
            "tool_loop_detected": False,
            "tool_loop_action": "continue",
            "tool_loop_details": None,
            "tool_call_counts": tool_call_counts,
        }

    def _should_continue_after_tool_guard(self, state: AgentState) -> Literal["continue", "replan", "stop"]:
        action = state.get("tool_loop_action") or "continue"
        if action in ("continue", "replan", "stop"):
            return action
        return "continue"

    def _replan_after_tool_loop(self, state: AgentState) -> dict:
        """
        Recover from repeated tool calls by asking the agent to self-reflect.
        Uses progressively more assertive prompts.
        """
        current = int(state.get("tool_loop_replan_count") or 0)
        details: Dict[str, Any] = state.get("tool_loop_details") or {}
        tool_calls = details.get("tool_calls") or []
        abuse_detected = details.get("abuse_detected", False)
        call_count = details.get("call_count", 0)

        # Extract tool results from recent messages for self-evaluation
        messages = state.get("messages") or []
        from langchain_core.messages import ToolMessage
        recent_tool_results = []
        for msg in reversed(list(messages)[-15:]):  # Last 15 messages
            if isinstance(msg, ToolMessage):
                content = str(msg.content)[:400]  # Truncate for context
                recent_tool_results.append(content)
        
        tool_results_summary = "\n---\n".join(recent_tool_results[:3]) if recent_tool_results else "Nessun risultato recente"
        
        # Get the abused tool name
        abused_tool = tool_calls[0].get("name") if tool_calls else "questo tool"

        # If tool abuse detected (same tool called many times with different args)
        if abuse_detected or call_count >= 5:
            prompt = f"""🛑 **STOP - HAI CHIAMATO `{abused_tool}` {call_count} VOLTE**

Stai chiamando lo stesso tool ripetutamente con query diverse, ma non stai facendo progressi.

**RISULTATI CHE HAI GIÀ OTTENUTO:**
{tool_results_summary}

**ANALIZZA LA SITUAZIONE:**
- Hai già cercato {call_count} volte - se non hai trovato quello che cerchi, probabilmente non c'è
- Guarda i risultati sopra: contengono informazioni utili?
- Puoi rispondere con quello che hai, anche se parziale?

**SCEGLI UNA DI QUESTE AZIONI (OBBLIGATORIO):**

1. **RISPONDI CON QUELLO CHE HAI**: Usa i paper/risultati che hai trovato per dare una risposta. Esempio:
   "Dai paper trovati su arXiv, le formule più comuni per l'assorbimento di carbonio sono:
   - Formula 1: [descrivi]
   - Formula 2: [descrivi]
   Fonte: [link ai paper]"

2. **AMMETTI I LIMITI E OFFRI ALTERNATIVE**: Se non hai trovato esattamente quello che l'utente cerca:
   "Non ho trovato formule specifiche sull'assorbimento di carbonio su arXiv per la tua query.
   Posso però aiutarti con:
   - Le formule di calcolo CO2 che ho già implementate (basate su DBH e altezza)
   - Il calcolo della biomassa con equazioni allometriche
   Vuoi che ti mostri queste?"

3. **CHIEDI CHIARIMENTI**: Se hai bisogno di più contesto:
   "Per aiutarti meglio, puoi specificare:
   - Stai cercando formule per alberi, foreste, o ecosistemi?
   - Hai un paper specifico in mente?"

**⛔ NON PUOI chiamare di nuovo `{abused_tool}`. Devi rispondere ora.**
"""
        # Progressive assertiveness for exact fingerprint repeats
        elif current < 2:
            prompt = f"""🔄 **MOMENTO DI AUTO-RIFLESSIONE**

Hai chiamato lo stesso tool più volte. Prima di procedere, chiediti:

**Risultati ottenuti finora:**
{tool_results_summary}

**Domande da porti:**
1. Questi risultati rispondono (anche parzialmente) alla domanda dell'utente?
2. Sto cercando qualcosa che potrebbe non esistere nei dati disponibili?
3. Posso dare una risposta utile con quello che ho?

**Azioni possibili:**
A) **RISPONDO**: Formula una risposta con ciò che hai trovato (anche se parziale)
B) **CHIEDO**: Fai una domanda specifica all'utente per capire meglio
C) **CAMBIO STRATEGIA**: Usa un tool diverso

NON richiamare lo stesso tool con la stessa query.
"""
        else:
            prompt = f"""🛑 **STOP - RISPOSTA OBBLIGATORIA**

Hai tentato {current + 1} volte senza successo. È il momento di rispondere all'utente.

**Risultati disponibili:**
{tool_results_summary}

**ISTRUZIONI FINALI:**
Scrivi ORA una risposta all'utente che:
1. Spiega onestamente cosa hai cercato e cosa hai (o non hai) trovato
2. Offre alternative concrete: "Non ho trovato X, ma posso aiutarti con Y..."
3. Chiede se l'utente vuole procedere diversamente

**RISPONDI ORA - Non chiamare altri tool.**
"""

        return {
            "messages": [SystemMessage(content=prompt)],
            "tool_loop_action": "continue",
            "tool_loop_detected": False,
            "tool_loop_replan_count": current + 1,
        }

    def _increment_retry_count(self, state: AgentState) -> dict:
        current = int(state.get("retry_count") or 0)
        return {"retry_count": current + 1}

    def _build_dynamic_fallback_response(
        self, 
        abused_tool: str, 
        call_count: int, 
        messages: Sequence[BaseMessage]
    ) -> str:
        """Build a dynamic fallback response when tool loop limit is reached.
        
        This method generates context-aware responses based on which tool was
        being called repeatedly, avoiding hardcoded formulas or domain-specific content.
        
        Args:
            abused_tool: Name of the tool that was called too many times.
            call_count: Number of times the tool was called.
            messages: Current conversation messages.
            
        Returns:
            User-friendly fallback response.
        """
        # Map tool names to user-friendly descriptions
        tool_descriptions = {
            "search_scientific_papers": "ricerca di paper scientifici",
            "query_tree_dataset": "interrogazione del dataset",
            "calculate_co2_sequestration": "calcolo del sequestro di CO2",
            "calculate_co2_aggregate": "calcolo aggregato CO2 (dataset)",
            "estimate_environment": "stima ambientale",
            "generate_chart": "generazione di grafici",
            "generate_map": "generazione di mappe",
            "query_species_list": "ricerca nella lista delle specie",
        }
        
        tool_desc = tool_descriptions.get(abused_tool, f"utilizzo del tool {abused_tool}")
        
        # Start with generic message
        fallback_response = (
            f"⚠️ **Limite di ricerca raggiunto**\n\n"
            f"Ho eseguito {call_count} tentativi di {tool_desc} senza trovare una risposta definitiva.\n\n"
        )
        
        # For paper search, try to extract and show found papers
        if abused_tool == "search_scientific_papers":
            papers_found = self._extract_papers_from_messages(messages)
            
            if papers_found:
                real_papers = [p for p in papers_found if p.get("source") != "error"]
                error_papers = [p for p in papers_found if p.get("source") == "error"]
                
                if real_papers:
                    fallback_response += "**📚 Paper trovati:**\n\n"
                    for i, paper in enumerate(real_papers[:5], 1):
                        title = paper.get("title", "Titolo non disponibile")
                        authors = paper.get("authors", "")
                        link = paper.get("link", "")
                        abstract = paper.get("abstract", "")
                        if abstract and len(abstract) > 200:
                            abstract = abstract[:200] + "..."
                        
                        fallback_response += f"{i}. **{title}**\n"
                        if authors and authors != "N/A":
                            fallback_response += f"   - Autori: {authors}\n"
                        if abstract:
                            fallback_response += f"   - Abstract: {abstract}\n"
                        if link:
                            fallback_response += f"   - 🔗 [Link al paper]({link})\n"
                        fallback_response += "\n"
                elif error_papers:
                    error_msg = error_papers[0].get("abstract", "errore sconosciuto")
                    fallback_response += f"*⚠️ Errore durante la ricerca: {error_msg}*\n\n"
                else:
                    fallback_response += "*Non ho trovato risultati specifici per la tua query.*\n\n"
            else:
                fallback_response += "*Non ho trovato risultati specifici per la tua query.*\n\n"
        
        # For dataset queries, show what was found
        elif abused_tool == "query_tree_dataset":
            fallback_response += (
                "**Suggerimenti:**\n"
                "- Prova a riformulare la domanda in modo più specifico\n"
                "- Verifica che i nomi delle colonne siano corretti\n"
                "- Chiedi prima la struttura del dataset con \"Mostrami le colonne disponibili\"\n\n"
            )
        
        # Generic suggestions for other tools
        else:
            fallback_response += (
                "**Cosa puoi fare:**\n"
                "- Riformula la domanda in modo più specifico\n"
                "- Suddividi la richiesta in domande più semplici\n"
                "- Chiedi informazioni più mirate\n\n"
            )
        
        # Add available tools suggestion
        fallback_response += (
            "**Altri tool disponibili:**\n"
            "- 📊 Analisi dataset (query, statistiche, grafici)\n"
            "- 🌳 Calcoli forestali (CO2, biomassa, volume)\n"
            "- 🗺️ Mappe interattive (solo dataset con coordinate GPS)\n"
            "- 📚 Ricerca paper scientifici\n\n"
            "Posso aiutarti con qualcosa di specifico?\n\n"
            f"Tool utilizzati: {abused_tool.replace('_', ' ').title()}"
        )
        
        return fallback_response

    def _extract_papers_from_messages(self, messages: Sequence[BaseMessage]) -> List[dict]:
        """Extract paper results from ToolMessages for search_scientific_papers."""
        from langchain_core.messages import ToolMessage
        import ast
        import re
        
        papers = []
        errors = []
        seen_titles = set()  # Avoid duplicates
        
        for msg in messages:
            if not isinstance(msg, ToolMessage):
                continue
            
            content = msg.content
            if not content:
                continue
            
            content_str = str(content)
            
            # Try to parse the content as JSON or Python dict
            parsed = None
            try:
                if isinstance(content, dict):
                    parsed = content
                elif isinstance(content, str):
                    # Try JSON first
                    try:
                        parsed = json.loads(content)
                    except json.JSONDecodeError:
                        # Try Python literal (handles single quotes)
                        try:
                            # Replace single quotes with double quotes for JSON parsing
                            # Be careful with nested quotes
                            parsed = ast.literal_eval(content)
                        except (ValueError, SyntaxError):
                            pass
            except Exception:
                pass
            
            # If parsing succeeded and we have papers
            if isinstance(parsed, dict) and "papers" in parsed:
                for paper in parsed.get("papers", []):
                    if not isinstance(paper, dict):
                        continue
                    
                    # Skip error entries
                    if "error" in paper:
                        errors.append(paper.get("error", "Unknown error"))
                        continue
                    
                    title = paper.get("title", "") or paper.get("Title", "")
                    if title and title != "N/A" and title not in seen_titles:
                        seen_titles.add(title)
                        papers.append({
                            "title": title,
                            "authors": paper.get("authors", "") or paper.get("Authors", ""),
                            "abstract": paper.get("abstract", "") or paper.get("Abstract", ""),
                            "link": paper.get("link", "") or paper.get("Link", ""),
                            "source": paper.get("source", "arxiv"),
                        })
                
                # Also check for errors array
                if "errors" in parsed and parsed["errors"]:
                    for err in parsed["errors"]:
                        if isinstance(err, dict) and "error" in err:
                            errors.append(err["error"])
            
            # Fallback: try to extract from raw text if parsing failed
            elif "arxiv" in content_str.lower() or "title" in content_str.lower():
                # Try to find title patterns in raw text
                title_matches = re.findall(r"['\"]title['\"]:\s*['\"]([^'\"]+)['\"]", content_str, re.IGNORECASE)
                link_matches = re.findall(r"https?://arxiv\.org/abs/[\w\.]+", content_str)
                
                for i, title in enumerate(title_matches):
                    if title and title != "N/A" and title not in seen_titles:
                        seen_titles.add(title)
                        link = link_matches[i] if i < len(link_matches) else ""
                        papers.append({
                            "title": title,
                            "authors": "",
                            "abstract": "",
                            "link": link,
                            "source": "arxiv",
                        })
        
        # If we have errors but no papers, include error info
        if not papers and errors:
            return [{"title": "Errore nella ricerca", "abstract": "; ".join(set(errors[:3])), "link": "", "authors": "", "source": "error"}]
        
        return papers

    def _get_dataset_tool(self) -> Optional[DatasetQueryTool]:
        for tool in self._tools:
            if isinstance(tool, DatasetQueryTool):
                return tool
        return None

    @staticmethod
    def _format_number_it(value: float, preserve_decimals: Optional[int] = None) -> str:
        try:
            d = Decimal(str(value))
        except (InvalidOperation, ValueError):
            return str(value)

        sign = "-" if d.is_signed() else ""
        mag = d.copy_abs()
        tup = mag.as_tuple()
        digits = "".join(str(x) for x in tup.digits) or "0"
        exp = tup.exponent

        if exp >= 0:
            digits = digits + ("0" * exp)
            int_part = digits or "0"
            frac_part = ""
        else:
            places = -exp
            if len(digits) <= places:
                digits = digits.zfill(places + 1)
            int_part = digits[:-places] or "0"
            frac_part = digits[-places:]

        if preserve_decimals is None:
            frac_part = frac_part.rstrip("0")
        else:
            frac_part = (frac_part + ("0" * max(0, preserve_decimals - len(frac_part))))[:preserve_decimals]

        groups = []
        while len(int_part) > 3:
            groups.append(int_part[-3:])
            int_part = int_part[:-3]
        groups.append(int_part)
        int_grouped = ".".join(reversed(groups))

        if frac_part:
            return f"{sign}{int_grouped},{frac_part}"
        return f"{sign}{int_grouped}"

    @staticmethod
    def _extract_first_numeric(text: str) -> Optional[float]:
        pattern = re.compile(r"\d[\d\.\,\s\u00a0\u202f']*")
        for m in pattern.findall(text or ""):
            cleaned = m.replace("\u00a0", " ").replace("\u202f", " ").replace(" ", "")
            cleaned = cleaned.replace(".", "").replace(",", ".").replace("'", "")
            try:
                return float(cleaned)
            except ValueError:
                continue
        return None

    def _compute_dataset_number(self, natural_question: str) -> Optional[float]:
        tool = self._get_dataset_tool()
        if tool is None:
            return None
        try:
            result = tool._run(natural_question)  # internal call within app boundary
        except Exception:
            return None

        if isinstance(result, dict) and "result" in result and isinstance(result.get("result"), (int, float)):
            return float(result["result"])

        if isinstance(result, dict) and isinstance(result.get("results"), list):
            results = result["results"]
            if results:
                row = results[0]
                for key in ("total", "count", "sum", "avg", "value"):
                    val = row.get(key)
                    if isinstance(val, (int, float)):
                        return float(val)
                for _, val in row.items():
                    if isinstance(val, (int, float)):
                        return float(val)

        text_blob = ""
        if isinstance(result, dict):
            for k in ("info", "sql_executed"):
                v = result.get(k)
                if isinstance(v, str):
                    text_blob += " " + v
        return self._extract_first_numeric(text_blob)

    def _retrieve_relevant_history(
        self,
        messages: Sequence[BaseMessage],
        query: str,
        top_k: int = 4,
        max_snippet_chars: int = 800,
        max_total_chars: int = 2200,
    ) -> List[str]:
        """Retrieve the most relevant past chat snippets using vector search."""
        # Build corpus from non-system messages, excluding the latest user query itself
        corpus = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                continue
            if isinstance(msg, HumanMessage) and msg.content == query:
                continue
            content = (msg.content or "").strip()
            if not content:
                continue
            # Truncate each message to avoid huge payloads
            if len(content) > max_snippet_chars:
                content = content[:max_snippet_chars] + "... [troncato]"
            corpus.append((content, msg))

        if not corpus:
            return []

        # Create vectorstore
        vectorstore = InMemoryVectorStore.from_texts(
            texts=[c[0] for c in corpus],
            embedding=self._embeddings,
            metadatas=[{"role": "user" if isinstance(c[1], HumanMessage) else "assistant"} for c in corpus],
        )

        # Similarity search
        k = min(top_k, len(corpus))
        results = vectorstore.similarity_search(query, k=k)

        snippets: List[str] = []
        total_chars = 0
        for doc in results:
            snippet = doc.page_content.strip()
            if not snippet:
                continue
            # Ensure we do not exceed global cap
            if total_chars + len(snippet) > max_total_chars:
                break
            snippets.append(snippet)
            total_chars += len(snippet)

        return snippets

    def _generate_one_line(self, question: str, number_str: str) -> Optional[str]:
        try:
            llm = self._create_chat_without_tools(model=self._primary_model, temperature=0.1)
            prompt = (
                "Scrivi una sola riga in italiano che risponda direttamente alla domanda, "
                "contenendo il numero esatto fornito e pochissimo testo. "
                "Imita la forma della domanda. Non aggiungere spiegazioni.\n\n"
                f"Domanda: {question}\n"
                f"Numero: {number_str}\n\n"
                "Risposta (una riga, deve includere il numero esatto):"
            )
            resp = llm.invoke([HumanMessage(content=prompt)])
            line = (resp.content or "").strip().splitlines()[0].strip()
            return line if number_str in line else f"{line} {number_str}"
        except Exception:
            return None

    def _create_chat_without_tools(self, model: str, temperature: float) -> Any:
        """Create a plain chat model (no tool binding) for small internal prompts."""
        if self._llm_settings.provider == LlmProvider.OLLAMA:
            from langchain_ollama import ChatOllama

            return ChatOllama(model=model, temperature=temperature, base_url=self._llm_settings.ollama_base_url)

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature, api_key=self._llm_settings.openai_api_key)

    def _finalize_response(self, question: str, response_text: str) -> str:
        first_line = (response_text or "").splitlines()[0] if response_text else ""
        if self._extract_first_numeric(first_line) is not None:
            return response_text

        num = self._compute_dataset_number(question)
        if num is None:
            num = self._extract_first_numeric(response_text)
        if num is None:
            return response_text

        num_str = self._format_number_it(num)
        one_line = self._generate_one_line(question, num_str)
        if not one_line:
            one_line = f"{num_str}"

        rest = response_text or ""
        if rest.startswith(first_line):
            rest = rest[len(first_line):].lstrip("\n")
        return f"{one_line}\n\n{rest}".rstrip()

    def _manage_context(self, state: AgentState) -> dict:
        """Manage conversation context to avoid token limit issues."""
        messages = list(state["messages"])
        
        # Configuration
        MAX_MESSAGES = 3  # Keep only last N message pairs (user + assistant)
        MAX_MESSAGE_LENGTH = 50000  # Max characters per message
        
        # Count current messages
        message_count = len(messages)
        
        # If conversation is too long, trim it
        if message_count > MAX_MESSAGES:
            # Always keep system messages
            system_messages = [m for m in messages if isinstance(m, SystemMessage)]
            
            # Keep only the most recent messages (excluding system)
            recent_messages = [m for m in messages if not isinstance(m, SystemMessage)][-MAX_MESSAGES:]
            
            # Create a summary of removed context
            removed_count = len(messages) - len(system_messages) - len(recent_messages)
            
            if removed_count > 0:
                context_note = SystemMessage(
                    content=f"[Nota: {removed_count} messaggi precedenti rimossi per gestione contesto. "
                    f"Concentrati sulla richiesta corrente dell'utente.]"
                )
                messages = system_messages + [context_note] + recent_messages
        
        # Compress very long messages (like detailed statistics)
        compressed_messages = []
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.content and len(msg.content) > MAX_MESSAGE_LENGTH:
                # If it's a very long AI response, create a summary
                if "DBH" in msg.content or "distretto" in msg.content or "specie" in msg.content:
                    # Looks like dataset statistics - compress it
                    summary = (
                        "[Statistiche dataset precedenti - riepilogo compresso]\n"
                        "Dataset analizzato con successo. "
                        "Per nuove analisi o grafici, specifica la tua richiesta."
                    )
                    compressed_msg = AIMessage(content=summary)
                    compressed_messages.append(compressed_msg)
                else:
                    # Keep as is but truncate
                    truncated_content = msg.content[:MAX_MESSAGE_LENGTH] + "\n\n[... messaggio troncato per gestione contesto]"
                    compressed_msg = AIMessage(content=truncated_content)
                    compressed_messages.append(compressed_msg)
            else:
                compressed_messages.append(msg)
        
        return {
            "messages": compressed_messages,
            "message_count": len(compressed_messages)
        }

    def _optimize_query(self, state: AgentState) -> dict:
        """Optimize user query and break it into tasks.
        
        Simplified to create fewer tasks and avoid over-complication.
        """
        messages = state["messages"]
        
        # Get the last user message
        last_user_msg = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_user_msg = msg.content
                break
        
        if not last_user_msg:
            return {"optimized_query": None, "tasks": []}
        
        # For simple/short queries, skip optimization to save time
        if len(last_user_msg) < 50:
            return {
                "optimized_query": last_user_msg,
                "tasks": [last_user_msg],
            }
        
        # Use LLM to optimize query and create tasks (simplified prompt)
        optimizer_prompt = f"""Analizza la seguente domanda e crea 2-3 task semplici.

Domanda: {last_user_msg}

Rispondi SOLO in formato JSON:
{{
    "optimized_query": "domanda riformulata brevemente",
    "tasks": ["task 1", "task 2"]
}}

REGOLE:
- Massimo 3 task
- Task brevi e diretti
- Se la domanda è già chiara, restituisci la domanda originale con 1 task

Esempi:
- "Quanti pini ci sono?" → {{"optimized_query": "Conta i pini nel dataset", "tasks": ["Cercare pini nel dataset", "Contare risultati"]}}
- "Calcola CO2 per un albero di 30cm" → {{"optimized_query": "Calcola CO2 per DBH 30cm", "tasks": ["Calcolare CO2 con DBH 30cm"]}}"""
        
        try:
            optimizer_llm = self._create_chat_without_tools(model=self._fallback_model, temperature=1)
            
            response = optimizer_llm.invoke([HumanMessage(content=optimizer_prompt)])
            
            # Parse JSON response
            response_text = response.content.strip()
            
            # Extract JSON from markdown code blocks if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            optimization_result = json.loads(response_text)
            
            optimized_query = optimization_result.get("optimized_query", last_user_msg)
            tasks = optimization_result.get("tasks", [])
            
            # Add optimization info as a system message
            optimization_msg = SystemMessage(
                content=f"""Query ottimizzata: {optimized_query}

Task da completare:
{chr(10).join(f'{i+1}. {task}' for i, task in enumerate(tasks))}"""
            )
            
            return {
                "messages": [optimization_msg],
                "optimized_query": optimized_query,
                "tasks": tasks,
            }
            
        except Exception as e:
            # If optimization fails, continue with original query
            return {
                "optimized_query": last_user_msg,
                "tasks": [last_user_msg],
            }

    def _call_model(self, state: AgentState) -> dict:
        """Call the LLM model."""
        messages = state["messages"]
        
        # Check timeout before calling LLM
        budget_data = state.get("budget")
        if budget_data:
            budget = AgentBudget.from_dict(budget_data)
            elapsed = time.time() - budget.start_time
            if elapsed > budget.max_execution_time_seconds:
                timeout_response = (
                    f"⚠️ **Timeout raggiunto** ({elapsed:.0f}s)\n\n"
                    "La richiesta ha impiegato troppo tempo. "
                    "Prova a riformulare la domanda in modo più specifico.\n\n"
                    "**Suggerimenti:**\n"
                    "- Per calcoli CO2: specifica DBH (cm) e altezza (m)\n"
                    "- Per query dataset: chiedi qualcosa di semplice come \"quanti pini ci sono?\"\n"
                    "- Per grafici: specifica il tipo di grafico desiderato\n\n"
                    "Tool utilizzati: Nessuno (timeout)"
                )
                return {"messages": [AIMessage(content=timeout_response)]}

        # Add system message if not present
        if not any(isinstance(m, SystemMessage) for m in messages):
            system_msg = SystemMessage(
                content="""You are a helpful tree evaluation assistant with access to:

1. **CO2 Calculation Tool**: Calculate CO2 sequestration and biomass for individual trees given their measurements.
2. **CO2 Aggregate Tool**: Calculate TOTAL/AVERAGE CO2 and biomass for a group of trees (by species, district, etc.) or the whole dataset. Use this for questions like "total CO2 for pines" or "stock di carbonio".
3. **Environmental Estimation Tool**: Compute volume, biomass, and carbon stock using alternative formulas.
4. **Dataset Query Tool**: Query a real Vienna trees dataset (BAUMKATOGD) with filtering, aggregation, and statistics.
4. **Chart Generation Tool**: Create interactive visualizations (bar, pie, line, scatter, histogram, box plots) from the dataset.
5. **Map Generation Tool**: Create interactive maps showing tree locations (markers, clusters, heatmaps). ONLY available for Milano dataset which has GPS coordinates.
6. **Advanced Biomass & Volume Equations**: Calculate Volume (Heyer, General, Simplified), Biomass (Leaf, Stem, Root, Total), and Allometric Relations using specific scientific formulas.
7. **Species List Query Tool**: Query a plant species list (taxonomy + traits) to provide botanical context (family/order/class, growth form, leaf type, etc.).
8. **Paper Search Tool**: Search arXiv and PubMed for scientific papers. Returns title, authors, abstract, and link to each paper.

Guidelines:
- When users ask about CO2 or carbon sequestration for specific measurements (single tree), use the CO2 calculation tool.
- When users ask about aggregate CO2 (total, average, stock) for the dataset or specific groups (e.g. "pines"), use the CO2 Aggregate Tool.
- When users ask about the dataset (counts, species, districts, statistics), use the dataset query tool.
- When users ask for botanical context about plant species (family/order/class, species code, growth form, leaf type, synonyms), use the species list query tool.
- When users ask to create, visualize, or show charts/graphs, use the chart generation tool.
- When users ask to show trees on a MAP, visualize distribution geographically, or create a map, use the map generation tool. NOTE: Maps are ONLY available for the Milano dataset (has GPS coordinates). Vienna dataset does NOT have coordinates.
- Use specific biomass/volume tools when the user asks for those specific equations (Heyer, Leaf Biomass, etc.).
- When users ask about scientific research, publications, papers, or literature, use the paper search tool. ALWAYS include the paper links in your response.
- Always provide clear, helpful responses in Italian.
- If you need more information, ask the user.
- When using tools, explain the results in a user-friendly way.
- For wood density, use species-specific values if known, otherwise default to 0.6 g/cm³.

**CRITICAL RULES - ALWAYS FOLLOW:**

1. **ALWAYS include units of measurement** in your answers:
   - Volumes: m³ (metri cubi)
   - Biomass: kg, t (tonnellate)
   - CO2: kg CO2, t CO2
   - Diameters: cm (centimetri)
   - Heights: m (metri)
   - Ratios: no unit (R/S = 0.24 significa rapporto adimensionale)
   - Counts: numero alberi, specie, record
   
2. **ALWAYS cite which tool(s) you used** at the end of your response:
   - Format: "Tool utilizzati: [nome tool]"
   - Examples:
     * "Tool utilizzati: Dataset Query Tool"
     * "Tool utilizzati: CO2 Calculation Tool"
     * "Tool utilizzati: Dataset Query Tool, Chart Generation Tool"
     * "Tool utilizzati: Allometric Relation Tool"

3. **ALWAYS cite scientific sources** when using calculation tools:
   - When tool results include "source" or "sources" fields with scientific papers, you MUST include them in your response.
   - When tool results include "data_source" with open data information, you MUST cite the data provider and URL.
   - When tool results include "formulas" with source information, cite the relevant papers for each formula used.
   - Format for scientific sources: "📚 Fonte: [title] - [url]"
   - Format for data sources: "📊 Dati: [provider] - [url]"
   - Example with scientific source:
     "La CO₂ sequestrata è 1.250 kg CO2
     
     Formule utilizzate:
     - AGB (Chave et al. 2014): https://www.researchgate.net/publication/262197290
     - Carbon content: https://www.researchgate.net/publication/259443596
     
     Tool utilizzati: CO2 Calculation Tool"
   - Example with data source:
     "A Milano la specie più diffusa è Platanus × acerifolia: 14.005 alberi
     
     📊 Dati: Comune di Milano - Open Data
     https://dati.comune.milano.it/dataset/ds447-infogeo-aree-verdi-alberi-702eb2e7
     
     Tool utilizzati: Dataset Query Tool"

4. **Complete answer format**:
   ```
   [Prima riga: risposta diretta con numero e unità di misura]
   
   [Dettagli aggiuntivi se necessari]
   
   [Fonti scientifiche/dati se presenti nel risultato del tool]
   
   Tool utilizzati: [nome tool(s)]
   ```

Answer style policy (CRITICAL for evaluation):
- First line must contain the final answer in Italian with the exact number, units of measurement, and minimal text.
- ALWAYS include units: kg, m³, cm, m, t CO2, etc.
- Prefer Italian numeric formatting: thousands with dot, decimals with comma (e.g., 33.612 alberi; 0,24 R/S; 15.000 kg CO2).
- Keep additional details only after a blank line, and keep them concise.
- Mirror user phrasing when possible to maximize textual similarity.
- ALWAYS end with "Tool utilizzati: [nome tool]"

Examples with units, sources, and tool citation:
  - "A Vienna ci sono 23 distretti\n\n📊 Dati: City of Vienna - Open Data\nhttps://www.data.gv.at/katalog/dataset/stadt-wien_baumkatasterderstadtwien\n\nTool utilizzati: Dataset Query Tool"
  - "A Milano la specie più diffusa è Platanus × acerifolia: 14.005 alberi\n\n📊 Dati: Comune di Milano - Open Data\nhttps://dati.comune.milano.it/dataset/ds447-infogeo-aree-verdi-alberi-702eb2e7\n\nTool utilizzati: Dataset Query Tool"
  - "La CO₂ sequestrata è 1.250 kg CO2\n\nCalcolo basato su DBH 30cm, altezza 15m, densità legno 0.56 g/cm³.\n\n📚 Fonti scientifiche:\n- Chave et al. (2014) - AGB equation: https://www.researchgate.net/publication/262197290\n- Carbon Content of Tree Tissues: https://www.researchgate.net/publication/259443596\n\nTool utilizzati: CO2 Calculation Tool"
  - "Il volume stimato è 2,5 m³\n\nCalcolo con formula V = a * D² * H\n\n📚 Fonte: Allometric relationships for volume and biomass for stone pine\nhttps://www.researchgate.net/publication/256199126\n\nTool utilizzati: General Volume Tool"
  - "La biomassa fogliare è 45,3 kg\n\nFormula: Leaf = e^(-7.21) * (D² * H)^0.6 * age^3.2 * 1.28\n\n📚 Fonte: Development of Allometric Equations for Argan Trees\nhttps://www.researchgate.net/publication/380957635\n\nTool utilizzati: Leaf Biomass Tool"

If a computation is needed but measurements are missing, state the short requirement in one line, then ask for the needed values in the next lines.

**IMPORTANT - Chart Tool Usage:**
When you use the chart generation tool and it returns chart data with "success": true, you MUST include the COMPLETE JSON response in your answer. Format it exactly like this:

Ho creato il grafico richiesto.

CHART_DATA_START
{the complete JSON from the tool}
CHART_DATA_END

Do not modify or summarize the JSON - include it verbatim between CHART_DATA_START and CHART_DATA_END markers.

**IMPORTANT - Map Tool Usage:**
When you use the map generation tool and it returns map data with "success": true, you MUST include the COMPLETE JSON response in your answer. Format it exactly like this:

Ho creato la mappa richiesta.

MAP_DATA_START
{the complete JSON from the tool}
MAP_DATA_END

Do not modify or summarize the JSON - include it verbatim between MAP_DATA_START and MAP_DATA_END markers.
IMPORTANT: Maps require GPS coordinates. Only the Milano dataset has coordinates. If the user tries to generate a map with Vienna dataset, explain that maps are not available for Vienna.

Common wood densities (g/cm³):
- Acer (Acero): 0.56
- Tilia (Tiglio): 0.49
- Carpinus (Carpino): 0.75
- Gleditsia: 0.62
- Aesculus (Ippocastano): 0.53
- Quercus (Quercia): 0.75
- Fraxinus (Frassino): 0.69
- Betula (Betulla): 0.65
"""
            )
            messages = [system_msg] + list(messages)

        def _truncate(msg: BaseMessage, max_len: int = 800) -> BaseMessage:
            """Return a shallow copy of msg with truncated content."""
            content = (msg.content or "") if hasattr(msg, "content") else ""
            if len(content) > max_len:
                content = content[:max_len] + "... [troncato]"
            if isinstance(msg, HumanMessage):
                return HumanMessage(content=content)
            if isinstance(msg, AIMessage):
                return AIMessage(content=content)
            if isinstance(msg, SystemMessage):
                return SystemMessage(content=content)
            return msg

        # Retrieve relevant history snippets to reduce full-context length
        last_user_content = None
        last_user_msg = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_user_content = msg.content
                last_user_msg = msg
                break

        context_block = None
        if last_user_content:
            # Ultra-conservative: disable context_block to minimize tokens
            snippets = []

        # Build a minimal message list: all system messages, last assistant (if any), last user, plus context block
        system_messages = [m for m in messages if isinstance(m, SystemMessage)]
        last_assistant = None
        if last_user_msg:
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    last_assistant = msg
                    break

        minimal_messages = []
        minimal_messages.extend(system_messages[:1])  # only the first system message
        if context_block:
            minimal_messages.append(context_block)
        if last_assistant:
            minimal_messages.append(last_assistant)
        if last_user_msg:
            minimal_messages.append(last_user_msg)

        # Truncate each message to keep tokens low
        minimal_messages = [_truncate(m, max_len=400) for m in minimal_messages]

        try:
            response = self._llm.invoke(minimal_messages)
        except Exception as e:
            # Fallback to lighter model on rate limit / request too large
            if "rate_limit" in str(e).lower() or "429" in str(e) or "request too large" in str(e).lower():
                try:
                    # Rebind tools to fallback LLM
                    for tool in self._tools:
                        if hasattr(tool, "_llm"):
                            object.__setattr__(tool, "_llm", self._fallback_llm)
                    self._llm = self._fallback_llm.bind_tools(self._tools)
                    # Also drop assistant context to shrink prompt further
                    minimal_fallback = []
                    minimal_fallback.extend(system_messages[:1])
                    if last_user_msg:
                        minimal_fallback.append(last_user_msg)
                    minimal_fallback = [_truncate(m, max_len=300) for m in minimal_fallback]
                    response = self._llm.invoke(minimal_fallback)
                except Exception:
                    raise
            else:
                raise
        try:
            user_question = None
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    user_question = msg.content
            if user_question and isinstance(response, AIMessage) and response.content:
                fixed = self._finalize_response(user_question, response.content)
                if fixed and fixed != response.content:
                    response = AIMessage(content=fixed)
        except Exception:
            pass
        return {"messages": [response]}

    def _should_continue(self, state: AgentState) -> Literal["continue", "validate"]:
        """Determine if we should continue to tools or validate response."""
        messages = state["messages"]
        last_message = messages[-1]

        # If the LLM makes a tool call, continue to tools
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "continue"

        # Otherwise, validate the response
        return "validate"
    
    def _validate_response(self, state: AgentState) -> dict:
        """Validate if the response adequately answers the user's question."""
        messages = state["messages"]
        tasks = state.get("tasks", [])
        optimized_query = state.get("optimized_query", "")
        
        # Get original user question
        user_question = None
        for msg in messages:
            if isinstance(msg, HumanMessage) and not msg.content.startswith("Query ottimizzata"):
                user_question = msg.content
                break
        
        # Get agent's response
        agent_response = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                agent_response = msg.content
                break
        
        if not user_question or not agent_response:
            return {"validation_result": {"is_complete": True, "feedback": ""}}
        
        # Create validation prompt
        validation_prompt = f"""Valuta se la seguente risposta risponde adeguatamente alla domanda dell'utente.

Domanda originale: {user_question}

Query ottimizzata: {optimized_query}

Task da completare:
{chr(10).join(f'{i+1}. {task}' for i, task in enumerate(tasks))}

Risposta fornita: {agent_response}

Analizza TUTTI questi criteri:
1. Tutti i task sono stati completati?
2. La risposta è accurata e completa?
3. La risposta risponde effettivamente alla domanda?
4. CRITICO: Se la risposta contiene numeri/misure, include le UNITÀ DI MISURA? (kg, m³, cm, kg CO2, etc.)
5. CRITICO: La risposta cita i TOOL UTILIZZATI alla fine? (formato: "Tool utilizzati: [nome tool]")

IMPORTANTE: Se mancano unità di misura o citazione dei tool, la risposta è INCOMPLETA.

Rispondi in formato JSON:
{{
    "is_complete": true/false,
    "completed_tasks": ["lista", "dei", "task", "completati"],
    "missing_tasks": ["lista", "dei", "task", "mancanti"],
    "has_units": true/false (se sono presenti unità di misura dove necessario),
    "has_tool_citation": true/false (se è presente "Tool utilizzati:"),
    "feedback": "breve feedback su cosa manca o cosa migliorare (se incompleto)"
}}"""
        
        try:
            validator_llm = self._create_chat_without_tools(model=self._fallback_model, temperature=1)
            
            response = validator_llm.invoke([HumanMessage(content=validation_prompt)])
            
            # Parse JSON response
            response_text = response.content.strip()
            
            # Extract JSON from markdown code blocks if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            validation_result = json.loads(response_text)
            
            # Check if response is truly complete (including units and tool citation)
            is_complete = validation_result.get("is_complete", True)
            has_units = validation_result.get("has_units", True)
            has_tool_citation = validation_result.get("has_tool_citation", True)
            
            # Override is_complete if units or tool citation are missing
            if not has_units or not has_tool_citation:
                is_complete = False
                missing_items = []
                if not has_units:
                    missing_items.append("unità di misura")
                if not has_tool_citation:
                    missing_items.append("citazione dei tool utilizzati")
                
                # Add to feedback
                current_feedback = validation_result.get('feedback', '')
                additional_feedback = f"Mancano: {', '.join(missing_items)}."
                validation_result['feedback'] = f"{current_feedback} {additional_feedback}".strip()
                validation_result['is_complete'] = False
            
            # If incomplete, add feedback as system message for retry
            if not is_complete:
                feedback_msg = SystemMessage(
                    content=f"""⚠️ Validazione risposta:
Task mancanti: {', '.join(validation_result.get('missing_tasks', []))}

Feedback: {validation_result.get('feedback', '')}

REGOLE OBBLIGATORIE:
- SEMPRE includere unità di misura (kg, m³, cm, kg CO2, etc.)
- SEMPRE terminare con "Tool utilizzati: [nome tool]"

Per favore, completa la risposta affrontando i task mancanti e rispettando le regole obbligatorie."""
                )
                return {
                    "messages": [feedback_msg],
                    "validation_result": validation_result,
                }
            
            return {"validation_result": validation_result}
            
        except Exception as e:
            # If validation fails, assume complete
            return {"validation_result": {"is_complete": True, "feedback": ""}}
    
    def _should_retry(self, state: AgentState) -> Literal["complete", "retry"]:
        """Determine if we should retry or complete based on validation."""
        validation_result = state.get("validation_result", {})
        retry_count = int(state.get("retry_count") or 0)
        max_retries = 1  # Reduced from 2 to prevent long validation loops
        
        # Check if response is complete
        is_complete = validation_result.get("is_complete", True)
        
        if is_complete:
            return "complete"
        if retry_count >= max_retries:
            return "complete"
        return "retry"

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

        # Add current message
        messages.append(HumanMessage(content=message))

        # Initialize fresh budget for this request
        initial_budget = AgentBudget()
        
        # Run graph with reduced recursion limit for safety
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
            },
            config={"recursion_limit": 30},  # Reduced from 80 for safety
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

        # Add current message
        messages.append(HumanMessage(content=message))

        # Track execution
        final_response = None
        retry_count = 0
        max_retries = 1  # Reduced from 2 to prevent long validation loops
        chart_data_json = None  # Track chart data if generated
        map_data_json = None  # Track map data if generated

        # Initialize fresh budget for this request
        initial_budget = AgentBudget()

        # Stream from graph with updates mode to see each node
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
            },
            config={"recursion_limit": 30},  # Reduced from 80 for safety
            stream_mode="updates",
        ):
            # event is a dict with node_name: node_output
            for node_name, node_output in event.items():
                
                # Emit reasoning for each node
                if node_name == "context_manager":
                    # Show context management info if messages were compressed
                    message_count = node_output.get("message_count", 0)
                    original_count = len(messages)
                    
                    if message_count < original_count:
                        reasoning = f"🧹 **Gestione Contesto**\n\n"
                        reasoning += f"Messaggi originali: {original_count}\n"
                        reasoning += f"Messaggi ottimizzati: {message_count}\n"
                        reasoning += f"Contesto lungo compresso per evitare limiti di token.\n"
                        yield {"type": "reasoning", "content": reasoning}
                
                elif node_name == "query_optimizer":
                    optimized = node_output.get("optimized_query", "")
                    tasks = node_output.get("tasks", [])
                    if optimized:
                        reasoning = f"🔍 **Ottimizzazione Query**\n\n"
                        reasoning += f"Query ottimizzata: *{optimized}*\n\n"
                        if tasks:
                            reasoning += "**Task identificati:**\n\n"
                            reasoning += "<ol>\n"
                            for task in tasks:
                                reasoning += f"<li>{task}</li>\n"
                            reasoning += "</ol>\n\n"
                        yield {"type": "reasoning", "content": reasoning}
                
                elif node_name == "agent":
                    # Check if agent is calling tools or responding
                    node_messages = node_output.get("messages", [])
                    if node_messages:
                        last_msg = node_messages[-1]
                        if isinstance(last_msg, AIMessage):
                            if last_msg.tool_calls:
                                # Agent is calling tools - show detailed parameters
                                reasoning = f"🛠️ **Chiamata Tool**\n\n"
                                for tool_call in last_msg.tool_calls:
                                    tool_name = tool_call.get("name", "unknown")
                                    tool_args = tool_call.get("args", {})
                                    
                                    reasoning += f"- **Tool**: `{tool_name}`\n"
                                    
                                    # Show parameters based on tool type
                                    if tool_name == "query_tree_dataset":
                                        natural_q = tool_args.get("natural_query", "N/A")
                                        reasoning += f"  - **Query**: _{natural_q}_\n"
                                    elif tool_name == "calculate_co2":
                                        dbh = tool_args.get("dbh_cm", "N/A")
                                        height = tool_args.get("height_m", "N/A")
                                        wood_density = tool_args.get("wood_density", "N/A")
                                        reasoning += f"  - **DBH**: {dbh} cm\n"
                                        reasoning += f"  - **Altezza**: {height} m\n"
                                        reasoning += f"  - **Densità legno**: {wood_density} g/cm³\n"
                                    elif tool_name == "calculate_co2_aggregate":
                                        query = tool_args.get("natural_query", "N/A")
                                        reasoning += f"  - **Query**: _{query}_\n"
                                    elif tool_name == "estimate_environment":
                                        dbh = tool_args.get("dbh_cm", "N/A")
                                        height = tool_args.get("height_m", "N/A")
                                        reasoning += f"  - **DBH**: {dbh} cm\n"
                                        reasoning += f"  - **Altezza**: {height} m\n"
                                    elif tool_name == "generate_chart":
                                        chart_type = tool_args.get("chart_type", "N/A")
                                        reasoning += f"  - **Tipo grafico**: {chart_type}\n"
                                    
                                    reasoning += "\n"
                                
                                yield {"type": "reasoning", "content": reasoning}
                            elif last_msg.content and not last_msg.tool_calls:
                                # Agent has a response (might be intermediate or final)
                                final_response = last_msg.content
                
                elif node_name == "tools":
                    # Tool execution completed - show detailed results
                    node_messages = node_output.get("messages", [])
                    
                    if node_messages:
                        for msg in node_messages:
                            # Tool messages contain the results
                            if hasattr(msg, 'content') and msg.content:
                                try:
                                    # Try to parse as JSON for structured results
                                    import json
                                    if isinstance(msg.content, str):
                                        result_data = json.loads(msg.content)
                                    else:
                                        result_data = msg.content
                                    
                                    # Check if this is chart data (chart tool returns "chart_json" key)
                                    if "chart_json" in result_data and result_data.get("success"):
                                        chart_data_json = json.dumps(result_data, ensure_ascii=False, indent=2)
                                        print(f"[DEBUG] Chart data captured! Length: {len(chart_data_json)} chars")
                                    
                                    # Check if this is map data (map tool returns "map_html" key)
                                    if "map_html" in result_data and result_data.get("success"):
                                        map_data_json = json.dumps(result_data, ensure_ascii=False, indent=2)
                                        print(f"[DEBUG] Map data captured! Length: {len(map_data_json)} chars")
                                    
                                    reasoning = f"✅ **Risultati Tool**\n\n"
                                    
                                    # Show SQL query if it's a dataset query
                                    if "sql_executed" in result_data:
                                        sql = result_data.get("sql_executed", "")
                                        reasoning += f"**Query SQL generata:**\n```sql\n{sql}\n```\n\n"
                                    
                                    # Show row count and vector search info
                                    if "row_count" in result_data:
                                        row_count = result_data.get("row_count", 0)
                                        
                                        # Check if vector search was applied
                                        if result_data.get("vector_search_applied", False):
                                            total_found = result_data.get("total_rows_found", row_count)
                                            reasoning += f"🔍 **Vector Search Applicata**\n"
                                            reasoning += f"📊 **Righe totali trovate**: {total_found}\n"
                                            reasoning += f"✨ **Top risultati più rilevanti**: {row_count}\n"
                                            if "info" in result_data:
                                                reasoning += f"ℹ️  {result_data['info']}\n"
                                        else:
                                            reasoning += f"📊 **Righe trovate**: {row_count}\n"
                                            
                                            # Show truncation warning if present (old style)
                                            if "warning" in result_data:
                                                warning = result_data.get("warning")
                                                reasoning += f"⚠️  **Attenzione**: {warning}\n"
                                        
                                        reasoning += "\n"
                                    
                                    # Show result preview for dataset queries
                                    if "results" in result_data:
                                        results = result_data.get("results", [])
                                        if results and len(results) > 0:
                                            reasoning += f"**Primi risultati:**\n\n"
                                            reasoning += "<ol>\n"
                                            # Show first 3 results as preview
                                            for row in results[:3]:
                                                reasoning += "<li>"
                                                # Show main fields
                                                if "genus_species" in row:
                                                    reasoning += f"Specie: {row['genus_species']} "
                                                if "count" in row:
                                                    reasoning += f"Count: {row['count']} "
                                                if "district" in row:
                                                    reasoning += f"Distretto: {row['district']} "
                                                if "trunk_circumference" in row:
                                                    reasoning += f"Circonferenza: {row['trunk_circumference']}cm "
                                                reasoning += "</li>\n"
                                            reasoning += "</ol>\n"
                                            
                                            if len(results) > 3:
                                                reasoning += f"... e altri {len(results) - 3} risultati\n"
                                    
                                    # Show single value results
                                    elif "result" in result_data and "column" in result_data:
                                        result_val = result_data.get("result")
                                        column_name = result_data.get("column")
                                        reasoning += f"**{column_name}**: {result_val}\n"
                                    
                                    # Show CO2 calculation results
                                    if "co2_sequestration_kg" in result_data:
                                        co2 = result_data.get("co2_sequestration_kg", 0)
                                        reasoning += f"🌱 **CO2 sequestrato**: {co2} kg\n"
                                    
                                    yield {"type": "reasoning", "content": reasoning}
                                    
                                except (json.JSONDecodeError, AttributeError):
                                    # If not JSON, just show completion message
                                    reasoning = f"✅ **Tool Eseguito**\n\nElaborazione risultati...\n"
                                    yield {"type": "reasoning", "content": reasoning}

                elif node_name == "budget_check":
                    # Show budget status if limit was hit
                    if node_output.get("budget_exceeded"):
                        node_messages = node_output.get("messages", [])
                        if node_messages:
                            last_msg = node_messages[-1]
                            if isinstance(last_msg, AIMessage) and last_msg.content:
                                final_response = last_msg.content
                        yield {"type": "reasoning", "content": "⚠️ **Budget Limit**\n\nLimite di esecuzione raggiunto. Interruzione per prevenire loop infiniti.\n"}
                    else:
                        status = node_output.get("budget_status", {})
                        if status:
                            reasoning = f"✓ **Budget Check**\n\n"
                            reasoning += f"Tool calls: {status.get('total_tool_calls', 'N/A')}\n"
                            reasoning += f"Tempo: {status.get('elapsed_time', 'N/A')}\n"
                            yield {"type": "reasoning", "content": reasoning}

                elif node_name == "tool_loop_guard":
                    # If the loop guard decided to stop, it injects an AIMessage with a user-facing prompt.
                    node_messages = node_output.get("messages", [])
                    if node_messages:
                        last_msg = node_messages[-1]
                        if isinstance(last_msg, AIMessage) and last_msg.content:
                            final_response = last_msg.content
                            yield {"type": "reasoning", "content": "🛑 **Stop Anti-Loop**\n\nRilevata ripetizione della stessa chiamata tool. Interrompo ed entro in modalità chiarimento.\n"}
                    else:
                        # Replan path: no user-facing message, we attempt a recovery step.
                        if node_output.get("tool_loop_action") == "replan":
                            yield {"type": "reasoning", "content": "🔁 **Recovery Anti-Loop**\n\nRilevata ripetizione della stessa chiamata tool. Provo a cambiare strategia (replanning) invece di fermarmi subito.\n"}

                elif node_name == "tool_loop_replanner":
                    yield {"type": "reasoning", "content": "🧠 **Replanning**\n\nSto riformulando il prossimo passo per evitare di ripetere la stessa query/tool e provare una strada alternativa.\n"}
                
                elif node_name == "validator":
                    validation = node_output.get("validation_result", {})
                    is_complete = validation.get("is_complete", True)
                    
                    if is_complete:
                        reasoning = f"✓ **Validazione Completata**\n\nLa risposta è completa e accurata.\n"
                        yield {"type": "reasoning", "content": reasoning}
                    else:
                        retry_count += 1
                        if retry_count > max_retries:
                            reasoning = f"⚠️ **Validazione**\n\nRaggiunto limite retry. Proseguo con la risposta attuale.\n"
                            yield {"type": "reasoning", "content": reasoning}
                            # Don't break - let the graph complete naturally
                        else:
                            missing = validation.get("missing_tasks", [])
                            feedback = validation.get("feedback", "")
                            reasoning = f"⚠️ **Validazione (Tentativo {retry_count})**\n\n"
                            if missing:
                                reasoning += f"Task mancanti: {', '.join(missing)}\n"
                            if feedback:
                                reasoning += f"\n{feedback}\n"
                            reasoning += "\nRielaborazione risposta...\n"
                            yield {"type": "reasoning", "content": reasoning}
        
        # Yield final response
        if final_response:
            # If we have chart data but it's not in the response, add it automatically
            print(f"[DEBUG] Final response check - chart_data_json: {chart_data_json is not None}, has markers: {'CHART_DATA_START' in final_response}")
            if chart_data_json and "CHART_DATA_START" not in final_response:
                print(f"[DEBUG] Adding chart data to response!")
                final_response = f"{final_response}\n\nCHART_DATA_START\n{chart_data_json}\nCHART_DATA_END"
            
            # If we have map data but it's not in the response, add it automatically
            print(f"[DEBUG] Final response check - map_data_json: {map_data_json is not None}, has markers: {'MAP_DATA_START' in final_response}")
            if map_data_json and "MAP_DATA_START" not in final_response:
                print(f"[DEBUG] Adding map data to response!")
                final_response = f"{final_response}\n\nMAP_DATA_START\n{map_data_json}\nMAP_DATA_END"
            try:
                last_user = None
                for msg in messages:
                    if isinstance(msg, HumanMessage):
                        last_user = msg.content
                if last_user:
                    fixed = self._finalize_response(last_user, final_response)
                    if fixed:
                        final_response = fixed
            except Exception:
                pass
            yield {"type": "response", "content": final_response}

