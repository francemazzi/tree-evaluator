"""Tree Evaluator Agent - Main orchestrator for tree analysis tools.

This module provides the main TreeEvaluatorAgent class that coordinates
multiple specialized tools for tree analysis, CO2 calculations, and dataset queries.

The agent is built on LangGraph and uses modular components:
- ToolInitializer: Handles tool setup and configuration
- GraphBuilder: Constructs the LangGraph workflow
- QueryOptimizer: Optimizes queries and plans tool execution
- BudgetHandler: Manages budget constraints and summaries
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolNode

from streamlit_app.agent.budget import AgentBudget
from streamlit_app.agent.budget_handler import BudgetHandler, get_detected_language
from streamlit_app.agent.context_manager import ConversationContextManager
from streamlit_app.agent.extraction import DataExtractor
from streamlit_app.agent.formatting import ItalianNumberFormatter
from streamlit_app.agent.graph_builder import GraphBuilder
from streamlit_app.agent.prompts import SystemPrompts
from streamlit_app.agent.query_optimizer import QueryOptimizer
from streamlit_app.agent.state import AgentState
from streamlit_app.agent.streaming_handler import StreamingHandler
from streamlit_app.agent.failure_tracker_adapter import FailureTrackerAdapter
from streamlit_app.agent.memory import PersistentMemory
from streamlit_app.agent.tool_initializer import ToolInitializer
from streamlit_app.agent.transcript import SessionTranscript
from streamlit_app.llm.factory import LlmFactory, LlmProvider, LlmSettings, LlmSettingsReader

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class TreeEvaluatorAgent:
    """LangGraph-based agent that orchestrates tree evaluation tools."""

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        openai_auth_method: str = "api_key",
        openai_codex_access_token: Optional[str] = None,
        openai_codex_account_id: Optional[str] = None,
        openai_codex_is_fedramp: bool = False,
        provider: Optional[str] = None,
        openai_chat_model: Optional[str] = None,
        openai_embedding_model: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
        ollama_chat_model: Optional[str] = None,
        ollama_embedding_model: Optional[str] = None,
        custom_db_path: Optional[Path] = None,
        custom_table_name: Optional[str] = None,
        dataset_column_roles: Optional[dict] = None,
        data_description: str = "",
        dataset_preset: str = "vienna",
        interface_language: str = "it",
        session_id: Optional[str] = None,
    ) -> None:
        """Initialize the agent with tools and LLM.

        Args:
            openai_api_key: OpenAI API key
            openai_auth_method: OpenAI auth method ("api_key" or "codex_oauth")
            openai_codex_access_token: Short-lived ChatGPT OAuth access token
            openai_codex_account_id: Optional ChatGPT account/workspace id
            openai_codex_is_fedramp: Whether the account requires FedRAMP routing
            provider: LLM provider (openai/ollama)
            openai_chat_model: OpenAI chat model name
            openai_embedding_model: OpenAI embedding model name
            ollama_base_url: Ollama base URL
            ollama_chat_model: Ollama chat model name
            ollama_embedding_model: Ollama embedding model name
            custom_db_path: Optional path to custom SQLite database
            custom_table_name: Optional custom table name in the database
            dataset_column_roles: Optional role hints inferred from uploaded dataset profiling
            data_description: Optional description of the data for context
            dataset_preset: Preset dataset to use ("vienna", "milano")
            interface_language: Language for agent responses ("it" or "en")
        """
        # Initialize LLM settings and factory
        self._llm_settings: LlmSettings = LlmSettingsReader().read(
            openai_api_key_override=openai_api_key,
            openai_auth_method_override=openai_auth_method,
            openai_codex_access_token_override=openai_codex_access_token,
            openai_codex_account_id_override=openai_codex_account_id,
            openai_codex_is_fedramp_override=openai_codex_is_fedramp,
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
        self._setup_llm_models()
        self._base_llm = self._llm_factory.create_chat_model()
        self._fallback_llm = self._llm_factory.create_fallback_chat_model()
        self._embeddings = self._llm_factory.create_embeddings()

        # Store interface language
        self._interface_language = interface_language if interface_language in ["it", "en"] else "it"

        # Session tracking, transcript logging, and persistent memory
        self._session_id = session_id or uuid.uuid4().hex[:12]
        self._transcript = SessionTranscript(self._session_id)
        self._memory = PersistentMemory(self._session_id)

        # Initialize tools using ToolInitializer
        tool_initializer = ToolInitializer(
            base_llm=self._base_llm,
            fallback_llm=self._fallback_llm,
            embeddings=self._embeddings,
            interface_language=self._interface_language,
        )
        self._tools = tool_initializer.initialize_tools(
            custom_db_path,
            custom_table_name,
            data_description,
            dataset_preset,
            dataset_column_roles=dataset_column_roles or {},
        )
        self._dynamic_tools_summary = tool_initializer.dynamic_tools_summary

        # Initialize LLM with tools bound
        self._llm = self._base_llm.bind_tools(self._tools)
        self._tool_node = ToolNode(self._tools)

        # Initialize utility classes
        self._context_manager = ConversationContextManager(self._embeddings)
        self._extractor = DataExtractor()
        self._formatter = ItalianNumberFormatter()
        self._tool_loop_manager = FailureTrackerAdapter()

        # Initialize handlers
        self._query_optimizer = QueryOptimizer(
            interface_language=self._interface_language,
            fallback_model=self._fallback_model,
            create_chat_without_tools=self._create_chat_without_tools,
            dynamic_tools_summary=self._dynamic_tools_summary,
        )
        self._budget_handler = BudgetHandler(
            fallback_model=self._fallback_model,
            create_chat_without_tools=self._create_chat_without_tools,
        )

        # Build graph using GraphBuilder
        graph_builder = GraphBuilder(self._tools)
        self._graph = graph_builder.build(
            manage_context=self._manage_context,
            optimize_query=self._query_optimizer.optimize_query,
            check_budget=self._check_budget,
            call_model=self._call_model,
            run_tools=self._run_tools,
            guard_tool_loop=self._guard_tool_loop,
            replan_after_tool_loop=self._replan_after_tool_loop,
            validate_response=self._validate_response,
            increment_retry_count=self._increment_retry_count,
        )

    def _setup_llm_models(self) -> None:
        """Setup primary and fallback model names based on provider."""
        if self._llm_settings.provider == LlmProvider.OLLAMA:
            self._primary_model = self._llm_settings.ollama_chat_model
            fallback = (self._llm_settings.ollama_fallback_model or "").strip()
            self._fallback_model = fallback or self._llm_settings.ollama_chat_model
        else:
            self._primary_model = self._llm_settings.openai_chat_model
            self._fallback_model = (
                self._llm_settings.openai_chat_model
                if self._llm_settings.openai_auth_method == "codex_oauth"
                else self._llm_settings.openai_fallback_model
            )

    # ===== Node Methods =====

    def _manage_context(self, state: AgentState) -> dict:
        """Manage conversation context to avoid token limit issues."""
        messages = state.get("messages") or []
        return self._context_manager.manage_context(messages)

    def _call_model(self, state: AgentState) -> dict:
        """Call the LLM model."""
        messages = state["messages"]
        detected_language = get_detected_language(state, self._interface_language)

        # Prepare messages with system prompt
        messages = self._prepare_messages_for_llm(messages, detected_language)

        self._transcript.log_llm_call(self._primary_model, len(messages))
        try:
            response = self._llm.invoke(messages)
            self._transcript.log_llm_response(self._primary_model)
        except Exception as e:
            self._transcript.log_error("llm_call", str(e))
            response = self._handle_llm_error(e, messages, detected_language)

        return {"messages": [response]}

    def _check_budget(self, state: AgentState) -> dict:
        """Check budget and log budget stops to the session transcript."""
        result = self._budget_handler.check_budget(state)
        if result.get("budget_exceeded"):
            self._transcript.log_budget_exceeded(result.get("budget_status", {}))
        return result

    def _run_tools(self, state: AgentState) -> dict:
        """Run LangGraph tools while recording structured transcript events."""
        pending_tool_calls = self._pending_tool_calls(state.get("messages") or [])
        for tool_call in pending_tool_calls:
            self._transcript.log_tool_call(
                tool_call.get("name", "unknown"),
                tool_call.get("args", {}),
            )

        started_at = time.monotonic()
        try:
            result = self._tool_node.invoke(state)
        except Exception as exc:
            for tool_call in pending_tool_calls:
                self._transcript.log_error(tool_call.get("name", "unknown"), str(exc))
            raise

        duration_ms = (time.monotonic() - started_at) * 1000
        for message in result.get("messages", []):
            if isinstance(message, ToolMessage):
                self._transcript.log_tool_result(
                    getattr(message, "name", "unknown") or "unknown",
                    message.content,
                    duration_ms=duration_ms,
                )
        return result

    @staticmethod
    def _pending_tool_calls(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Return pending tool calls from the most recent AI message."""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                return list(msg.tool_calls)
        return []

    def _prepare_messages_for_llm(
        self, messages: List[BaseMessage], language: str
    ) -> List[BaseMessage]:
        """Prepare messages for LLM invocation with proper truncation."""
        # Remove existing system messages and add fresh one
        messages = [m for m in messages if not isinstance(m, SystemMessage)]
        system_prompt = SystemPrompts.get_system_prompt(language)
        memory_section = self._memory.get_memory_for_prompt(max_chars=2000)
        if memory_section:
            system_prompt = system_prompt + "\n\n" + memory_section
        system_msg = SystemMessage(content=system_prompt)
        messages = [system_msg] + list(messages)

        # Build minimal message list
        truncate_label = "... [truncated]" if language == "en" else "... [troncato]"
        minimal_messages = self._build_minimal_messages(messages, truncate_label)

        return minimal_messages

    def _build_minimal_messages(
        self, messages: List[BaseMessage], truncate_label: str
    ) -> List[BaseMessage]:
        """Build minimal message list for LLM to save tokens."""
        system_messages = [m for m in messages if isinstance(m, SystemMessage)]

        # Find last user message
        last_user_msg = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_user_msg = msg
                break

        # Find last AIMessage with tool_calls and its ToolMessages
        last_ai_with_tools = None
        tool_messages_after_ai = []

        for i, msg in enumerate(messages):
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                last_ai_with_tools = msg
                tool_messages_after_ai = []
                for j in range(i + 1, len(messages)):
                    if isinstance(messages[j], ToolMessage):
                        tool_messages_after_ai.append(messages[j])
                    elif isinstance(messages[j], AIMessage):
                        break

        # Build minimal list
        minimal_messages = []
        minimal_messages.extend(system_messages[:1])
        if last_user_msg:
            minimal_messages.append(last_user_msg)

        if last_ai_with_tools:
            minimal_messages.append(last_ai_with_tools)
            minimal_messages.extend(tool_messages_after_ai)

        # Truncate messages
        def truncate(msg: BaseMessage, max_len: int) -> BaseMessage:
            content = (msg.content or "") if hasattr(msg, "content") else ""
            if len(content) > max_len:
                content = content[:max_len] + truncate_label
            if isinstance(msg, HumanMessage):
                return HumanMessage(content=content)
            if isinstance(msg, AIMessage):
                return AIMessage(content=content, tool_calls=getattr(msg, "tool_calls", None) or [])
            if isinstance(msg, SystemMessage):
                return SystemMessage(content=content)
            if isinstance(msg, ToolMessage):
                return ToolMessage(
                    content=content,
                    tool_call_id=getattr(msg, "tool_call_id", ""),
                    name=getattr(msg, "name", ""),
                )
            return msg

        return [
            truncate(m, 1500 if isinstance(m, ToolMessage) else 400)
            for m in minimal_messages
        ]

    def _handle_llm_error(
        self, error: Exception, messages: List[BaseMessage], language: str
    ) -> AIMessage:
        """Handle LLM errors with fallback strategy."""
        error_str = str(error).lower()
        is_rate_limit = "rate_limit" in error_str or "429" in error_str or "request too large" in error_str

        if is_rate_limit:
            logger.warning(f"Rate limit hit, switching to fallback model: {error}")
            try:
                fallback_llm = self._fallback_llm.bind_tools(self._tools)
                truncate_label = "... [truncated]" if language == "en" else "... [troncato]"
                minimal = self._build_minimal_messages(messages, truncate_label)
                return fallback_llm.invoke(minimal)
            except Exception:
                raise

        raise error

    def _guard_tool_loop(self, state: AgentState) -> dict:
        """Guard against tool call loops."""
        messages = state.get("messages") or []
        result = self._tool_loop_manager.check_for_loops(messages, state)
        if result.get("tool_loop_detected"):
            details = result.get("tool_loop_details") or {}
            self._transcript.log_loop_detected(
                tool_name=details.get("tool_name", "unknown"),
                reason=str(details.get("reason", "")),
                call_count=details.get("call_count", 0),
                action=result.get("tool_loop_action", "continue"),
            )
        return result

    def _replan_after_tool_loop(self, state: AgentState) -> dict:
        """Recover from repeated tool calls by asking the agent to self-reflect."""
        messages = state.get("messages") or []
        replan_msg = self._tool_loop_manager.create_replan_prompt(state, messages)
        replan_count = int(state.get("tool_loop_replan_count") or 0) + 1
        self._transcript.log_replan("tool_loop", replan_count)

        return {
            "messages": [replan_msg],
            "tool_loop_action": "continue",
            "tool_loop_detected": False,
            "tool_loop_replan_count": replan_count,
        }

    def _validate_response(self, state: AgentState) -> dict:
        """Validate if the response adequately answers the user's question."""
        return {"validation_result": {"is_complete": True, "feedback": ""}}

    def _increment_retry_count(self, state: AgentState) -> dict:
        """Increment retry counter."""
        current = int(state.get("retry_count") or 0)
        return {"retry_count": current + 1}

    # ===== Public Methods =====

    def chat(self, message: str, history: Optional[List[dict]] = None) -> str:
        """Chat with the agent.

        Args:
            message: User message
            history: Optional chat history as list of dicts with 'role' and 'content' keys

        Returns:
            Agent response as string
        """
        messages = self._convert_history_to_messages(history)
        messages.append(HumanMessage(content=message))
        self._transcript.log_user_message(message)

        initial_state = self._create_initial_state(messages)
        result = self._graph.invoke(initial_state, config={"recursion_limit": 30})

        final_message = result["messages"][-1]
        response_text = final_message.content if isinstance(final_message, AIMessage) else str(final_message.content)
        self._transcript.log_agent_response(response_text[:300] if response_text else "")
        self._save_response_facts(response_text)
        return response_text

    def stream_chat(self, message: str, history: Optional[List[dict]] = None):
        """Stream chat response with LangGraph streaming.

        Args:
            message: User message
            history: Optional chat history

        Yields:
            Dict with 'type' and 'content' keys
        """
        messages = self._convert_history_to_messages(history)
        messages.append(HumanMessage(content=message))
        self._transcript.log_user_message(message)

        final_response = None
        retry_count = 0
        max_retries = 1
        chart_data_json = None
        map_data_json = None

        initial_state = self._create_initial_state(messages)

        for event in self._graph.stream(
            initial_state,
            config={"recursion_limit": 30},
            stream_mode="updates",
        ):
            current_language = self._interface_language
            if "detected_language" in event.get("query_optimizer", {}):
                current_language = event["query_optimizer"].get(
                    "detected_language", self._interface_language
                )

            for node_name, node_output in event.items():
                result = self._process_stream_event(
                    node_name, node_output, current_language, len(messages)
                )

                if result:
                    if result.get("type") == "final_response":
                        final_response = result.get("content")
                    elif "final_response" in result:
                        final_response = result.pop("final_response")
                        if result.get("type"):
                            yield result
                    elif result.get("type") == "chart_data":
                        chart_data_json = result.get("data")
                    elif result.get("type") == "map_data":
                        map_data_json = result.get("data")
                    elif result.get("type") == "reasoning":
                        yield result

        # Yield final response
        if final_response:
            self._transcript.log_agent_response(final_response[:300])

            if chart_data_json and "CHART_DATA_START" not in final_response:
                final_response = f"{final_response}\n\nCHART_DATA_START\n{chart_data_json}\nCHART_DATA_END"

            if map_data_json and "MAP_DATA_START" not in final_response:
                final_response = f"{final_response}\n\nMAP_DATA_START\n{map_data_json}\nMAP_DATA_END"

            self._save_response_facts(final_response)
            yield {"type": "response", "content": final_response}

    def _save_response_facts(self, response_text: str) -> None:
        """Extract key facts from agent response and save to persistent memory."""
        if not response_text:
            return
        try:
            key_facts = self._extractor.extract_key_facts(
                [AIMessage(content=response_text)]
            )
            for fact in key_facts[:3]:
                self._memory.save_fact(fact, category="result")
        except Exception:
            pass  # Memory saving is best-effort

    @property
    def memory(self) -> PersistentMemory:
        """Access persistent memory for this session."""
        return self._memory

    def _process_stream_event(
        self, node_name: str, node_output: Any, language: str, msg_count: int
    ) -> Optional[Dict]:
        """Process a single streaming event."""
        if node_name == "language_detector":
            return StreamingHandler.handle_language_detector_event(node_output, language)
        elif node_name == "context_manager":
            return StreamingHandler.handle_context_manager_event(node_output, msg_count, language)
        elif node_name == "query_optimizer":
            return StreamingHandler.handle_query_optimizer_event(node_output, language)
        elif node_name == "agent":
            result = StreamingHandler.handle_agent_event(node_output, language)
            return result
        elif node_name == "tools":
            result, chart_json, map_json = StreamingHandler.handle_tools_event(node_output, language)
            if chart_json:
                return {"type": "chart_data", "data": chart_json}
            if map_json:
                return {"type": "map_data", "data": map_json}
            return result
        elif node_name == "budget_check":
            return StreamingHandler.handle_budget_check_event(node_output, language)
        elif node_name == "tool_loop_guard":
            return StreamingHandler.handle_tool_loop_guard_event(node_output, language)
        elif node_name == "tool_loop_replanner":
            from streamlit_app.agent.translations import get_translation
            return {
                "type": "reasoning",
                "content": f"{get_translation('replanning', language)}\n\n{get_translation('reformulating_step', language)}\n",
            }
        elif node_name == "validator":
            result, _ = StreamingHandler.handle_validator_event(node_output, 0, 1, language)
            return result

        return None

    def _convert_history_to_messages(
        self, history: Optional[List[dict]]
    ) -> List[BaseMessage]:
        """Convert chat history to LangChain messages."""
        messages: List[BaseMessage] = []
        if history:
            for msg in history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
        return messages

    def _create_initial_state(self, messages: List[BaseMessage]) -> Dict[str, Any]:
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
            "detected_language": self._interface_language,
            "tool_plan": None,
            "available_tools_summary": None,
        }

    # ===== Utility Methods =====

    def _create_chat_without_tools(self, model: str, temperature: float) -> Any:
        """Create a plain chat model (no tool binding) for internal prompts."""
        if self._llm_settings.provider == LlmProvider.OLLAMA:
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=model,
                temperature=temperature,
                base_url=self._llm_settings.ollama_base_url,
            )

        if self._llm_settings.openai_auth_method == "codex_oauth":
            from streamlit_app.llm.codex_backend import (
                ChatGPTCodexBackendChatModel,
                resolve_codex_oauth_model,
            )

            return ChatGPTCodexBackendChatModel(
                model_name=resolve_codex_oauth_model(model),
                access_token=self._llm_settings.openai_codex_access_token or "",
                account_id=self._llm_settings.openai_codex_account_id,
                is_fedramp_account=self._llm_settings.openai_codex_is_fedramp,
            )

        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=self._llm_settings.openai_api_key,
        )
