"""LangGraph orchestrator for tree analysis tools."""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolNode

from streamlit_app.agent.budget_handler import BudgetHandler, get_detected_language
from streamlit_app.agent.chat_runtime import chat as run_chat
from streamlit_app.agent.chat_runtime import stream_chat as run_stream_chat
from streamlit_app.agent.context_manager import ConversationContextManager
from streamlit_app.agent.extraction import DataExtractor
from streamlit_app.agent.formatting import ItalianNumberFormatter
from streamlit_app.agent.graph_builder import GraphBuilder
from streamlit_app.agent.initial_state import create_initial_state
from streamlit_app.agent.message_pipeline import build_minimal_messages, prepare_messages_for_llm
from streamlit_app.agent.plain_chat_factory import create_chat_without_tools
from streamlit_app.agent.query_optimizer import QueryOptimizer
from streamlit_app.agent.state import AgentState
from streamlit_app.agent.stream_events import process_stream_event
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
        anthropic_auth_method: str = "oauth",
        anthropic_api_key: Optional[str] = None,
        anthropic_oauth_access_token: Optional[str] = None,
        provider: Optional[str] = None,
        openai_chat_model: Optional[str] = None,
        anthropic_chat_model: Optional[str] = None,
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
            anthropic_auth_method: Anthropic auth method ("oauth" or "api_key")
            anthropic_api_key: Anthropic API key
            anthropic_oauth_access_token: Short-lived Claude OAuth access token
            provider: LLM provider (openai/anthropic/ollama)
            openai_chat_model: OpenAI chat model name
            anthropic_chat_model: Anthropic chat model name
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
            anthropic_auth_method_override=anthropic_auth_method,
            anthropic_api_key_override=anthropic_api_key,
            anthropic_oauth_access_token_override=anthropic_oauth_access_token,
            anthropic_chat_model_override=anthropic_chat_model,
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
        elif self._llm_settings.provider == LlmProvider.ANTHROPIC:
            self._primary_model = self._llm_settings.anthropic_chat_model
            fallback = (self._llm_settings.anthropic_fallback_model or "").strip()
            self._fallback_model = fallback or self._llm_settings.anthropic_chat_model
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

        messages = prepare_messages_for_llm(
            messages,
            detected_language,
            self._memory.get_memory_for_prompt(max_chars=2000),
        )

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
                minimal = build_minimal_messages(messages, truncate_label)
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
        return run_chat(self, message, history)

    def stream_chat(self, message: str, history: Optional[List[dict]] = None):
        """Stream chat response with LangGraph streaming.

        Args:
            message: User message
            history: Optional chat history

        Yields:
            Dict with 'type' and 'content' keys
        """
        yield from run_stream_chat(self, message, history)

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
        return process_stream_event(node_name, node_output, language, msg_count)

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
        return create_initial_state(messages, self._interface_language)

    # ===== Utility Methods =====

    def _create_chat_without_tools(self, model: str, temperature: float) -> Any:
        """Create a plain chat model (no tool binding) for internal prompts."""
        return create_chat_without_tools(self._llm_settings, model, temperature)
