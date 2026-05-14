from __future__ import annotations

from typing import List, Optional

import streamlit as st

from streamlit_app.models import ChatMessage, Conversation, UserLlmSettings
from streamlit_app.service import ChatService
from streamlit_app.ui_messages import render_messages
from streamlit_app.ui_reasoning import (
    format_reasoning_step,
    inject_custom_css,
    render_reasoning_box,
)
from streamlit_app.ui_sidebar import render_sidebar
from streamlit_app.ui_visuals import (
    extract_chart_from_response,
    extract_map_from_response,
    render_chart_data,
    render_map,
)


class ChatUI:
    """Streamlit UI layer for the chat demo with conversation management."""

    _OPENAI_AUTH_METHODS = {
        "API key OpenAI Platform": "api_key",
        "Login ChatGPT (OAuth)": "codex_oauth",
    }
    _ANTHROPIC_AUTH_METHODS = {
        "Login Claude (OAuth)": "oauth",
        "API key Anthropic": "api_key",
    }
    _OPENAI_API_KEYS_URL = "https://platform.openai.com/settings/organization/api-keys"
    _OPENAI_CODEX_DEVICE_PAIRING_URL = "https://auth.openai.com/codex/device"
    _CLAUDE_CODE_SETUP_URL = "https://docs.anthropic.com/en/docs/claude-code/setup"
    _ANTHROPIC_CONSOLE_KEYS_URL = "https://console.anthropic.com/settings/keys"

    def __init__(self, service: ChatService) -> None:
        self._service = service

    def _build_openai_chat_model_options(
        self, current: str, *, for_codex_oauth: bool = False
    ) -> List[str]:
        """Return OpenAI chat model options, ensuring the current selection is included.

        When ``for_codex_oauth`` is True only models accepted by the ChatGPT
        backend are listed (see ``CODEX_OAUTH_SUPPORTED_MODELS``); platform-only
        models like ``gpt-5`` or ``gpt-4o`` would otherwise return a 400
        "unsupported-model" error.
        """
        if for_codex_oauth:
            from streamlit_app.llm.codex_backend import CODEX_OAUTH_SUPPORTED_MODELS

            defaults = list(CODEX_OAUTH_SUPPORTED_MODELS)
        else:
            defaults = [
                "gpt-5.5",
                "gpt-5.4",
                "gpt-5.4-mini",
                "gpt-5.3-codex",
                "gpt-5.2-codex",
                "gpt-5",
                "gpt-5-mini",
                "gpt-4o",
                "gpt-4o-mini",
                "o3-mini",
                "o1",
            ]
        current_clean = (current or "").strip()
        options: List[str] = []
        if current_clean and current_clean not in defaults:
            options.append(current_clean)
        options.extend(defaults)
        options.append("Altro…")
        seen = set()
        out: List[str] = []
        for m in options:
            if m and m not in seen:
                seen.add(m)
                out.append(m)
        return out

    def _build_openai_embedding_model_options(self, current: str) -> List[str]:
        """Return OpenAI embedding model options, ensuring the current selection is included."""
        defaults = [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        ]
        current_clean = (current or "").strip()
        options = []
        if current_clean and current_clean not in defaults:
            options.append(current_clean)
        options.extend(defaults)
        options.append("Altro…")
        seen = set()
        out: List[str] = []
        for m in options:
            if m and m not in seen:
                seen.add(m)
                out.append(m)
        return out

    def _build_anthropic_chat_model_options(
        self, current: str, *, for_oauth: bool = False
    ) -> List[str]:
        """Return Claude chat model options, ensuring the current selection is included."""
        if for_oauth:
            from streamlit_app.llm.anthropic_backend import CLAUDE_OAUTH_SUPPORTED_MODELS

            defaults = list(CLAUDE_OAUTH_SUPPORTED_MODELS)
        else:
            defaults = [
                "claude-sonnet-4-5",
                "claude-opus-4-5",
                "claude-haiku-4-5",
                "claude-3-7-sonnet-latest",
                "claude-3-5-sonnet-latest",
                "claude-3-5-haiku-latest",
            ]
        current_clean = (current or "").strip()
        options: List[str] = []
        if current_clean and current_clean not in defaults:
            options.append(current_clean)
        options.extend(defaults)
        options.append("Altro…")
        seen = set()
        out: List[str] = []
        for model in options:
            if model and model not in seen:
                seen.add(model)
                out.append(model)
        return out

    def _resolve_openai_model_selection(self, selected: str, custom_value: str) -> str:
        """Resolve a selectbox value with optional custom input."""
        if selected == "Altro…":
            return (custom_value or "").strip()
        return (selected or "").strip()

    def _resolve_model_selection(self, selected: str, custom_value: str) -> str:
        """Resolve a generic model selectbox value with optional custom input."""
        if selected == "Altro…":
            return (custom_value or "").strip()
        return (selected or "").strip()

    def _openai_auth_method_label(self, method: str) -> str:
        """Return the UI label for a persisted OpenAI auth method."""
        for label, value in self._OPENAI_AUTH_METHODS.items():
            if value == method:
                return label
        return "API key OpenAI Platform"

    def _anthropic_auth_method_label(self, method: str) -> str:
        """Return the UI label for a persisted Anthropic auth method."""
        for label, value in self._ANTHROPIC_AUTH_METHODS.items():
            if value == method:
                return label
        return "Login Claude (OAuth)"

    def _ensure_session(self) -> None:
        """Initialize session state variables."""
        if "user_id" not in st.session_state:
            st.session_state.user_id = "guest"
        if "llm_settings" not in st.session_state:
            st.session_state.llm_settings = self._service.get_user_llm_settings(st.session_state.get("user_id", "guest"))
        if "current_conversation_id" not in st.session_state:
            st.session_state.current_conversation_id: Optional[int] = None
        if "messages" not in st.session_state:
            st.session_state.messages: List[ChatMessage] = []
        if "conversations" not in st.session_state:
            st.session_state.conversations: List[Conversation] = []
        if "editing_conversation_id" not in st.session_state:
            st.session_state.editing_conversation_id: Optional[int] = None

    def _load_conversations(self) -> None:
        """Load all conversations for the current user."""
        st.session_state.conversations = self._service.list_user_conversations(st.session_state.user_id)

    def _load_conversation_messages(self, conversation_id: int) -> None:
        """Load messages for a specific conversation."""
        st.session_state.messages = self._service.get_conversation_messages(conversation_id)
        st.session_state.current_conversation_id = conversation_id

    def _create_new_conversation(self) -> None:
        """Create a new conversation for the current user."""
        conversation = self._service.create_new_conversation(st.session_state.user_id)
        st.session_state.conversations.insert(0, conversation)
        st.session_state.current_conversation_id = conversation.id
        st.session_state.messages = []

    def _render_sidebar(self) -> None:
        """Render the sidebar with user settings and conversation list."""
        render_sidebar(self)

    def _render_messages(self) -> None:
        """Render all messages in the current conversation."""
        render_messages(self)

    def render(self) -> None:
        """Main render method for the chat UI."""
        self._ensure_session()
        st.set_page_config(page_title="Tree Evaluator Chat", page_icon="🌳", layout="centered")
        inject_custom_css()
        st.title("🌳 Tree Evaluator — AI Chat")
        st.caption("Chatbot intelligente con LangChain/LangGraph per analisi alberi e dataset Vienna")

        self._render_sidebar()

        settings: UserLlmSettings = st.session_state.llm_settings
        block_reason = st.session_state.get("llm_block_reason")

        # Main chat area
        if st.session_state.current_conversation_id is None:
            st.info("👈 Seleziona una conversazione dalla sidebar o creane una nuova per iniziare!")
            
            # Show welcome message with instructions
            if (
                settings.provider == "openai"
                and settings.openai_auth_method == "api_key"
                and not settings.openai_api_key
            ):
                st.warning("""
                ### 🔑 Configurazione richiesta
                
                Per usare il chatbot intelligente, inserisci la tua **OpenAI API Key** nelle impostazioni (sidebar in alto).
                
                **Come ottenere una chiave:**
                1. Vai su [platform.openai.com](https://platform.openai.com/api-keys)
                2. Crea un account o effettua il login
                3. Genera una nuova API key (inizia con `sk-...`)
                4. Copia e incolla la chiave nelle impostazioni
                
                **Senza chiave API**, il chatbot userà risposte demo di fallback.
                """)
        else:
            self._render_messages()

            if block_reason:
                st.warning(block_reason)
                return

            # Chat input
            if prompt := st.chat_input("Scrivi un messaggio…"):
                # Check if API key is provided (warn but continue)
                if (
                    settings.provider == "openai"
                    and settings.openai_auth_method == "api_key"
                    and not settings.openai_api_key
                ):
                    st.info("ℹ️ Nessuna API key configurata. Userò risposte demo. Inserisci la chiave OpenAI nelle impostazioni per usare l'agent intelligente.")
                
                user_id = st.session_state.user_id
                conversation_id = st.session_state.current_conversation_id
                
                # Add user message immediately
                user_msg = self._service.add_user_message(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    content=prompt
                )
                st.session_state.messages.append(user_msg)
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(user_msg.content)
                
                # Stream assistant response
                with st.chat_message("assistant"):
                    # Create container for reasoning steps
                    reasoning_placeholder = st.empty()
                    response_placeholder = st.empty()
                    # IMPORTANT: use stable containers for rich media.
                    # `st.empty()` + `components.html()` (Plotly iframe) often only appears after a rerun
                    # (e.g., when switching conversation). Containers render reliably in the same run.
                    chart_placeholder = st.container()
                    map_placeholder = st.container()
                    
                    reasoning_steps = []
                    full_response = ""
                    chart_data = None
                    map_data = None
                    
                    # Stream from agent
                    for chunk in self._service.stream_reply(
                        user_id=user_id,
                        conversation_id=conversation_id,
                        last_user_message=prompt,
                        openai_api_key=(
                            settings.openai_api_key or None
                            if settings.openai_auth_method == "api_key"
                            else None
                        )
                    ):
                        chunk_type = chunk.get("type", "response")
                        chunk_content = chunk.get("content", "")
                        
                        if chunk_type == "reasoning":
                            # Add reasoning step
                            reasoning_steps.append(chunk_content)
                            # Update reasoning display with styled box
                            reasoning_html = render_reasoning_box(reasoning_steps, is_loading=True)
                            reasoning_placeholder.markdown(reasoning_html, unsafe_allow_html=True)
                        
                        elif chunk_type == "response":
                            # Update final response
                            full_response = chunk_content
                            # Show reasoning in collapsed expander with better styling
                            if reasoning_steps:
                                with reasoning_placeholder:
                                    with st.expander("🧠 Processo di ragionamento", expanded=False):
                                        # Use styled HTML for each step
                                        steps_html = "".join([format_reasoning_step(s) for s in reasoning_steps])
                                        st.markdown(f'<div class="reasoning-content">{steps_html}</div>', unsafe_allow_html=True)
                            
                            # Check if response contains chart data
                            text_content, extracted_chart = extract_chart_from_response(full_response)
                            
                            # Check if response contains map data
                            text_content, extracted_map = extract_map_from_response(text_content)
                            
                            if extracted_chart and extracted_chart.get("success"):
                                chart_data = extracted_chart
                            if extracted_map and extracted_map.get("success"):
                                map_data = extracted_map
                            
                            if chart_data or map_data:
                                response_placeholder.markdown(text_content + "▌")
                            else:
                                response_placeholder.markdown(full_response + "▌")
                    
                    # Final update without cursor
                    if full_response:
                        text_content, extracted_chart = extract_chart_from_response(full_response)
                        text_content, extracted_map = extract_map_from_response(text_content)
                        
                        has_visualization = False
                        
                        # Display text content
                        if extracted_chart and extracted_chart.get("success"):
                            has_visualization = True
                            if text_content:
                                response_placeholder.markdown(text_content)
                            
                            # Display chart
                            with chart_placeholder:
                                try:
                                    render_chart_data(extracted_chart)
                                except Exception as e:
                                    st.error(f"Errore nella visualizzazione del grafico: {e}")
                        
                        # Display map if present
                        if extracted_map and extracted_map.get("success"):
                            has_visualization = True
                            if not extracted_chart:  # Only show text if not already shown for chart
                                if text_content:
                                    response_placeholder.markdown(text_content)
                            
                            # Display map
                            with map_placeholder:
                                render_map(extracted_map)
                        
                        # If no visualization, show full response
                        if not has_visualization:
                            response_placeholder.markdown(full_response)
                
                # Add assistant message to session state
                # (already persisted by stream_reply, just update UI state)
                from streamlit_app.models import ChatMessage
                assistant_msg = ChatMessage.new(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    role="assistant",
                    content=full_response
                )
                st.session_state.messages.append(assistant_msg)
