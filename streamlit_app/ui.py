from __future__ import annotations

import json
import hashlib
import os
from typing import List, Optional, Tuple

import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from streamlit_app.llm.ollama_client import OllamaClient
from streamlit_app.models import ChatMessage, Conversation, UserLlmSettings
from streamlit_app.service import ChatService


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
        with st.sidebar:
            st.header("⚙️ Settings")
            
            settings: UserLlmSettings = st.session_state.llm_settings
            st.session_state.llm_block_reason = None

            # Language selector
            language_options = {"🇮🇹 Italiano": "it", "🇬🇧 English": "en"}
            language_labels = list(language_options.keys())
            current_lang_index = 0 if settings.interface_language == "it" else 1
            
            selected_language_label = st.selectbox(
                "🌐 Lingua / Language",
                options=language_labels,
                index=current_lang_index,
                help="Scegli la lingua dell'assistente / Choose assistant language",
            )
            
            selected_language = language_options[selected_language_label]
            if selected_language != settings.interface_language:
                settings.interface_language = selected_language
                self._service.save_user_llm_settings(settings)
                self._service._agent = None  # Reset agent to apply new language
                st.rerun()

            provider = st.selectbox(
                "Provider LLM",
                options=["openai", "anthropic", "ollama"],
                index=(
                    ["openai", "anthropic", "ollama"].index(settings.provider)
                    if settings.provider in {"openai", "anthropic", "ollama"}
                    else 0
                ),
                help="Scegli se usare OpenAI (GPT), Anthropic (Claude) oppure Ollama in locale.",
            )

            if provider != settings.provider:
                settings.provider = provider
                self._service.save_user_llm_settings(settings)
                self._service._agent = None
                st.rerun()

            if provider == "openai":
                auth_labels = list(self._OPENAI_AUTH_METHODS.keys())
                current_auth_label = self._openai_auth_method_label(settings.openai_auth_method)
                selected_auth_label = st.radio(
                    "Metodo autenticazione OpenAI",
                    options=auth_labels,
                    index=auth_labels.index(current_auth_label),
                    help=(
                        "Puoi usare una API key della piattaforma OpenAI oppure un login ChatGPT OAuth "
                        "con refresh token."
                    ),
                )
                selected_auth_method = self._OPENAI_AUTH_METHODS[selected_auth_label]
                if selected_auth_method != settings.openai_auth_method:
                    settings.openai_auth_method = selected_auth_method
                    self._service.save_user_llm_settings(settings)
                    self._service._agent = None
                    st.rerun()

                if settings.openai_auth_method == "api_key":
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        new_api_key = st.text_input(
                            "OpenAI API Key",
                            value=settings.openai_api_key,
                            type="password",
                            key="api_key_input",
                            help="Inserisci la tua chiave API OpenAI (sk-...). Verrà salvata in modo persistente.",
                            placeholder="sk-..."
                        )
                        st.link_button(
                            "Apri pagina API keys",
                            self._OPENAI_API_KEYS_URL,
                            use_container_width=True,
                        )
                    with col2:
                        st.write("")
                        st.write("")
                        if st.button("Salva", use_container_width=True):
                            settings.openai_api_key = new_api_key.strip()
                            settings.openai_auth_method = "api_key"
                            self._service.save_user_llm_settings(settings)
                            self._service._agent = None
                            st.success("✅ Impostazioni salvate!")
                            st.rerun()
                else:
                    if st.button("Genera codice dispositivo", use_container_width=True):
                        try:
                            from streamlit_app.llm.openai_oauth import request_device_code

                            st.session_state.openai_device_pairing = request_device_code()
                        except Exception as exc:
                            st.error(f"Impossibile generare il codice dispositivo: {exc}")

                    device_pairing = st.session_state.get("openai_device_pairing")
                    if device_pairing:
                        st.caption("Inserisci questo codice nella pagina di device pairing.")
                        st.code(str(device_pairing.get("user_code") or ""), language="text")
                        if device_pairing.get("expires_at"):
                            st.caption(f"Scade: {device_pairing['expires_at']}")
                        st.link_button(
                            "OpenAI Codex Device Pairing",
                            self._OPENAI_CODEX_DEVICE_PAIRING_URL,
                            use_container_width=True,
                        )
                        st.caption("Pair in browser with a device code.")
                        if st.button("Ho completato il pairing: salva token", use_container_width=True):
                            try:
                                from streamlit_app.llm.openai_oauth import (
                                    DeviceAuthorizationPending,
                                    complete_device_code_login,
                                )

                                token_payload = complete_device_code_login(
                                    str(device_pairing.get("device_auth_id") or ""),
                                    str(device_pairing.get("user_code") or ""),
                                )
                                refresh_token = str(token_payload.get("refresh_token") or "").strip()
                                if not refresh_token:
                                    raise ValueError("il refresh token non è presente nella risposta OAuth.")
                                settings.openai_codex_oauth_token = refresh_token
                                settings.openai_auth_method = "codex_oauth"
                                self._service.save_user_llm_settings(settings)
                                self._service._agent = None
                                del st.session_state.openai_device_pairing
                                st.success("✅ Login ChatGPT salvato!")
                                st.rerun()
                            except DeviceAuthorizationPending:
                                st.warning("Il pairing non risulta ancora completato. Inserisci il codice nel browser e riprova.")
                            except Exception as exc:
                                st.error(f"Impossibile completare il pairing: {exc}")

                    with st.expander("Ho già un refresh token", expanded=False):
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            new_refresh_token = st.text_input(
                                "Refresh token OAuth",
                                value=settings.openai_codex_oauth_token,
                                type="password",
                                key="openai_codex_oauth_token_input",
                                help="Se hai già un refresh token, puoi incollarlo direttamente senza rifare il login.",
                                placeholder="refresh token"
                            )
                        with col2:
                            st.write("")
                            st.write("")
                            if st.button("Salva token", use_container_width=True):
                                settings.openai_codex_oauth_token = new_refresh_token.strip()
                                settings.openai_auth_method = "codex_oauth"
                                self._service.save_user_llm_settings(settings)
                                self._service._agent = None
                                st.success("✅ Refresh token salvato!")
                                st.rerun()

                    st.info(
                        "Il codice ora viene generato direttamente da questa sidebar. "
                        "Apri il link di device pairing, inserisci il codice visualizzato qui e poi premi "
                        "“Ho completato il pairing: salva token”."
                    )
                    if settings.openai_codex_oauth_token:
                        st.success("Device pairing configurato.")
                        st.caption(
                            "La chat userà il backend ChatGPT/Codex associato al tuo account."
                        )
                        st.session_state.llm_block_reason = None
                    else:
                        st.session_state.llm_block_reason = (
                            "Completa il login ChatGPT OAuth o incolla un refresh token, "
                            "oppure usa una API key OpenAI Platform."
                        )

                is_codex_oauth = settings.openai_auth_method == "codex_oauth"
                chat_options = self._build_openai_chat_model_options(
                    settings.openai_chat_model,
                    for_codex_oauth=is_codex_oauth,
                )
                chat_help = (
                    "Con login ChatGPT (OAuth) sono ammessi solo i modelli del Codex CLI "
                    "(es. gpt-5.5, gpt-5.4). I modelli della Platform (gpt-5, gpt-4o…) "
                    "sono accettati solo con API key."
                    if is_codex_oauth
                    else "Seleziona il modello chat OpenAI. Puoi scegliere “Altro…” per inserire un nome custom."
                )
                chat_selected = st.selectbox(
                    "Modello OpenAI (chat)",
                    options=chat_options,
                    index=(
                        chat_options.index(settings.openai_chat_model)
                        if settings.openai_chat_model in chat_options
                        else 0
                    ),
                    help=chat_help,
                )
                custom_chat = ""
                if chat_selected == "Altro…":
                    custom_chat = st.text_input(
                        "Nome modello chat (custom)",
                        value=settings.openai_chat_model,
                        help=(
                            "Inserisci il nome esatto del modello (es. gpt-5.5 per OAuth, "
                            "gpt-5 per API key)."
                        ),
                    )

                embed_options = self._build_openai_embedding_model_options(settings.openai_embedding_model)
                embed_selected = st.selectbox(
                    "Modello OpenAI (embeddings)",
                    options=embed_options,
                    index=(
                        embed_options.index(settings.openai_embedding_model)
                        if settings.openai_embedding_model in embed_options
                        else 0
                    ),
                    help="Seleziona il modello embeddings OpenAI. Puoi scegliere “Altro…” per inserire un nome custom.",
                )
                custom_embed = ""
                if embed_selected == "Altro…":
                    custom_embed = st.text_input(
                        "Nome modello embeddings (custom)",
                        value=settings.openai_embedding_model,
                        help="Inserisci il nome esatto del modello embeddings (es. text-embedding-3-small).",
                    )

                resolved_chat = self._resolve_openai_model_selection(chat_selected, custom_chat)
                resolved_embed = self._resolve_openai_model_selection(embed_selected, custom_embed)

                changed = (
                    resolved_chat
                    and resolved_embed
                    and (
                        resolved_chat != (settings.openai_chat_model or "").strip()
                        or resolved_embed != (settings.openai_embedding_model or "").strip()
                    )
                )
                if changed:
                    if st.button("Applica modelli OpenAI"):
                        settings.openai_chat_model = resolved_chat
                        settings.openai_embedding_model = resolved_embed
                        self._service.save_user_llm_settings(settings)
                        self._service._agent = None
                        st.success("✅ Modelli OpenAI aggiornati!")
                        st.rerun()

            elif provider == "anthropic":
                auth_labels = list(self._ANTHROPIC_AUTH_METHODS.keys())
                current_auth_label = self._anthropic_auth_method_label(settings.anthropic_auth_method)
                selected_auth_label = st.radio(
                    "Metodo autenticazione Anthropic",
                    options=auth_labels,
                    index=auth_labels.index(current_auth_label),
                    help="Puoi usare il login Claude OAuth oppure una API key Anthropic.",
                )
                selected_auth_method = self._ANTHROPIC_AUTH_METHODS[selected_auth_label]
                if selected_auth_method != settings.anthropic_auth_method:
                    settings.anthropic_auth_method = selected_auth_method
                    self._service.save_user_llm_settings(settings)
                    self._service._agent = None
                    st.rerun()

                if settings.anthropic_auth_method == "api_key":
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        anthropic_api_key = st.text_input(
                            "Anthropic API Key",
                            value=settings.anthropic_api_key,
                            type="password",
                            key="anthropic_api_key_input",
                            help="Inserisci la tua chiave API Anthropic (sk-ant-...).",
                            placeholder="sk-ant-...",
                        )
                        st.link_button(
                            "Apri Console Anthropic",
                            self._ANTHROPIC_CONSOLE_KEYS_URL,
                            use_container_width=True,
                        )
                    with col2:
                        st.write("")
                        st.write("")
                        if st.button("Salva Anthropic", use_container_width=True):
                            settings.anthropic_api_key = anthropic_api_key.strip()
                            settings.anthropic_auth_method = "api_key"
                            self._service.save_user_llm_settings(settings)
                            self._service._agent = None
                            st.success("✅ Impostazioni Anthropic salvate!")
                            st.rerun()
                else:
                    from streamlit_app.llm.anthropic_oauth import (
                        MANUAL_REDIRECT_URI,
                        build_authorize_url,
                        exchange_code_for_tokens,
                        generate_pkce,
                        generate_state,
                        parse_pasted_code,
                        start_loopback_server,
                    )

                    if st.button("Avvia login Claude", use_container_width=True):
                        try:
                            existing_flow = st.session_state.get("anthropic_oauth_flow") or {}
                            existing_server = existing_flow.get("server")
                            if existing_server:
                                existing_server.close()
                            pkce = generate_pkce()
                            state = generate_state()
                            loopback = start_loopback_server(state)
                            st.session_state.anthropic_oauth_flow = {
                                "mode": "loopback",
                                "state": state,
                                "verifier": pkce.verifier,
                                "redirect_uri": loopback.redirect_uri,
                                "server": loopback,
                                "authorize_url": build_authorize_url(
                                    redirect_uri=loopback.redirect_uri,
                                    code_challenge=pkce.challenge,
                                    state=state,
                                ),
                            }
                            st.rerun()
                        except Exception as exc:
                            st.warning(
                                "Loopback OAuth non disponibile. Uso il fallback copy/paste."
                            )
                            pkce = generate_pkce()
                            state = generate_state()
                            st.session_state.anthropic_oauth_flow = {
                                "mode": "manual",
                                "state": state,
                                "verifier": pkce.verifier,
                                "redirect_uri": MANUAL_REDIRECT_URI,
                                "authorize_url": build_authorize_url(
                                    redirect_uri=MANUAL_REDIRECT_URI,
                                    code_challenge=pkce.challenge,
                                    state=state,
                                ),
                                "error": str(exc),
                            }
                            st.rerun()

                    oauth_flow = st.session_state.get("anthropic_oauth_flow") or {}
                    if oauth_flow:
                        if oauth_flow.get("error"):
                            st.caption(f"Fallback attivo: {oauth_flow['error']}")
                        st.link_button(
                            "Login Claude",
                            str(oauth_flow.get("authorize_url") or ""),
                            use_container_width=True,
                        )
                        if oauth_flow.get("mode") == "loopback":
                            st.info(
                                "Dopo il login Claude, torna qui e premi “Controlla callback” "
                                "se la pagina non si aggiorna automaticamente."
                            )
                            server = oauth_flow.get("server")
                            callback = server.poll() if server else None
                            if st.button("Controlla callback Claude", use_container_width=True):
                                callback = server.poll() if server else callback
                            if callback:
                                try:
                                    token_payload = exchange_code_for_tokens(
                                        code=str(callback.get("code") or ""),
                                        state=str(callback.get("state") or ""),
                                        verifier=str(oauth_flow.get("verifier") or ""),
                                        redirect_uri=str(oauth_flow.get("redirect_uri") or ""),
                                    )
                                    refresh_token = str(token_payload.get("refresh_token") or "").strip()
                                    if not refresh_token:
                                        raise ValueError("il refresh token non è presente nella risposta OAuth.")
                                    settings.anthropic_oauth_refresh_token = refresh_token
                                    settings.anthropic_auth_method = "oauth"
                                    self._service.save_user_llm_settings(settings)
                                    self._service._agent = None
                                    if server:
                                        server.close()
                                    del st.session_state.anthropic_oauth_flow
                                    st.success("✅ Login Claude salvato!")
                                    st.rerun()
                                except Exception as exc:
                                    st.error(f"Impossibile completare il login Claude: {exc}")

                        with st.expander("Fallback copy/paste", expanded=oauth_flow.get("mode") == "manual"):
                            st.caption(
                                "Se il loopback non funziona, usa questo link e incolla qui il codice "
                                "mostrato dal browser (CODE oppure CODE#STATE)."
                            )
                            if st.button("Usa fallback copy/paste", use_container_width=True):
                                pkce = generate_pkce()
                                state = generate_state()
                                st.session_state.anthropic_oauth_flow = {
                                    "mode": "manual",
                                    "state": state,
                                    "verifier": pkce.verifier,
                                    "redirect_uri": MANUAL_REDIRECT_URI,
                                    "authorize_url": build_authorize_url(
                                        redirect_uri=MANUAL_REDIRECT_URI,
                                        code_challenge=pkce.challenge,
                                        state=state,
                                    ),
                                }
                                st.rerun()
                            pasted_code = st.text_input(
                                "Codice autorizzazione Claude",
                                value="",
                                key="anthropic_oauth_code_input",
                                placeholder="CODE oppure CODE#STATE",
                            )
                            if st.button("Completa pairing Claude", use_container_width=True):
                                try:
                                    parsed = parse_pasted_code(
                                        pasted_code,
                                        default_state=str(oauth_flow.get("state") or ""),
                                    )
                                    token_payload = exchange_code_for_tokens(
                                        code=parsed["code"],
                                        state=parsed.get("state") or str(oauth_flow.get("state") or ""),
                                        verifier=str(oauth_flow.get("verifier") or ""),
                                        redirect_uri=str(oauth_flow.get("redirect_uri") or MANUAL_REDIRECT_URI),
                                    )
                                    refresh_token = str(token_payload.get("refresh_token") or "").strip()
                                    if not refresh_token:
                                        raise ValueError("il refresh token non è presente nella risposta OAuth.")
                                    settings.anthropic_oauth_refresh_token = refresh_token
                                    settings.anthropic_auth_method = "oauth"
                                    self._service.save_user_llm_settings(settings)
                                    self._service._agent = None
                                    server = oauth_flow.get("server")
                                    if server:
                                        server.close()
                                    del st.session_state.anthropic_oauth_flow
                                    st.success("✅ Login Claude salvato!")
                                    st.rerun()
                                except Exception as exc:
                                    st.error(f"Impossibile completare il pairing Claude: {exc}")

                    if settings.anthropic_oauth_refresh_token:
                        st.success("Login Claude OAuth configurato.")
                        st.caption("La chat userà il backend Claude associato al tuo account.")
                        st.session_state.llm_block_reason = None
                    else:
                        st.session_state.llm_block_reason = (
                            "Completa il login Claude OAuth oppure usa una API key Anthropic."
                        )

                is_anthropic_oauth = settings.anthropic_auth_method == "oauth"
                anthropic_options = self._build_anthropic_chat_model_options(
                    settings.anthropic_chat_model,
                    for_oauth=is_anthropic_oauth,
                )
                anthropic_selected = st.selectbox(
                    "Modello Claude (chat)",
                    options=anthropic_options,
                    index=(
                        anthropic_options.index(settings.anthropic_chat_model)
                        if settings.anthropic_chat_model in anthropic_options
                        else 0
                    ),
                    help=(
                        "Con login Claude OAuth sono mostrati solo i modelli compatibili. "
                        "Con API key puoi inserire un modello custom."
                    ),
                )
                custom_anthropic_model = ""
                if anthropic_selected == "Altro…":
                    custom_anthropic_model = st.text_input(
                        "Nome modello Claude (custom)",
                        value=settings.anthropic_chat_model,
                        help="Inserisci il nome esatto del modello Claude.",
                    )
                resolved_anthropic_model = self._resolve_model_selection(
                    anthropic_selected,
                    custom_anthropic_model,
                )
                if resolved_anthropic_model and resolved_anthropic_model != (
                    settings.anthropic_chat_model or ""
                ).strip():
                    if st.button("Applica modello Claude"):
                        settings.anthropic_chat_model = resolved_anthropic_model
                        self._service.save_user_llm_settings(settings)
                        self._service._agent = None
                        st.success("✅ Modello Claude aggiornato!")
                        st.rerun()

            else:
                st.info("Stai usando **Ollama in locale**. La chiave OpenAI non è necessaria.")

                base_url = st.text_input(
                    "Ollama base URL",
                    value=settings.ollama_base_url,
                    help="Suggerito: macOS/Windows Docker Desktop → http://host.docker.internal:11434; Linux → http://172.17.0.1:11434; fuori da Docker → http://localhost:11434",
                )

                # Model discovery
                if "ollama_models_cache" not in st.session_state:
                    st.session_state.ollama_models_cache = []

                cols = st.columns([1, 1, 2])
                with cols[0]:
                    refresh = st.button("↻ Aggiorna modelli", use_container_width=True)
                with cols[1]:
                    use_cached = st.checkbox("Usa cache", value=True)
                if refresh or (not use_cached) or not st.session_state.ollama_models_cache:
                    client = OllamaClient(base_url=base_url.strip() or settings.ollama_base_url)
                    st.session_state.ollama_models_cache = client.list_model_names()

                model_names: List[str] = st.session_state.ollama_models_cache
                if not model_names:
                    st.error("Non riesco a leggere **nessun modello** da Ollama (o Ollama non è raggiungibile).")
                    st.markdown("Se non hai ancora scaricato nulla, esegui sul tuo host:")
                    st.code("ollama pull nomic-embed-text\nollama pull gpt-oss:20b", language="bash")
                    st.session_state.llm_block_reason = (
                        "Ollama non ha modelli disponibili (o non è raggiungibile). "
                        "Scarica almeno un modello chat e un modello embeddings, poi premi “Aggiorna modelli”."
                    )

                chat_model = st.selectbox(
                    "Modello Ollama (chat)",
                    options=model_names if model_names else [settings.ollama_chat_model],
                    index=(model_names.index(settings.ollama_chat_model) if model_names and settings.ollama_chat_model in model_names else 0),
                )

                # For embeddings, prefer embedding-capable models (e.g., nomic-embed-text)
                embedding_candidates = [
                    n for n in model_names
                    if any(k in n.lower() for k in ("embed", "bge", "mxbai"))
                ]
                if model_names and not embedding_candidates:
                    st.warning("Non vedo embedding model tra quelli installati in Ollama.")
                    st.markdown("Suggerimento (host):")
                    st.code("ollama pull nomic-embed-text", language="bash")
                embedding_options = embedding_candidates or (model_names if model_names else [settings.ollama_embedding_model])
                embed_model = st.selectbox(
                    "Modello Ollama (embeddings)",
                    options=embedding_options,
                    index=(
                        embedding_options.index(settings.ollama_embedding_model)
                        if settings.ollama_embedding_model in embedding_options
                        else 0
                    ),
                )
                if embed_model and not any(k in embed_model.lower() for k in ("embed", "bge", "mxbai")):
                    st.warning("Il modello embeddings selezionato non sembra un embedding model. Consigliato: **nomic-embed-text** (o simili).")

                if st.button("Salva impostazioni Ollama"):
                    if not model_names:
                        st.error("Impossibile salvare: Ollama non ha modelli disponibili (o non è raggiungibile).")
                        st.stop()
                    settings.ollama_base_url = base_url.strip() or settings.ollama_base_url
                    settings.ollama_chat_model = chat_model
                    settings.ollama_embedding_model = embed_model
                    self._service.save_user_llm_settings(settings)
                    self._service._agent = None
                    st.success("✅ Impostazioni Ollama salvate!")
                    st.rerun()

            with st.expander("Claude Code setup-token (legacy)", expanded=False):
                st.caption(
                    "Il setup-token Claude Code e' mantenuto solo per retrocompatibilita'. "
                    "Per usare Claude nella chat seleziona provider Anthropic e completa il login OAuth "
                    "o inserisci una API key."
                )
                col_doc, col_console = st.columns(2)
                with col_doc:
                    st.link_button(
                        "Guida Claude Code",
                        self._CLAUDE_CODE_SETUP_URL,
                        use_container_width=True,
                    )
                with col_console:
                    st.link_button(
                        "Console Anthropic",
                        self._ANTHROPIC_CONSOLE_KEYS_URL,
                        use_container_width=True,
                    )
                if settings.anthropic_setup_token:
                    st.info("Setup-token legacy presente nel database, ma non viene usato dal runtime.")

            st.divider()
            st.header("📊 Gestione Dataset")
            
            # Dataset source selection
            data_source = st.radio(
                "Fonte Dati",
                [
                    "🇦🇹 Dataset Vienna (229K alberi)", 
                    "🇮🇹 Dataset Milano (251K alberi)",
                    "📁 Carica CSV Personalizzato"
                ],
                key="data_source_radio"
            )
            
            # Handle Vienna dataset
            if data_source == "🇦🇹 Dataset Vienna (229K alberi)":
                # Clear custom dataset state
                # Note: Don't delete "data_description_input" as it's controlled by the widget
                # The widget will reset automatically when stored_data_description is cleared
                for key in ["custom_db_path", "custom_table_name", "stored_data_description",
                           "uploaded_file_name", "uploaded_file_signature", "dataset_metadata", "selected_preset"]:
                    if key in st.session_state:
                        del st.session_state[key]
                
                # Set Vienna as selected preset
                if st.session_state.get("selected_preset") != "vienna":
                    st.session_state.selected_preset = "vienna"
                    self._service._agent = None
                
                st.info("""🌳 **Dataset Vienna Trees (BAUMKATOGD)**
                
- **Alberi totali:** 229.298
- **Distretti:** 23
- **Colonne principali:** specie, anno piantumazione, circonferenza, altezza, via
- **Periodo:** Dati storici fino ad oggi
                """)
            
            # Handle Milano dataset
            elif data_source == "🇮🇹 Dataset Milano (251K alberi)":
                # Clear custom dataset state
                # Note: Don't delete "data_description_input" as it's controlled by the widget
                # The widget will reset automatically when stored_data_description is cleared
                for key in ["custom_db_path", "custom_table_name", "stored_data_description",
                           "uploaded_file_name", "uploaded_file_signature", "dataset_metadata"]:
                    if key in st.session_state:
                        del st.session_state[key]
                
                # Set Milano as selected preset
                if st.session_state.get("selected_preset") != "milano":
                    st.session_state.selected_preset = "milano"
                    self._service._agent = None
                
                st.info("""🌳 **Dataset Milano Trees**
                
- **Alberi totali:** 251.165
- **Municipi:** 9
- **Specie uniche:** 338
- **Colonne principali:** genere, specie, varietà, diametro tronco, altezza, via, coordinate GPS
- **Periodo:** Dal 1984 ad oggi
                """)
            
            elif data_source == "📁 Carica CSV Personalizzato":
                st.info("📁 Carica un file CSV per analizzarlo con l'AI")
                
                uploaded_file = st.file_uploader(
                    "Seleziona file CSV",
                    type=["csv"],
                    key="csv_uploader",
                    help="Il file verrà automaticamente convertito in database SQL"
                )
                
                # Optional: description of the data
                # #region agent log
                import json
                log_path = "/Users/francesco/Sviluppo/frasma_studio/tree-evaluator/.cursor/debug.log"
                try:
                    with open(log_path, "a") as f:
                        f.write(json.dumps({"sessionId": "debug-session", "runId": "post-fix-v2", "hypothesisId": "A", "location": "ui.py:357", "message": "BEFORE text_area widget creation", "data": {"stored_data_description_in_state": "stored_data_description" in st.session_state, "data_description_input_in_state": "data_description_input" in st.session_state}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
                except: pass
                # #endregion
                
                # Use a different key for the widget to avoid conflicts
                # Read stored value directly without initializing the widget key
                # This avoids the "cannot be modified after widget instantiation" error
                stored_value = st.session_state.get("stored_data_description", "")
                
                description = st.text_area(
                    "Descrizione dati (opzionale)",
                    value=stored_value,
                    placeholder="Es: Questo dataset contiene vendite mensili per regione dal 2020 al 2024...",
                    key="data_description_input",
                    help="Fornisci un contesto che aiuti l'AI a comprendere meglio i tuoi dati",
                    height=100
                )
                
                # #region agent log
                try:
                    with open(log_path, "a") as f:
                        f.write(json.dumps({"sessionId": "debug-session", "runId": "post-fix", "hypothesisId": "A", "location": "ui.py:380", "message": "AFTER text_area widget creation", "data": {"description_value": description, "description_type": type(description).__name__}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
                except: pass
                # #endregion
                
                if uploaded_file:
                    # Only process if file has changed or not yet processed
                    current_file_name = st.session_state.get("uploaded_file_name", None)
                    file_bytes = uploaded_file.getbuffer()
                    uploaded_file_signature = hashlib.sha256(
                        file_bytes.tobytes() if hasattr(file_bytes, "tobytes") else bytes(file_bytes)
                    ).hexdigest()
                    current_file_signature = st.session_state.get("uploaded_file_signature", None)
                    
                    # #region agent log
                    try:
                        with open(log_path, "a") as f:
                            f.write(json.dumps({"sessionId": "debug-session", "runId": "post-fix", "hypothesisId": "B", "location": "ui.py:388", "message": "INSIDE uploaded_file block", "data": {"uploaded_file_name": uploaded_file.name, "current_file_name": current_file_name, "description_value": description}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
                    except: pass
                    # #endregion
                    
                    if current_file_name != uploaded_file.name or current_file_signature != uploaded_file_signature:
                        with st.spinner("📥 Caricamento e conversione CSV in corso..."):
                            try:
                                from pathlib import Path
                                from streamlit_app.services.data_manager import DynamicDataManager
                                
                                # Initialize manager
                                manager = DynamicDataManager(Path("temp_data"))
                                
                                # Process uploaded file
                                db_path, table_name, metadata = manager.process_uploaded_file(uploaded_file)
                                
                                # #region agent log
                                try:
                                    with open(log_path, "a") as f:
                                        f.write(json.dumps({"sessionId": "debug-session", "runId": "post-fix", "hypothesisId": "A", "location": "ui.py:405", "message": "BEFORE stored_data_description assignment", "data": {"description_value": description}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
                                except: pass
                                # #endregion
                                
                                # Update session state
                                # Use stored_data_description instead of data_description to avoid widget conflict
                                st.session_state.custom_db_path = str(db_path)
                                st.session_state.custom_table_name = table_name
                                st.session_state.stored_data_description = description
                                st.session_state.uploaded_file_name = uploaded_file.name
                                st.session_state.uploaded_file_signature = metadata.get("file_hash", uploaded_file_signature)
                                st.session_state.dataset_metadata = metadata
                                
                                # #region agent log
                                try:
                                    with open(log_path, "a") as f:
                                        f.write(json.dumps({"sessionId": "debug-session", "runId": "post-fix", "hypothesisId": "A", "location": "ui.py:415", "message": "AFTER stored_data_description assignment", "data": {"stored_data_description_in_state": "stored_data_description" in st.session_state, "assignment_successful": True}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
                                except: pass
                                # #endregion
                                
                                # Force agent re-initialization
                                self._service._agent = None
                                
                                st.success(f"✅ Dataset caricato con successo!")
                                
                                # Show metadata
                                with st.expander("📋 Info Dataset"):
                                    st.write(f"**File:** {metadata['original_filename']}")
                                    st.write(f"**Righe:** {metadata['row_count']:,}")
                                    st.write(f"**Colonne:** {metadata['column_count']}")
                                    st.write(f"**Separatore rilevato:** `{metadata.get('detected_delimiter', ',')}`")
                                    st.write(f"**Encoding:** `{metadata.get('detected_encoding', 'n/d')}`")
                                    st.write(f"**Tabella SQL:** `{table_name}`")
                                    if metadata.get("warnings"):
                                        st.warning("\n".join(metadata["warnings"]))
                                    if metadata.get("profile_summary"):
                                        st.text_area(
                                            "Profilo automatico",
                                            value=metadata["profile_summary"],
                                            height=180,
                                            disabled=True,
                                        )
                                    st.write("\n**Colonne:**")
                                    for orig, sql in metadata['column_mapping'].items():
                                        st.write(f"- {orig} → `{sql}`")
                                
                            except Exception as e:
                                # #region agent log
                                try:
                                    with open(log_path, "a") as f:
                                        f.write(json.dumps({"sessionId": "debug-session", "runId": "post-fix", "hypothesisId": "A", "location": "ui.py:425", "message": "EXCEPTION caught", "data": {"error_type": type(e).__name__, "error_message": str(e)}, "timestamp": int(__import__("time").time() * 1000)}) + "\n")
                                except: pass
                                # #endregion
                                
                                st.error(f"❌ Errore nel caricamento: {str(e)}")
                                # Clear any partial state
                                if "custom_db_path" in st.session_state:
                                    del st.session_state.custom_db_path
                    else:
                        # File already loaded, just show info
                        st.success(f"✅ Dataset attivo: {uploaded_file.name}")
                        if "dataset_metadata" in st.session_state:
                            metadata = st.session_state.dataset_metadata
                            with st.expander("📋 Info Dataset"):
                                st.write(f"**File:** {metadata['original_filename']}")
                                st.write(f"**Righe:** {metadata['row_count']:,}")
                                st.write(f"**Colonne:** {metadata['column_count']}")
                                st.write(f"**Separatore rilevato:** `{metadata.get('detected_delimiter', ',')}`")
                                st.write(f"**Encoding:** `{metadata.get('detected_encoding', 'n/d')}`")
                                if metadata.get("warnings"):
                                    st.warning("\n".join(metadata["warnings"]))
                                if metadata.get("profile_summary"):
                                    st.text_area(
                                        "Profilo automatico",
                                        value=metadata["profile_summary"],
                                        height=180,
                                        disabled=True,
                                    )
                
                # Button to reset to default dataset
                if st.button("🔄 Torna al Dataset Vienna", use_container_width=True):
                    # Clear custom dataset state
                    # Note: Don't delete "data_description_input" as it's controlled by the widget
                    # The widget will reset automatically when stored_data_description is cleared
                    for key in ["custom_db_path", "custom_table_name", "stored_data_description",
                               "uploaded_file_name", "uploaded_file_signature", "dataset_metadata", "selected_preset"]:
                        if key in st.session_state:
                            del st.session_state[key]
                    # Force agent re-initialization
                    self._service._agent = None
                    st.rerun()

            st.divider()
            st.header("💬 Conversazioni")
            
            # New conversation button
            if st.button("➕ Nuova Chat", use_container_width=True, type="primary"):
                self._create_new_conversation()
                st.rerun()

            # Load conversations if not loaded
            if not st.session_state.conversations:
                self._load_conversations()

            # Display conversation list
            if st.session_state.conversations:
                for conv in st.session_state.conversations:
                    # Check if this conversation is being edited
                    is_editing = st.session_state.editing_conversation_id == conv.id
                    
                    if is_editing:
                        # Show rename input
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            new_title = st.text_input(
                                "Rinomina",
                                value=conv.title,
                                key=f"rename_input_{conv.id}",
                                label_visibility="collapsed"
                            )
                        with col2:
                            if st.button("✓", key=f"save_{conv.id}", help="Salva"):
                                if new_title.strip():
                                    self._service.rename_conversation(conv.id, new_title.strip())
                                    conv.title = new_title.strip()
                                st.session_state.editing_conversation_id = None
                                st.rerun()
                        with col3:
                            if st.button("✗", key=f"cancel_{conv.id}", help="Annulla"):
                                st.session_state.editing_conversation_id = None
                                st.rerun()
                    else:
                        # Normal view
                        is_current = conv.id == st.session_state.current_conversation_id
                        
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            button_kwargs = {
                                "label": f"{'▶️ ' if is_current else ''}{conv.title}",
                                "key": f"conv_{conv.id}",
                                "use_container_width": True,
                            }
                            if is_current:
                                button_kwargs["type"] = "secondary"
                            
                            if st.button(**button_kwargs):
                                self._load_conversation_messages(conv.id)
                                st.rerun()
                        
                        with col2:
                            if st.button("✏️", key=f"edit_{conv.id}", help="Rinomina conversazione"):
                                st.session_state.editing_conversation_id = conv.id
                                st.rerun()
                        
                        with col3:
                            if st.button("🗑️", key=f"del_{conv.id}", help="Elimina conversazione"):
                                self._service.delete_conversation(conv.id)
                                st.session_state.conversations.remove(conv)
                                if conv.id == st.session_state.current_conversation_id:
                                    st.session_state.current_conversation_id = None
                                    st.session_state.messages = []
                                st.rerun()
            else:
                st.info("Nessuna conversazione. Crea la tua prima chat!")

    def _extract_chart_from_response(self, content: str) -> tuple[str, Optional[dict]]:
        """Extract chart JSON from assistant response if present.
        
        Returns:
            Tuple of (text_content, chart_data_dict or None)
        """
        # Look for chart data between markers
        if "CHART_DATA_START" in content and "CHART_DATA_END" in content:
            try:
                start_marker = "CHART_DATA_START"
                end_marker = "CHART_DATA_END"
                
                start_idx = content.find(start_marker) + len(start_marker)
                end_idx = content.find(end_marker)
                
                if start_idx > len(start_marker) and end_idx > start_idx:
                    json_str = content[start_idx:end_idx].strip()
                    chart_data = json.loads(json_str)
                    
                    if chart_data.get("success") and "chart_json" in chart_data:
                        # Remove chart data section from text
                        text_before = content[:content.find(start_marker)].strip()
                        text_after = content[content.find(end_marker) + len(end_marker):].strip()
                        text_content = (text_before + " " + text_after).strip()
                        return text_content, chart_data
            except (json.JSONDecodeError, ValueError):
                pass
        
        # Fallback: try old method (for backward compatibility)
        if "chart_json" in content.lower() or '"success": true' in content:
            try:
                # Try to extract JSON object from text
                start_idx = content.find('{')
                end_idx = content.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    json_str = content[start_idx:end_idx+1]
                    chart_data = json.loads(json_str)
                    if chart_data.get("success") and "chart_json" in chart_data:
                        # Remove JSON from text
                        text_before = content[:start_idx].strip()
                        text_after = content[end_idx+1:].strip()
                        text_content = (text_before + " " + text_after).strip()
                        return text_content, chart_data
            except (json.JSONDecodeError, ValueError):
                pass
        
        return content, None

    def _extract_map_from_response(self, content: str) -> Tuple[str, Optional[dict]]:
        """Extract map JSON from assistant response if present.
        
        Returns:
            Tuple of (text_content, map_data_dict or None)
        """
        # Look for map data between markers
        if "MAP_DATA_START" in content and "MAP_DATA_END" in content:
            try:
                start_marker = "MAP_DATA_START"
                end_marker = "MAP_DATA_END"
                
                start_idx = content.find(start_marker) + len(start_marker)
                end_idx = content.find(end_marker)
                
                if start_idx > len(start_marker) and end_idx > start_idx:
                    json_str = content[start_idx:end_idx].strip()
                    map_data = json.loads(json_str)
                    
                    if map_data.get("success") and "map_html" in map_data:
                        # Remove map data section from text
                        text_before = content[:content.find(start_marker)].strip()
                        text_after = content[content.find(end_marker) + len(end_marker):].strip()
                        text_content = (text_before + " " + text_after).strip()
                        return text_content, map_data
            except (json.JSONDecodeError, ValueError):
                pass
        
        # Fallback: try old method (for backward compatibility)
        if "map_html" in content.lower() or ('"success": true' in content and "map_type" in content):
            try:
                # Try to extract JSON object from text
                start_idx = content.find('{')
                end_idx = content.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    json_str = content[start_idx:end_idx+1]
                    map_data = json.loads(json_str)
                    if map_data.get("success") and "map_html" in map_data:
                        # Remove JSON from text
                        text_before = content[:start_idx].strip()
                        text_after = content[end_idx+1:].strip()
                        text_content = (text_before + " " + text_after).strip()
                        return text_content, map_data
            except (json.JSONDecodeError, ValueError):
                pass
        
        return content, None

    def _render_map(self, map_data: dict, placeholder=None) -> None:
        """Render a Folium map from map data."""
        container = placeholder if placeholder else st
        
        try:
            map_html = map_data["map_html"]
            
            # Render the map using Streamlit components
            container.markdown(f"### 🗺️ {map_data.get('title', 'Mappa')}")
            components.html(map_html, height=500, scrolling=True)
            
            # Show map info in expander
            with container.expander("ℹ️ Dettagli mappa"):
                container.write(f"**Tipo:** {map_data.get('map_type', 'N/A')}")
                container.write(f"**Punti visualizzati:** {map_data.get('data_points', 'N/A'):,}")
                if "sql_executed" in map_data:
                    container.code(map_data["sql_executed"], language="sql")
                if "center" in map_data:
                    center = map_data["center"]
                    container.write(f"**Centro:** {center.get('lat', 'N/A')}, {center.get('lon', 'N/A')}")
                container.write(f"**Zoom:** {map_data.get('zoom', 'N/A')}")
        except Exception as e:
            container.error(f"Errore nella visualizzazione della mappa: {e}")

    def _render_plotly_figure(self, fig: go.Figure, height: int = 520) -> None:
        """Render a Plotly figure in a robust way inside Streamlit chat.

        Note: `st.plotly_chart` can intermittently fail to render in some chat layouts.
        Using `components.html(fig.to_html(...))` is more reliable.
        """
        try:
            html = fig.to_html(include_plotlyjs="inline", full_html=False)
            components.html(html, height=height, scrolling=False)
        except Exception as e:
            st.error(f"Errore nella visualizzazione del grafico (Plotly HTML): {e}")
    
    def _render_messages(self) -> None:
        """Render all messages in the current conversation."""
        for message in st.session_state.messages:
            with st.chat_message(message.role):
                # Check if message contains chart or map data
                if message.role == "assistant":
                    # Show saved reasoning if present (in collapsed expander)
                    if hasattr(message, 'reasoning') and message.reasoning:
                        with st.expander("🧠 Processo di ragionamento", expanded=False):
                            # Split reasoning into steps and format them
                            reasoning_steps = message.reasoning.split("\n")
                            for step in reasoning_steps:
                                if step.strip():
                                    st.markdown(step)
                    
                    # First check for chart data
                    text_content, chart_data = self._extract_chart_from_response(message.content)
                    
                    # Then check for map data in the remaining text
                    text_content, map_data = self._extract_map_from_response(text_content)
                    
                    has_visualization = False
                    
                    # Display text content first
                    if text_content:
                        st.markdown(text_content)
                    
                    # Display chart if present
                    if chart_data and chart_data.get("success"):
                        has_visualization = True
                        try:
                            chart_json = chart_data["chart_json"]
                            fig = go.Figure(json.loads(chart_json))
                            self._render_plotly_figure(fig)
                            
                            # Show chart info
                            with st.expander("ℹ️ Dettagli grafico"):
                                st.write(f"**Tipo:** {chart_data.get('chart_type', 'N/A')}")
                                st.write(f"**Punti dati:** {chart_data.get('data_points', 'N/A')}")
                                if "sql_executed" in chart_data:
                                    st.code(chart_data["sql_executed"], language="sql")
                        except Exception as e:
                            st.error(f"Errore nella visualizzazione del grafico: {e}")
                    
                    # Display map if present
                    if map_data and map_data.get("success"):
                        has_visualization = True
                        self._render_map(map_data)
                    
                    # If no visualization was found and no text was displayed, show original content
                    if not has_visualization and not text_content:
                        st.markdown(message.content)
                else:
                    st.markdown(message.content)

    def _inject_custom_css(self) -> None:
        """Inject custom CSS for reasoning section styling."""
        st.markdown("""
        <style>
        /* Reasoning section styling - visually distinct from chat */
        .reasoning-container {
            background: linear-gradient(135deg, rgba(45, 55, 72, 0.95) 0%, rgba(26, 32, 44, 0.95) 100%);
            border: 1px solid rgba(99, 179, 237, 0.3);
            border-left: 4px solid #63b3ed;
            border-radius: 8px;
            padding: 16px 20px;
            margin: 12px 0;
            font-size: 0.9em;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        
        .reasoning-header {
            display: flex;
            align-items: center;
            gap: 8px;
            color: #90cdf4;
            font-weight: 600;
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid rgba(99, 179, 237, 0.2);
        }
        
        .reasoning-content {
            color: #e2e8f0;
            line-height: 1.6;
        }
        
        .reasoning-step {
            padding: 8px 12px;
            margin: 6px 0;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 6px;
            border-left: 2px solid rgba(99, 179, 237, 0.5);
        }
        
        .reasoning-step-emoji {
            margin-right: 8px;
        }
        
        /* Loading animation for reasoning */
        .reasoning-loading {
            display: inline-block;
            animation: pulse 1.5s ease-in-out infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 0.6; }
            50% { opacity: 1; }
        }
        
        /* Expander override for reasoning */
        .stExpander[data-testid="stExpander"] > div:first-child {
            background: linear-gradient(135deg, rgba(45, 55, 72, 0.8) 0%, rgba(26, 32, 44, 0.8) 100%);
            border: 1px solid rgba(99, 179, 237, 0.3);
            border-radius: 8px;
        }
        
        /* Better contrast for reasoning expander header */
        .stExpander[data-testid="stExpander"] summary {
            color: #90cdf4 !important;
        }
        
        /* Tool result styling */
        .tool-result {
            background: rgba(72, 187, 120, 0.1);
            border-left: 3px solid #48bb78;
            padding: 8px 12px;
            margin: 8px 0;
            border-radius: 0 6px 6px 0;
            font-size: 0.88em;
        }
        
        /* Validation styling */
        .validation-result {
            background: rgba(159, 122, 234, 0.1);
            border-left: 3px solid #9f7aea;
            padding: 8px 12px;
            margin: 8px 0;
            border-radius: 0 6px 6px 0;
            font-size: 0.88em;
        }
        </style>
        """, unsafe_allow_html=True)

    def _format_reasoning_step(self, step: str) -> str:
        """Format a reasoning step with appropriate emoji and styling."""
        # Determine step type and assign emoji
        step_lower = step.lower()
        
        if "✅" in step or "risultati" in step_lower:
            emoji = "✅"
            css_class = "tool-result"
        elif "✓" in step or "validazione" in step_lower:
            emoji = "✓"
            css_class = "validation-result"
        elif "query sql" in step_lower or "select" in step_lower:
            emoji = "🔍"
            css_class = "reasoning-step"
        else:
            emoji = "💭"
            css_class = "reasoning-step"
        
        return f'<div class="{css_class}"><span class="reasoning-step-emoji">{emoji}</span>{step}</div>'

    def _render_reasoning_box(self, steps: list, is_loading: bool = False) -> str:
        """Render reasoning steps in a styled container."""
        header_text = "🧠 Processo di ragionamento"
        if is_loading:
            header_text += ' <span class="reasoning-loading">●●●</span>'
        
        formatted_steps = "".join([self._format_reasoning_step(s) for s in steps])
        
        return f"""
        <div class="reasoning-container">
            <div class="reasoning-header">{header_text}</div>
            <div class="reasoning-content">{formatted_steps}</div>
        </div>
        """

    def render(self) -> None:
        """Main render method for the chat UI."""
        self._ensure_session()
        st.set_page_config(page_title="Tree Evaluator Chat", page_icon="🌳", layout="centered")
        self._inject_custom_css()
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
                            reasoning_html = self._render_reasoning_box(reasoning_steps, is_loading=True)
                            reasoning_placeholder.markdown(reasoning_html, unsafe_allow_html=True)
                        
                        elif chunk_type == "response":
                            # Update final response
                            full_response = chunk_content
                            # Show reasoning in collapsed expander with better styling
                            if reasoning_steps:
                                with reasoning_placeholder:
                                    with st.expander("🧠 Processo di ragionamento", expanded=False):
                                        # Use styled HTML for each step
                                        steps_html = "".join([self._format_reasoning_step(s) for s in reasoning_steps])
                                        st.markdown(f'<div class="reasoning-content">{steps_html}</div>', unsafe_allow_html=True)
                            
                            # Check if response contains chart data
                            text_content, extracted_chart = self._extract_chart_from_response(full_response)
                            
                            # Check if response contains map data
                            text_content, extracted_map = self._extract_map_from_response(text_content)
                            
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
                        text_content, extracted_chart = self._extract_chart_from_response(full_response)
                        text_content, extracted_map = self._extract_map_from_response(text_content)
                        
                        has_visualization = False
                        
                        # Display text content
                        if extracted_chart and extracted_chart.get("success"):
                            has_visualization = True
                            if text_content:
                                response_placeholder.markdown(text_content)
                            
                            # Display chart
                            with chart_placeholder:
                                try:
                                    chart_json = extracted_chart["chart_json"]
                                    fig = go.Figure(json.loads(chart_json))
                                    self._render_plotly_figure(fig)
                                    
                                    # Show chart info
                                    with st.expander("ℹ️ Dettagli grafico"):
                                        st.write(f"**Tipo:** {extracted_chart.get('chart_type', 'N/A')}")
                                        st.write(f"**Punti dati:** {extracted_chart.get('data_points', 'N/A')}")
                                        if "sql_executed" in extracted_chart:
                                            st.code(extracted_chart["sql_executed"], language="sql")
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
                                self._render_map(extracted_map)
                        
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
