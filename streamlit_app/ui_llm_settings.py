from __future__ import annotations

import streamlit as st

from streamlit_app.ui_anthropic_settings import render_anthropic_settings
from streamlit_app.ui_ollama_settings import render_ollama_settings
from streamlit_app.ui_openai_settings import render_openai_settings


def render_llm_settings(ui, settings) -> None:
    """Render language, provider, auth, and model controls."""
    st.header("⚙️ Settings")
    st.session_state.llm_block_reason = None

    _render_language_selector(ui, settings)
    provider = _render_provider_selector(ui, settings)

    if provider == "openai":
        render_openai_settings(ui, settings)
    elif provider == "anthropic":
        render_anthropic_settings(ui, settings)
    else:
        render_ollama_settings(ui, settings)

    _render_legacy_claude_setup(ui, settings)


def _render_language_selector(ui, settings) -> None:
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
        ui._service.save_user_llm_settings(settings)
        ui._service._agent = None
        st.rerun()


def _render_provider_selector(ui, settings) -> str:
    providers = ["openai", "anthropic", "ollama"]
    provider = st.selectbox(
        "Provider LLM",
        options=providers,
        index=providers.index(settings.provider) if settings.provider in providers else 0,
        help="Scegli se usare OpenAI (GPT), Anthropic (Claude) oppure Ollama in locale.",
    )
    if provider != settings.provider:
        settings.provider = provider
        ui._service.save_user_llm_settings(settings)
        ui._service._agent = None
        st.rerun()
    return provider


def _render_legacy_claude_setup(ui, settings) -> None:
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
                ui._CLAUDE_CODE_SETUP_URL,
                use_container_width=True,
            )
        with col_console:
            st.link_button(
                "Console Anthropic",
                ui._ANTHROPIC_CONSOLE_KEYS_URL,
                use_container_width=True,
            )
        if settings.anthropic_setup_token:
            st.info("Setup-token legacy presente nel database, ma non viene usato dal runtime.")
