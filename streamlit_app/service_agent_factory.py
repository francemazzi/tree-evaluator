from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import streamlit as st

logger = logging.getLogger(__name__)


def get_or_create_agent(service, user_id: str, openai_api_key: Optional[str] = None):
    """Lazy-load the LangGraph agent using saved per-user preferences."""
    preferences = service.get_user_llm_settings(user_id)
    if openai_api_key is not None:
        preferences.openai_api_key = openai_api_key

    openai_oauth_tokens: dict = {}
    anthropic_oauth_tokens: dict = {}
    if preferences.provider == "openai":
        if preferences.openai_auth_method == "codex_oauth":
            openai_oauth_tokens = service._resolve_openai_oauth_tokens(preferences)
            if not openai_oauth_tokens.get("access_token"):
                return None
            service._agent = None
        elif not preferences.openai_api_key:
            return None
    elif preferences.provider == "anthropic":
        if preferences.anthropic_auth_method == "oauth":
            anthropic_oauth_tokens = service._resolve_anthropic_oauth_tokens(preferences)
            if not anthropic_oauth_tokens.get("access_token"):
                return None
            service._agent = None
        elif not preferences.anthropic_api_key:
            return None

    if service._agent is not None:
        return service._agent

    try:
        from streamlit_app.agent import TreeEvaluatorAgent

        dataset_config = _resolve_dataset_config()
        service._agent = TreeEvaluatorAgent(
            **_openai_kwargs(preferences, openai_oauth_tokens),
            **_anthropic_kwargs(preferences, anthropic_oauth_tokens),
            provider=preferences.provider,
            openai_chat_model=preferences.openai_chat_model,
            openai_embedding_model=preferences.openai_embedding_model,
            ollama_base_url=preferences.ollama_base_url,
            ollama_chat_model=preferences.ollama_chat_model,
            ollama_embedding_model=preferences.ollama_embedding_model,
            interface_language=preferences.interface_language,
            **dataset_config,
        )
        return service._agent
    except ImportError as exc:
        st.error(f"❌ Errore import dipendenze: {exc}\nInstalla: pip install -r requirements.txt")
        logger.error("Import error: %s", exc)
        return None
    except ValueError as exc:
        st.error(f"❌ Credenziali LLM non valide: {exc}")
        logger.error("ValueError: %s", exc)
        return None
    except Exception as exc:
        st.error(f"❌ Errore inizializzazione agent: {exc}")
        logger.exception("Agent init error: %s", exc)
        return None


def _resolve_dataset_config() -> dict:
    custom_db_path = st.session_state.get("custom_db_path", None)
    custom_table_name = st.session_state.get("custom_table_name", None)
    data_description = st.session_state.get(
        "stored_data_description",
        st.session_state.get("data_description_input", ""),
    )
    dataset_metadata = st.session_state.get("dataset_metadata", {}) or {}
    dataset_column_roles = (
        dataset_metadata.get("profile", {}).get("roles", {})
        if isinstance(dataset_metadata, dict)
        else {}
    )
    profile_summary = dataset_metadata.get("profile_summary", "")
    if profile_summary:
        data_description = (
            f"{data_description.strip()}\n\nProfilo dataset:\n{profile_summary}"
            if data_description and data_description.strip()
            else f"Profilo dataset:\n{profile_summary}"
        )

    if custom_db_path and custom_table_name:
        return {
            "custom_db_path": Path(custom_db_path),
            "custom_table_name": custom_table_name,
            "dataset_column_roles": dataset_column_roles,
            "data_description": data_description,
        }
    if st.session_state.get("selected_preset", "vienna") == "milano":
        return {"dataset_preset": "milano"}
    return {}


def _openai_kwargs(preferences, openai_oauth_tokens: dict) -> dict:
    return {
        "openai_api_key": (
            preferences.openai_api_key or None
            if preferences.openai_auth_method == "api_key"
            else None
        ),
        "openai_auth_method": preferences.openai_auth_method,
        "openai_codex_access_token": str(openai_oauth_tokens.get("access_token") or "") or None,
        "openai_codex_account_id": str(openai_oauth_tokens.get("account_id") or "") or None,
        "openai_codex_is_fedramp": bool(openai_oauth_tokens.get("is_fedramp_account") or False),
    }


def _anthropic_kwargs(preferences, anthropic_oauth_tokens: dict) -> dict:
    return {
        "anthropic_auth_method": preferences.anthropic_auth_method,
        "anthropic_api_key": (
            preferences.anthropic_api_key or None
            if preferences.anthropic_auth_method == "api_key"
            else None
        ),
        "anthropic_oauth_access_token": str(anthropic_oauth_tokens.get("access_token") or "") or None,
        "anthropic_chat_model": preferences.anthropic_chat_model,
    }
