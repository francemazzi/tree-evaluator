from __future__ import annotations

from typing import Any

from streamlit_app.llm.factory import LlmProvider, LlmSettings


def create_chat_without_tools(
    settings: LlmSettings,
    model: str,
    temperature: float,
) -> Any:
    """Create a provider-specific chat model without bound tools."""
    if settings.provider == LlmProvider.OLLAMA:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            temperature=temperature,
            base_url=settings.ollama_base_url,
        )

    if settings.provider == LlmProvider.ANTHROPIC:
        return _create_anthropic_chat_without_tools(settings, model, temperature)

    return _create_openai_chat_without_tools(settings, model, temperature)


def _create_anthropic_chat_without_tools(
    settings: LlmSettings,
    model: str,
    temperature: float,
) -> Any:
    if settings.anthropic_auth_method == "oauth":
        from streamlit_app.llm.anthropic_backend import (
            ClaudeOAuthChatModel,
            resolve_claude_oauth_model,
        )

        return ClaudeOAuthChatModel(
            model_name=resolve_claude_oauth_model(model),
            access_token=settings.anthropic_oauth_access_token or "",
            temperature=temperature,
        )

    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=model,
        temperature=temperature,
        anthropic_api_key=settings.anthropic_api_key,
    )


def _create_openai_chat_without_tools(
    settings: LlmSettings,
    model: str,
    temperature: float,
) -> Any:
    if settings.openai_auth_method == "codex_oauth":
        from streamlit_app.llm.codex_backend import (
            ChatGPTCodexBackendChatModel,
            resolve_codex_oauth_model,
        )

        return ChatGPTCodexBackendChatModel(
            model_name=resolve_codex_oauth_model(model),
            access_token=settings.openai_codex_access_token or "",
            account_id=settings.openai_codex_account_id,
            is_fedramp_account=settings.openai_codex_is_fedramp,
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=settings.openai_api_key,
    )
