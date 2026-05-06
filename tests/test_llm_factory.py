from __future__ import annotations

import sys
import types

from streamlit_app.llm.anthropic_backend import CLAUDE_OAUTH_DEFAULT_MODEL
from streamlit_app.llm.codex_backend import CODEX_OAUTH_DEFAULT_MODEL
from streamlit_app.llm.factory import LlmFactory, LlmSettings, LlmProvider


def _oauth_settings(chat_model: str) -> LlmSettings:
    return LlmSettings(
        provider=LlmProvider.OPENAI,
        openai_auth_method="codex_oauth",
        openai_api_key=None,
        openai_codex_access_token="access-token",
        openai_codex_account_id="account-123",
        openai_codex_is_fedramp=False,
        openai_chat_model=chat_model,
        openai_fallback_model=chat_model,
        openai_embedding_model="text-embedding-3-small",
        anthropic_auth_method="oauth",
        anthropic_api_key=None,
        anthropic_oauth_access_token=None,
        anthropic_chat_model="claude-sonnet-4-5",
        anthropic_fallback_model="claude-sonnet-4-5",
        ollama_base_url="http://localhost:11434",
        ollama_chat_model="qwen2.5:7b-instruct",
        ollama_fallback_model="",
        ollama_embedding_model="nomic-embed-text",
        chat_temperature=1.0,
        fallback_temperature=0.7,
    )


def test_factory_substitutes_unsupported_oauth_chat_model():
    factory = LlmFactory(_oauth_settings("gpt-5"))

    chat = factory.create_chat_model()

    assert chat.model_name == CODEX_OAUTH_DEFAULT_MODEL


def test_factory_keeps_supported_oauth_chat_model():
    factory = LlmFactory(_oauth_settings("gpt-5.4-mini"))

    chat = factory.create_chat_model()

    assert chat.model_name == "gpt-5.4-mini"


def test_factory_substitutes_unsupported_oauth_fallback_model():
    settings = _oauth_settings("gpt-5")
    factory = LlmFactory(settings)

    fallback = factory.create_fallback_chat_model()

    assert fallback.model_name == CODEX_OAUTH_DEFAULT_MODEL


def _anthropic_oauth_settings(chat_model: str) -> LlmSettings:
    settings = _oauth_settings("gpt-5.5")
    return LlmSettings(
        provider=LlmProvider.ANTHROPIC,
        openai_auth_method=settings.openai_auth_method,
        openai_api_key=settings.openai_api_key,
        openai_codex_access_token=settings.openai_codex_access_token,
        openai_codex_account_id=settings.openai_codex_account_id,
        openai_codex_is_fedramp=settings.openai_codex_is_fedramp,
        openai_chat_model=settings.openai_chat_model,
        openai_fallback_model=settings.openai_fallback_model,
        openai_embedding_model=settings.openai_embedding_model,
        anthropic_auth_method="oauth",
        anthropic_api_key=None,
        anthropic_oauth_access_token="access-token",
        anthropic_chat_model=chat_model,
        anthropic_fallback_model=chat_model,
        ollama_base_url=settings.ollama_base_url,
        ollama_chat_model=settings.ollama_chat_model,
        ollama_fallback_model=settings.ollama_fallback_model,
        ollama_embedding_model=settings.ollama_embedding_model,
        chat_temperature=settings.chat_temperature,
        fallback_temperature=settings.fallback_temperature,
    )


def test_factory_substitutes_unsupported_anthropic_oauth_chat_model():
    factory = LlmFactory(_anthropic_oauth_settings("claude-2"))

    chat = factory.create_chat_model()

    assert chat.model_name == CLAUDE_OAUTH_DEFAULT_MODEL


def test_factory_keeps_supported_anthropic_oauth_chat_model():
    factory = LlmFactory(_anthropic_oauth_settings("claude-3-5-haiku-latest"))

    chat = factory.create_chat_model()

    assert chat.model_name == "claude-3-5-haiku-latest"


def test_factory_uses_chat_anthropic_for_api_key(monkeypatch):
    class FakeChatAnthropic:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_module = types.SimpleNamespace(ChatAnthropic=FakeChatAnthropic)
    monkeypatch.setitem(sys.modules, "langchain_anthropic", fake_module)

    settings = _anthropic_oauth_settings("claude-sonnet-4-5")
    settings = LlmSettings(
        provider=settings.provider,
        openai_auth_method=settings.openai_auth_method,
        openai_api_key=settings.openai_api_key,
        openai_codex_access_token=settings.openai_codex_access_token,
        openai_codex_account_id=settings.openai_codex_account_id,
        openai_codex_is_fedramp=settings.openai_codex_is_fedramp,
        openai_chat_model=settings.openai_chat_model,
        openai_fallback_model=settings.openai_fallback_model,
        openai_embedding_model=settings.openai_embedding_model,
        anthropic_auth_method="api_key",
        anthropic_api_key="sk-ant-test",
        anthropic_oauth_access_token=None,
        anthropic_chat_model=settings.anthropic_chat_model,
        anthropic_fallback_model=settings.anthropic_fallback_model,
        ollama_base_url=settings.ollama_base_url,
        ollama_chat_model=settings.ollama_chat_model,
        ollama_fallback_model=settings.ollama_fallback_model,
        ollama_embedding_model=settings.ollama_embedding_model,
        chat_temperature=settings.chat_temperature,
        fallback_temperature=settings.fallback_temperature,
    )

    chat = LlmFactory(settings).create_chat_model()

    assert isinstance(chat, FakeChatAnthropic)
    assert chat.kwargs["model"] == "claude-sonnet-4-5"
    assert chat.kwargs["anthropic_api_key"] == "sk-ant-test"
