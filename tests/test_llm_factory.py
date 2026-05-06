from __future__ import annotations

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
