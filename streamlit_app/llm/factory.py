from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from streamlit_app.llm.ollama_base_url import OllamaBaseUrlResolver

class LlmProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"

    @classmethod
    def from_string(cls, value: Optional[str]) -> "LlmProvider":
        normalized = (value or "").strip().lower()
        if normalized in ("ollama",):
            return cls.OLLAMA
        if normalized in ("anthropic", "claude"):
            return cls.ANTHROPIC
        return cls.OPENAI


@dataclass(frozen=True)
class LlmSettings:
    provider: LlmProvider

    # OpenAI
    openai_auth_method: str
    openai_api_key: Optional[str]
    openai_codex_access_token: Optional[str]
    openai_codex_account_id: Optional[str]
    openai_codex_is_fedramp: bool
    openai_chat_model: str
    openai_fallback_model: str
    openai_embedding_model: str

    # Anthropic
    anthropic_auth_method: str
    anthropic_api_key: Optional[str]
    anthropic_oauth_access_token: Optional[str]
    anthropic_chat_model: str
    anthropic_fallback_model: str

    # Ollama
    ollama_base_url: str
    ollama_chat_model: str
    ollama_fallback_model: str
    ollama_embedding_model: str

    # Temperatures
    chat_temperature: float
    fallback_temperature: float


class LlmSettingsReader:
    """Reads LLM settings from environment variables (single source of truth)."""

    def read(
        self,
        openai_api_key_override: Optional[str] = None,
        openai_auth_method_override: Optional[str] = None,
        openai_codex_access_token_override: Optional[str] = None,
        openai_codex_account_id_override: Optional[str] = None,
        openai_codex_is_fedramp_override: Optional[bool] = None,
        provider_override: Optional[str] = None,
        openai_chat_model_override: Optional[str] = None,
        openai_fallback_model_override: Optional[str] = None,
        openai_embedding_model_override: Optional[str] = None,
        anthropic_auth_method_override: Optional[str] = None,
        anthropic_api_key_override: Optional[str] = None,
        anthropic_oauth_access_token_override: Optional[str] = None,
        anthropic_chat_model_override: Optional[str] = None,
        anthropic_fallback_model_override: Optional[str] = None,
        ollama_base_url_override: Optional[str] = None,
        ollama_chat_model_override: Optional[str] = None,
        ollama_fallback_model_override: Optional[str] = None,
        ollama_embedding_model_override: Optional[str] = None,
    ) -> LlmSettings:
        provider = LlmProvider.from_string(provider_override or os.getenv("LLM_PROVIDER"))

        openai_api_key = openai_api_key_override or os.getenv("OPENAI_API_KEY")
        openai_auth_method = (
            openai_auth_method_override
            or os.getenv("OPENAI_AUTH_METHOD")
            or "api_key"
        )
        if openai_auth_method not in {"api_key", "codex_oauth"}:
            openai_auth_method = "api_key"
        anthropic_auth_method = (
            anthropic_auth_method_override
            or os.getenv("ANTHROPIC_AUTH_METHOD")
            or "oauth"
        )
        if anthropic_auth_method not in {"oauth", "api_key"}:
            anthropic_auth_method = "oauth"

        return LlmSettings(
            provider=provider,
            openai_auth_method=openai_auth_method,
            openai_api_key=openai_api_key,
            openai_codex_access_token=(
                openai_codex_access_token_override
                or os.getenv("OPENAI_CODEX_ACCESS_TOKEN")
            ),
            openai_codex_account_id=(
                openai_codex_account_id_override
                or os.getenv("OPENAI_CODEX_ACCOUNT_ID")
            ),
            openai_codex_is_fedramp=(
                openai_codex_is_fedramp_override
                if openai_codex_is_fedramp_override is not None
                else os.getenv("OPENAI_CODEX_FEDRAMP", "").strip().lower() in {"1", "true", "yes"}
            ),
            openai_chat_model=openai_chat_model_override or os.getenv("OPENAI_CHAT_MODEL", "gpt-4o"),
            openai_fallback_model=openai_fallback_model_override or os.getenv("OPENAI_FALLBACK_MODEL", "gpt-4o-mini"),
            openai_embedding_model=openai_embedding_model_override or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            anthropic_auth_method=anthropic_auth_method,
            anthropic_api_key=anthropic_api_key_override or os.getenv("ANTHROPIC_API_KEY"),
            anthropic_oauth_access_token=(
                anthropic_oauth_access_token_override
                or os.getenv("ANTHROPIC_OAUTH_ACCESS_TOKEN")
            ),
            anthropic_chat_model=anthropic_chat_model_override or os.getenv("ANTHROPIC_CHAT_MODEL", "claude-sonnet-4-5"),
            anthropic_fallback_model=anthropic_fallback_model_override or os.getenv("ANTHROPIC_FALLBACK_MODEL", "claude-sonnet-4-5"),
            ollama_base_url=(
                ollama_base_url_override
                or os.getenv("OLLAMA_BASE_URL")
                or OllamaBaseUrlResolver().resolve()
            ),
            ollama_chat_model=ollama_chat_model_override or os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b-instruct"),
            # Fallback is optional for Ollama; we intentionally ignore any env var fallback to avoid
            # "model not found" issues when the UI-selected model differs from env defaults.
            ollama_fallback_model=ollama_fallback_model_override or "",
            ollama_embedding_model=ollama_embedding_model_override or os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
            chat_temperature=float(os.getenv("LLM_TEMPERATURE", "1.0")),
            fallback_temperature=float(os.getenv("LLM_FALLBACK_TEMPERATURE", "0.7")),
        )


class LlmFactory:
    """Creates chat models and embeddings for the configured provider."""

    def __init__(self, settings: LlmSettings) -> None:
        self._settings = settings

    def get_provider(self) -> LlmProvider:
        return self._settings.provider

    def create_chat_model(self) -> Any:
        if self._settings.provider == LlmProvider.OLLAMA:
            return self._create_ollama_chat(
                model=self._settings.ollama_chat_model,
                temperature=self._settings.chat_temperature,
            )
        if self._settings.provider == LlmProvider.ANTHROPIC:
            return self._create_anthropic_chat(
                model=self._settings.anthropic_chat_model,
                temperature=self._settings.chat_temperature,
            )
        return self._create_openai_chat(
            model=self._settings.openai_chat_model,
            temperature=self._settings.chat_temperature,
        )

    def create_fallback_chat_model(self) -> Any:
        if self._settings.provider == LlmProvider.OLLAMA:
            fallback_model = (self._settings.ollama_fallback_model or "").strip() or self._settings.ollama_chat_model
            return self._create_ollama_chat(
                model=fallback_model,
                temperature=self._settings.fallback_temperature,
            )
        if self._settings.provider == LlmProvider.ANTHROPIC:
            fallback_model = (
                (self._settings.anthropic_fallback_model or "").strip()
                or self._settings.anthropic_chat_model
            )
            return self._create_anthropic_chat(
                model=fallback_model,
                temperature=self._settings.fallback_temperature,
            )
        if self._settings.openai_auth_method == "codex_oauth":
            return self._create_openai_chat(
                model=self._settings.openai_chat_model,
                temperature=self._settings.fallback_temperature,
            )
        return self._create_openai_chat(
            model=self._settings.openai_fallback_model,
            temperature=self._settings.fallback_temperature,
        )

    def create_embeddings(self) -> Any:
        if self._settings.provider == LlmProvider.OLLAMA:
            return self._create_ollama_embeddings(model=self._settings.ollama_embedding_model)
        if (
            self._settings.provider == LlmProvider.ANTHROPIC
            or self._settings.openai_auth_method == "codex_oauth"
        ):
            from streamlit_app.llm.codex_backend import DeterministicHashEmbeddings

            return DeterministicHashEmbeddings()
        return self._create_openai_embeddings(model=self._settings.openai_embedding_model)

    def validate(self) -> None:
        """Raise a ValueError if mandatory settings for the provider are missing."""
        if self._settings.provider == LlmProvider.OLLAMA:
            return
        if self._settings.provider == LlmProvider.ANTHROPIC:
            if self._settings.anthropic_auth_method == "oauth":
                if not self._settings.anthropic_oauth_access_token:
                    raise ValueError(
                        "Claude OAuth access token not found. Complete Claude login in the UI."
                    )
                return
            if not self._settings.anthropic_api_key:
                raise ValueError(
                    "Anthropic API key not found. Provide it via UI Settings or set ANTHROPIC_API_KEY."
                )
            return
        if self._settings.openai_auth_method == "codex_oauth":
            if not self._settings.openai_codex_access_token:
                raise ValueError(
                    "ChatGPT OAuth access token not found. Complete device pairing in the UI."
                )
            return
        if not self._settings.openai_api_key:
            raise ValueError(
                "OpenAI API key not found. Provide it via UI Settings or set OPENAI_API_KEY environment variable."
            )

    def _create_openai_chat(self, model: str, temperature: float) -> Any:
        if self._settings.openai_auth_method == "codex_oauth":
            from streamlit_app.llm.codex_backend import (
                ChatGPTCodexBackendChatModel,
                resolve_codex_oauth_model,
            )

            return ChatGPTCodexBackendChatModel(
                model_name=resolve_codex_oauth_model(model),
                access_token=self._settings.openai_codex_access_token or "",
                account_id=self._settings.openai_codex_account_id,
                is_fedramp_account=self._settings.openai_codex_is_fedramp,
            )

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=self._settings.openai_api_key,
        )

    def _create_anthropic_chat(self, model: str, temperature: float) -> Any:
        if self._settings.anthropic_auth_method == "oauth":
            from streamlit_app.llm.anthropic_backend import (
                ClaudeOAuthChatModel,
                resolve_claude_oauth_model,
            )

            return ClaudeOAuthChatModel(
                model_name=resolve_claude_oauth_model(model),
                access_token=self._settings.anthropic_oauth_access_token or "",
                temperature=temperature,
            )

        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model,
            temperature=temperature,
            anthropic_api_key=self._settings.anthropic_api_key,
        )

    def _create_openai_embeddings(self, model: str) -> Any:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model, api_key=self._settings.openai_api_key)

    def _create_ollama_chat(self, model: str, temperature: float) -> Any:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            temperature=temperature,
            base_url=self._settings.ollama_base_url,
        )

    def _create_ollama_embeddings(self, model: str) -> Any:
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=model,
            base_url=self._settings.ollama_base_url,
        )
