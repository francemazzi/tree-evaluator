from __future__ import annotations

import sqlite3

from streamlit_app.repository import ChatRepository
from streamlit_app.service import ChatService


def test_external_auth_profiles_are_persisted(tmp_path):
    repository = ChatRepository(tmp_path / "chat.db")
    service = ChatService(repository)

    settings = service.get_user_llm_settings("guest")
    settings.openai_auth_method = "codex_oauth"
    settings.openai_codex_oauth_token = "openai-refresh-token"
    settings.anthropic_setup_token = "anthropic-setup-token"
    settings.anthropic_auth_method = "oauth"
    settings.anthropic_oauth_refresh_token = "anthropic-refresh-token"
    settings.anthropic_api_key = "sk-ant-existing"
    settings.anthropic_chat_model = "claude-3-5-haiku-latest"
    service.save_user_llm_settings(settings)

    reloaded = service.get_user_llm_settings("guest")

    assert reloaded.openai_auth_method == "codex_oauth"
    assert reloaded.openai_codex_oauth_token == "openai-refresh-token"
    assert reloaded.anthropic_setup_token == "anthropic-setup-token"
    assert reloaded.anthropic_auth_method == "oauth"
    assert reloaded.anthropic_oauth_refresh_token == "anthropic-refresh-token"
    assert reloaded.anthropic_api_key == "sk-ant-existing"
    assert reloaded.anthropic_chat_model == "claude-3-5-haiku-latest"


def test_external_auth_columns_are_added_to_existing_settings_table(tmp_path):
    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path.as_posix()) as connection:
        connection.execute(
            """
            CREATE TABLE user_settings (
                user_id TEXT PRIMARY KEY,
                openai_api_key TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO user_settings (user_id, openai_api_key, updated_at)
            VALUES ('guest', 'sk-existing', '2026-05-06T00:00:00+00:00')
            """
        )

    repository = ChatRepository(db_path)
    service = ChatService(repository)

    settings = service.get_user_llm_settings("guest")
    settings.openai_auth_method = "codex_oauth"
    settings.openai_codex_oauth_token = "migrated-refresh-token"
    settings.anthropic_auth_method = "oauth"
    settings.anthropic_oauth_refresh_token = "anthropic-migrated-refresh-token"
    settings.anthropic_api_key = "sk-ant-migrated"
    settings.anthropic_chat_model = "claude-sonnet-4-5"
    service.save_user_llm_settings(settings)

    reloaded = service.get_user_llm_settings("guest")

    assert reloaded.openai_api_key == "sk-existing"
    assert reloaded.openai_auth_method == "codex_oauth"
    assert reloaded.openai_codex_oauth_token == "migrated-refresh-token"
    assert reloaded.anthropic_auth_method == "oauth"
    assert reloaded.anthropic_oauth_refresh_token == "anthropic-migrated-refresh-token"
    assert reloaded.anthropic_api_key == "sk-ant-migrated"
    assert reloaded.anthropic_chat_model == "claude-sonnet-4-5"


def test_codex_oauth_profile_refreshes_backend_tokens(tmp_path, monkeypatch):
    repository = ChatRepository(tmp_path / "chat.db")
    service = ChatService(repository)

    settings = service.get_user_llm_settings("guest")
    settings.provider = "openai"
    settings.openai_auth_method = "codex_oauth"
    settings.openai_codex_oauth_token = "refresh-token"
    service.save_user_llm_settings(settings)

    def fake_refresh_access_token(refresh_token: str) -> dict:
        assert refresh_token == "refresh-token"
        return {
            "access_token": "access-token",
            "refresh_token": "next-refresh-token",
            "account_id": "account-123",
            "is_fedramp_account": True,
        }

    monkeypatch.setattr(
        "streamlit_app.llm.openai_oauth.refresh_access_token",
        fake_refresh_access_token,
    )

    tokens = service._resolve_openai_oauth_tokens(settings)
    reloaded = service.get_user_llm_settings("guest")

    assert tokens == {
        "access_token": "access-token",
        "account_id": "account-123",
        "is_fedramp_account": True,
    }
    assert reloaded.openai_codex_oauth_token == "next-refresh-token"


def test_codex_oauth_errors_are_not_rendered_as_echo(tmp_path):
    repository = ChatRepository(tmp_path / "chat.db")
    service = ChatService(repository)

    settings = service.get_user_llm_settings("guest")
    settings.provider = "openai"
    settings.openai_auth_method = "codex_oauth"

    message = service._format_llm_error_for_user(
        settings,
        ValueError("Codex backend error 401: token expired"),
        "ciao",
    )

    assert not message.startswith("Echo")
    assert "backend ChatGPT/Codex" in message
    assert "Codex backend error 401" in message


def test_claude_oauth_profile_refreshes_backend_tokens(tmp_path, monkeypatch):
    repository = ChatRepository(tmp_path / "chat.db")
    service = ChatService(repository)

    settings = service.get_user_llm_settings("guest")
    settings.provider = "anthropic"
    settings.anthropic_auth_method = "oauth"
    settings.anthropic_oauth_refresh_token = "refresh-token"
    service.save_user_llm_settings(settings)

    def fake_refresh_access_token(refresh_token: str) -> dict:
        assert refresh_token == "refresh-token"
        return {
            "access_token": "access-token",
            "refresh_token": "next-refresh-token",
            "expires_at": 123456789,
        }

    monkeypatch.setattr(
        "streamlit_app.llm.anthropic_oauth.refresh_access_token",
        fake_refresh_access_token,
    )

    tokens = service._resolve_anthropic_oauth_tokens(settings)
    reloaded = service.get_user_llm_settings("guest")

    assert tokens == {
        "access_token": "access-token",
        "expires_at": 123456789,
    }
    assert reloaded.anthropic_oauth_refresh_token == "next-refresh-token"


def test_claude_errors_are_not_rendered_as_echo(tmp_path):
    repository = ChatRepository(tmp_path / "chat.db")
    service = ChatService(repository)

    settings = service.get_user_llm_settings("guest")
    settings.provider = "anthropic"
    settings.anthropic_auth_method = "oauth"

    message = service._format_llm_error_for_user(
        settings,
        ValueError("Anthropic backend error 401: token expired"),
        "ciao",
    )

    assert not message.startswith("Echo")
    assert "backend Claude/Anthropic" in message
    assert "Anthropic backend error 401" in message
