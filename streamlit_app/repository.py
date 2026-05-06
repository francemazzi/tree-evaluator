from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, List, Optional

from streamlit_app.models import ChatMessage, Conversation


class ChatRepository:
    """SQLite-backed repository to persist conversations and messages.

    This repository manages schema creation and provides CRUD methods
    for conversations and their messages.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path.as_posix())
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            # Check if old schema exists (messages without conversation_id)
            cursor = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='messages'"
            )
            result = cursor.fetchone()
            if result and 'conversation_id' not in result[0]:
                # Drop old schema
                connection.execute("DROP TABLE IF EXISTS messages")
                connection.execute("DROP INDEX IF EXISTS idx_messages_user_time")
            
            # Create conversations table
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversations_user
                ON conversations (user_id, updated_at DESC);
                """
            )
            
            # Create messages table
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    reasoning TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages (conversation_id, created_at);
                """
            )
            # Migrate: add reasoning column if not exists
            self._ensure_messages_reasoning_column(connection)
            
            # Create user_settings table for persistent settings
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS user_settings (
                    user_id TEXT PRIMARY KEY,
                    openai_api_key TEXT,
                    openai_auth_method TEXT,
                    openai_codex_oauth_token TEXT,
                    anthropic_setup_token TEXT,
                    anthropic_auth_method TEXT,
                    anthropic_oauth_refresh_token TEXT,
                    anthropic_api_key TEXT,
                    anthropic_chat_model TEXT,
                    llm_provider TEXT,
                    openai_chat_model TEXT,
                    openai_embedding_model TEXT,
                    ollama_base_url TEXT,
                    ollama_chat_model TEXT,
                    ollama_embedding_model TEXT,
                    updated_at TEXT NOT NULL
                );
                """
            )
            self._ensure_user_settings_columns(connection)

    def _ensure_messages_reasoning_column(self, connection: sqlite3.Connection) -> None:
        """Add reasoning column to messages table if it doesn't exist."""
        try:
            cursor = connection.execute("PRAGMA table_info(messages)")
            existing = {row[1] for row in cursor.fetchall()}  # type: ignore[index]
            if "reasoning" not in existing:
                connection.execute("ALTER TABLE messages ADD COLUMN reasoning TEXT")
        except Exception:
            pass  # Column might already exist or table doesn't exist yet

    def _ensure_user_settings_columns(self, connection: sqlite3.Connection) -> None:
        """Best-effort schema migration for user_settings (SQLite ALTER TABLE ADD COLUMN)."""
        try:
            cursor = connection.execute("PRAGMA table_info(user_settings)")
            existing = {row[1] for row in cursor.fetchall()}  # type: ignore[index]
        except Exception:
            return

        desired_columns = {
            "llm_provider": "TEXT",
            "openai_auth_method": "TEXT",
            "openai_codex_oauth_token": "TEXT",
            "anthropic_setup_token": "TEXT",
            "anthropic_auth_method": "TEXT",
            "anthropic_oauth_refresh_token": "TEXT",
            "anthropic_api_key": "TEXT",
            "anthropic_chat_model": "TEXT",
            "openai_chat_model": "TEXT",
            "openai_embedding_model": "TEXT",
            "ollama_base_url": "TEXT",
            "ollama_chat_model": "TEXT",
            "ollama_embedding_model": "TEXT",
            "interface_language": "TEXT",
        }
        for col, col_type in desired_columns.items():
            if col not in existing:
                try:
                    connection.execute(f"ALTER TABLE user_settings ADD COLUMN {col} {col_type}")
                except Exception:
                    # Ignore if running on older SQLite or concurrent migration.
                    pass

    # Conversation methods
    
    def create_conversation(self, conversation: Conversation) -> int:
        """Create a new conversation and return its ID."""
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO conversations (user_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                conversation.to_persistence_tuple(),
            )
            return int(cursor.lastrowid)

    def list_conversations_by_user(self, user_id: str) -> List[Conversation]:
        """List all conversations for a user, ordered by most recent update."""
        with self._connect() as connection:
            cursor = connection.execute(
                """
                SELECT id, user_id, title, created_at, updated_at
                FROM conversations
                WHERE user_id = ?
                ORDER BY datetime(updated_at) DESC
                """,
                (user_id,),
            )
            rows: Iterable[sqlite3.Row] = cursor.fetchall()
            return [Conversation.from_persistence_row(dict(r)) for r in rows]

    def get_conversation(self, conversation_id: int) -> Optional[Conversation]:
        """Get a specific conversation by ID."""
        with self._connect() as connection:
            cursor = connection.execute(
                """
                SELECT id, user_id, title, created_at, updated_at
                FROM conversations
                WHERE id = ?
                """,
                (conversation_id,),
            )
            row = cursor.fetchone()
            return Conversation.from_persistence_row(dict(row)) if row else None

    def update_conversation_timestamp(self, conversation_id: int) -> None:
        """Update the updated_at timestamp of a conversation."""
        from datetime import datetime, timezone
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE conversations
                SET updated_at = ?
                WHERE id = ?
                """,
                (datetime.now(tz=timezone.utc).isoformat(), conversation_id),
            )

    def update_conversation_title(self, conversation_id: int, new_title: str) -> None:
        """Update the title of a conversation."""
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE conversations
                SET title = ?
                WHERE id = ?
                """,
                (new_title, conversation_id),
            )

    def delete_conversation(self, conversation_id: int) -> int:
        """Delete a conversation and all its messages (CASCADE)."""
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM conversations WHERE id = ?",
                (conversation_id,),
            )
            return int(cursor.rowcount or 0)

    # Message methods
    
    def add_message(self, message: ChatMessage) -> int:
        """Add a message to a conversation."""
        from datetime import datetime, timezone
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO messages (user_id, conversation_id, role, content, created_at, reasoning)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                message.to_persistence_tuple(),
            )
            message_id = int(cursor.lastrowid)
            # Update conversation timestamp in the same transaction
            connection.execute(
                """
                UPDATE conversations
                SET updated_at = ?
                WHERE id = ?
                """,
                (datetime.now(tz=timezone.utc).isoformat(), message.conversation_id),
            )
            return message_id

    def list_messages_by_conversation(self, conversation_id: int) -> List[ChatMessage]:
        """List all messages in a conversation, ordered chronologically."""
        with self._connect() as connection:
            cursor = connection.execute(
                """
                SELECT user_id, conversation_id, role, content, created_at, reasoning
                FROM messages
                WHERE conversation_id = ?
                ORDER BY datetime(created_at) ASC
                """,
                (conversation_id,),
            )
            rows: Iterable[sqlite3.Row] = cursor.fetchall()
            return [ChatMessage.from_persistence_row(dict(r)) for r in rows]

    # User settings methods
    
    def save_user_settings(
        self,
        user_id: str,
        openai_api_key: str,
        openai_auth_method: str = "api_key",
        openai_codex_oauth_token: str = "",
        anthropic_setup_token: str = "",
        anthropic_auth_method: str = "oauth",
        anthropic_oauth_refresh_token: str = "",
        anthropic_api_key: str = "",
        anthropic_chat_model: str = "claude-sonnet-4-5",
        llm_provider: str = "openai",
        openai_chat_model: str = "gpt-5.5",
        openai_embedding_model: str = "text-embedding-3-small",
        ollama_base_url: str = "",
        ollama_chat_model: str = "qwen2.5:7b-instruct",
        ollama_embedding_model: str = "nomic-embed-text",
        interface_language: str = "it",
    ) -> None:
        """Save or update user settings (LLM provider + auth profile + models + interface language)."""
        from datetime import datetime, timezone
        from streamlit_app.llm.ollama_base_url import OllamaBaseUrlResolver
        with self._connect() as connection:
            resolved_ollama_base_url = ollama_base_url.strip() or OllamaBaseUrlResolver().resolve()
            connection.execute(
                """
                INSERT INTO user_settings (
                    user_id,
                    openai_api_key,
                    openai_auth_method,
                    openai_codex_oauth_token,
                    anthropic_setup_token,
                    anthropic_auth_method,
                    anthropic_oauth_refresh_token,
                    anthropic_api_key,
                    anthropic_chat_model,
                    llm_provider,
                    openai_chat_model,
                    openai_embedding_model,
                    ollama_base_url,
                    ollama_chat_model,
                    ollama_embedding_model,
                    interface_language,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    openai_api_key = excluded.openai_api_key,
                    openai_auth_method = excluded.openai_auth_method,
                    openai_codex_oauth_token = excluded.openai_codex_oauth_token,
                    anthropic_setup_token = excluded.anthropic_setup_token,
                    anthropic_auth_method = excluded.anthropic_auth_method,
                    anthropic_oauth_refresh_token = excluded.anthropic_oauth_refresh_token,
                    anthropic_api_key = excluded.anthropic_api_key,
                    anthropic_chat_model = excluded.anthropic_chat_model,
                    llm_provider = excluded.llm_provider,
                    openai_chat_model = excluded.openai_chat_model,
                    openai_embedding_model = excluded.openai_embedding_model,
                    ollama_base_url = excluded.ollama_base_url,
                    ollama_chat_model = excluded.ollama_chat_model,
                    ollama_embedding_model = excluded.ollama_embedding_model,
                    interface_language = excluded.interface_language,
                    updated_at = excluded.updated_at
                """,
                (
                    user_id,
                    openai_api_key,
                    openai_auth_method,
                    openai_codex_oauth_token,
                    anthropic_setup_token,
                    anthropic_auth_method,
                    anthropic_oauth_refresh_token,
                    anthropic_api_key,
                    anthropic_chat_model,
                    llm_provider,
                    openai_chat_model,
                    openai_embedding_model,
                    resolved_ollama_base_url,
                    ollama_chat_model,
                    ollama_embedding_model,
                    interface_language,
                    datetime.now(tz=timezone.utc).isoformat(),
                ),
            )
    
    def get_user_settings(self, user_id: str) -> Optional[dict]:
        """Get user settings (returns dict with openai_api_key, interface_language, etc.)."""
        with self._connect() as connection:
            cursor = connection.execute(
                """
                SELECT
                    openai_api_key,
                    openai_auth_method,
                    openai_codex_oauth_token,
                    anthropic_setup_token,
                    anthropic_auth_method,
                    anthropic_oauth_refresh_token,
                    anthropic_api_key,
                    anthropic_chat_model,
                    llm_provider,
                    openai_chat_model,
                    openai_embedding_model,
                    ollama_base_url,
                    ollama_chat_model,
                    ollama_embedding_model,
                    interface_language
                FROM user_settings
                WHERE user_id = ?
                """,
                (user_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

