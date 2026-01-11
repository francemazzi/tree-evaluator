from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, Optional

from streamlit_app.llm.ollama_base_url import OllamaBaseUrlResolver

Role = Literal["user", "assistant"]


@dataclass
class UserLlmSettings:
    """Persisted per-user LLM preferences used by Streamlit UI and agent initialization."""

    user_id: str
    provider: str  # "openai" | "ollama"
    openai_api_key: str
    openai_chat_model: str
    openai_embedding_model: str
    ollama_base_url: str
    ollama_chat_model: str
    ollama_embedding_model: str
    interface_language: str  # "it" | "en" - Language for agent responses

    @staticmethod
    def default(user_id: str) -> "UserLlmSettings":
        return UserLlmSettings(
            user_id=user_id,
            provider="openai",
            openai_api_key="",
            openai_chat_model="gpt-5",
            openai_embedding_model="text-embedding-3-small",
            ollama_base_url=OllamaBaseUrlResolver().resolve(),
            ollama_chat_model="qwen2.5:7b-instruct",
            ollama_embedding_model="nomic-embed-text",
            interface_language="it",
        )


@dataclass
class Conversation:
    """Value object representing a chat conversation/session."""

    id: int
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime

    @staticmethod
    def new(user_id: str, title: str) -> "Conversation":
        now = datetime.now(tz=timezone.utc)
        return Conversation(
            id=0,  # will be set by DB
            user_id=user_id,
            title=title,
            created_at=now,
            updated_at=now,
        )

    def to_persistence_tuple(self) -> tuple[str, str, str, str]:
        """Return tuple for INSERT: (user_id, title, created_at, updated_at)."""
        return (
            self.user_id,
            self.title,
            self.created_at.isoformat(),
            self.updated_at.isoformat(),
        )

    @staticmethod
    def from_persistence_row(row: dict) -> "Conversation":
        return Conversation(
            id=row["id"],
            user_id=row["user_id"],
            title=row["title"],
            created_at=datetime.fromisoformat(row["created_at"]),  # type: ignore[arg-type]
            updated_at=datetime.fromisoformat(row["updated_at"]),  # type: ignore[arg-type]
        )


@dataclass
class ChatMessage:
    """Value object representing a single chat message."""

    user_id: str
    conversation_id: int
    role: Role
    content: str
    created_at: datetime
    reasoning: Optional[str] = None  # Reasoning steps for assistant messages

    @staticmethod
    def new(user_id: str, conversation_id: int, role: Role, content: str, reasoning: Optional[str] = None) -> "ChatMessage":
        created_at = datetime.now(tz=timezone.utc)
        return ChatMessage(
            user_id=user_id,
            conversation_id=conversation_id,
            role=role,
            content=content,
            created_at=created_at,
            reasoning=reasoning,
        )

    def to_persistence_tuple(self) -> tuple[str, int, str, str, str, Optional[str]]:
        """Return tuple for INSERT: (user_id, conversation_id, role, content, created_at, reasoning)."""
        return (
            self.user_id,
            self.conversation_id,
            self.role,
            self.content,
            self.created_at.isoformat(),
            self.reasoning,
        )

    @staticmethod
    def from_persistence_row(row: dict) -> "ChatMessage":
        created_at = datetime.fromisoformat(row["created_at"])  # type: ignore[arg-type]
        return ChatMessage(
            user_id=row["user_id"],
            conversation_id=row["conversation_id"],
            role=row["role"],
            content=row["content"],
            created_at=created_at,
            reasoning=row.get("reasoning"),
        )


