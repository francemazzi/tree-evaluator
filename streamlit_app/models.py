from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, Optional


Role = Literal["user", "assistant"]


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

    @staticmethod
    def new(user_id: str, conversation_id: int, role: Role, content: str) -> "ChatMessage":
        created_at = datetime.now(tz=timezone.utc)
        return ChatMessage(
            user_id=user_id,
            conversation_id=conversation_id,
            role=role,
            content=content,
            created_at=created_at,
        )

    def to_persistence_tuple(self) -> tuple[str, int, str, str, str]:
        """Return tuple for INSERT: (user_id, conversation_id, role, content, created_at)."""
        return (
            self.user_id,
            self.conversation_id,
            self.role,
            self.content,
            self.created_at.isoformat(),
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
        )


