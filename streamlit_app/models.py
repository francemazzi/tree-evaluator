from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal


Role = Literal["user", "assistant"]


@dataclass
class ChatMessage:
    """Value object representing a single chat message.

    The class is intentionally simple and immutable in practice; use factory
    methods to create instances with sensible defaults.
    """

    user_id: str
    role: Role
    content: str
    created_at: datetime

    @staticmethod
    def new(user_id: str, role: Role, content: str) -> "ChatMessage":
        created_at = datetime.now(tz=timezone.utc)
        return ChatMessage(
            user_id=user_id,
            role=role,
            content=content,
            created_at=created_at,
        )

    def to_persistence_tuple(self) -> tuple[str, str, str, str]:
        """Return a tuple matching the persistence layer column order.

        Order: (user_id, role, content, created_at_iso)
        """
        return (
            self.user_id,
            self.role,
            self.content,
            self.created_at.isoformat(),
        )

    @staticmethod
    def from_persistence_row(row: dict) -> "ChatMessage":
        created_at = datetime.fromisoformat(row["created_at"])  # type: ignore[arg-type]
        return ChatMessage(
            user_id=row["user_id"],
            role=row["role"],
            content=row["content"],
            created_at=created_at,
        )


