from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, List

from .models import ChatMessage


class ChatRepository:
    """SQLite-backed repository to persist and query chat messages per user.

    This repository manages schema creation and provides simple CRUD methods
    tailored to the chat use case.
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
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_user_time
                ON messages (user_id, created_at);
                """
            )

    def add_message(self, message: ChatMessage) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO messages (user_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                message.to_persistence_tuple(),
            )
            return int(cursor.lastrowid)

    def list_messages_by_user(self, user_id: str) -> List[ChatMessage]:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                SELECT user_id, role, content, created_at
                FROM messages
                WHERE user_id = ?
                ORDER BY datetime(created_at) ASC
                """,
                (user_id,),
            )
            rows: Iterable[sqlite3.Row] = cursor.fetchall()
            return [ChatMessage.from_persistence_row(dict(r)) for r in rows]

    def clear_user_history(self, user_id: str) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM messages WHERE user_id = ?",
                (user_id,),
            )
            return int(cursor.rowcount or 0)


