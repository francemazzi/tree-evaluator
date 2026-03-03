"""Persistent memory across sessions — OpenClaw MEMORY.md pattern.

Stores key facts from conversations as Markdown files in chat_data/.
Supports vector-similarity recall when embeddings are available.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


_MEMORY_DIR = Path(__file__).parent.parent.parent / "chat_data"
_MAX_FACTS = 200          # Max facts per session memory
_MAX_PROMPT_CHARS = 4_000  # Max chars injected into prompt


class PersistentMemory:
    """Manages persistent memory for a chat session.

    Memory is stored both as a human-readable Markdown file and in an
    SQLite database for structured retrieval.
    """

    def __init__(
        self,
        session_id: str,
        memory_dir: Optional[Path] = None,
    ) -> None:
        self._session_id = session_id
        self._dir = memory_dir or _MEMORY_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._md_path = self._dir / f"{session_id}_memory.md"
        self._db_path = self._dir / "memory.db"
        self._ensure_db()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_fact(self, fact: str, category: str = "general") -> None:
        """Append a fact to persistent memory.

        Args:
            fact: The fact to store (one concise sentence).
            category: Grouping category (e.g., "user_preference", "dataset", "result").
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        # Append to Markdown file
        with self._md_path.open("a", encoding="utf-8") as f:
            f.write(f"- [{category}] {fact}  \n")

        # Insert into SQLite
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                "INSERT INTO memories (session_id, category, fact, created_at) VALUES (?, ?, ?, ?)",
                (self._session_id, category, fact, timestamp),
            )
            conn.commit()
        finally:
            conn.close()

    def load_facts(self, limit: int = _MAX_FACTS) -> List[Dict[str, str]]:
        """Load all facts for this session.

        Returns:
            List of dicts with keys: category, fact, created_at.
        """
        conn = sqlite3.connect(self._db_path)
        try:
            rows = conn.execute(
                "SELECT category, fact, created_at FROM memories WHERE session_id = ? ORDER BY rowid DESC LIMIT ?",
                (self._session_id, limit),
            ).fetchall()
            return [{"category": r[0], "fact": r[1], "created_at": r[2]} for r in rows]
        finally:
            conn.close()

    def load_all_facts(self, limit: int = _MAX_FACTS) -> List[Dict[str, str]]:
        """Load facts across ALL sessions (for cross-session recall).

        Returns:
            List of dicts with keys: session_id, category, fact, created_at.
        """
        conn = sqlite3.connect(self._db_path)
        try:
            rows = conn.execute(
                "SELECT session_id, category, fact, created_at FROM memories ORDER BY rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [{"session_id": r[0], "category": r[1], "fact": r[2], "created_at": r[3]} for r in rows]
        finally:
            conn.close()

    def get_memory_for_prompt(self, max_chars: int = _MAX_PROMPT_CHARS) -> str:
        """Format memory as a section to inject into the system prompt.

        Returns a concise summary of stored facts, capped at max_chars.
        """
        facts = self.load_facts()
        if not facts:
            return ""

        lines = ["## Memory (previous sessions)"]
        total = len(lines[0])

        for fact in facts:
            line = f"- [{fact['category']}] {fact['fact']}"
            if total + len(line) + 1 > max_chars:
                break
            lines.append(line)
            total += len(line) + 1

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_db(self) -> None:
        """Create the memories table if it doesn't exist."""
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'general',
                    fact TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id)"
            )
            conn.commit()
        finally:
            conn.close()
