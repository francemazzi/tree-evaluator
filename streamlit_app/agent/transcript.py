"""Structured session transcript.

Records every agent event (tool calls, LLM calls, errors, loop detections)
as newline-delimited JSON (JSONL) in chat_data/ for debugging and analytics.

Usage::

    transcript = SessionTranscript("session-123")
    transcript.log_tool_call("query_tree_dataset", {"query": "top species"})
    transcript.log_tool_result("query_tree_dataset", {"rows": 10}, duration_ms=42)
    transcript.log_error("query_tree_dataset", "timeout after 30s")
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Optional

logger = logging.getLogger(__name__)

_TRANSCRIPT_DIR = Path(__file__).parent.parent.parent / "chat_data"

EventType = Literal[
    "tool_call",
    "tool_result",
    "llm_call",
    "llm_response",
    "error",
    "loop_detected",
    "replan",
    "budget_exceeded",
    "user_message",
    "agent_response",
]


class SessionTranscript:
    """Append-only structured event log for a single agent session.

    Each event is a JSON object on its own line, written immediately
    to ``chat_data/{session_id}.jsonl``.
    """

    def __init__(
        self,
        session_id: str,
        transcript_dir: Optional[Path] = None,
    ) -> None:
        self._session_id = session_id
        self._dir = transcript_dir or _TRANSCRIPT_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / f"{session_id}.jsonl"
        self._event_seq = 0
        self._start_time = time.monotonic()

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def log_tool_call(
        self, tool_name: str, args: Dict[str, Any]
    ) -> None:
        self._write_event("tool_call", tool=tool_name, data={"args": args})

    def log_tool_result(
        self,
        tool_name: str,
        result: Any,
        duration_ms: Optional[float] = None,
    ) -> None:
        data: Dict[str, Any] = {"result_preview": _truncate(str(result), 500)}
        if duration_ms is not None:
            data["duration_ms"] = round(duration_ms, 1)
        self._write_event("tool_result", tool=tool_name, data=data)

    def log_llm_call(
        self, model: str, message_count: int, token_estimate: Optional[int] = None
    ) -> None:
        data: Dict[str, Any] = {"model": model, "message_count": message_count}
        if token_estimate is not None:
            data["token_estimate"] = token_estimate
        self._write_event("llm_call", data=data)

    def log_llm_response(
        self, model: str, duration_ms: Optional[float] = None, tokens_used: Optional[int] = None
    ) -> None:
        data: Dict[str, Any] = {"model": model}
        if duration_ms is not None:
            data["duration_ms"] = round(duration_ms, 1)
        if tokens_used is not None:
            data["tokens_used"] = tokens_used
        self._write_event("llm_response", data=data)

    def log_error(
        self, source: str, error: str, recoverable: bool = True
    ) -> None:
        self._write_event(
            "error",
            data={"source": source, "error": _truncate(error, 1000), "recoverable": recoverable},
        )

    def log_loop_detected(
        self, tool_name: str, reason: str, call_count: int, action: str
    ) -> None:
        self._write_event(
            "loop_detected",
            tool=tool_name,
            data={"reason": reason, "call_count": call_count, "action": action},
        )

    def log_replan(self, reason: str, replan_count: int) -> None:
        self._write_event("replan", data={"reason": reason, "replan_count": replan_count})

    def log_budget_exceeded(self, budget_status: Dict[str, Any]) -> None:
        self._write_event("budget_exceeded", data=budget_status)

    def log_user_message(self, message_preview: str) -> None:
        self._write_event("user_message", data={"preview": _truncate(message_preview, 200)})

    def log_agent_response(self, response_preview: str) -> None:
        self._write_event("agent_response", data={"preview": _truncate(response_preview, 300)})

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    @property
    def path(self) -> Path:
        """Path to the JSONL transcript file."""
        return self._path

    @property
    def event_count(self) -> int:
        """Number of events logged so far."""
        return self._event_seq

    def _write_event(
        self,
        event_type: EventType,
        tool: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write a single event to the JSONL file."""
        self._event_seq += 1
        elapsed_ms = round((time.monotonic() - self._start_time) * 1000, 1)

        event: Dict[str, Any] = {
            "seq": self._event_seq,
            "ts": datetime.now(timezone.utc).isoformat(),
            "elapsed_ms": elapsed_ms,
            "session_id": self._session_id,
            "event": event_type,
        }
        if tool:
            event["tool"] = tool
        if data:
            event["data"] = data

        try:
            line = json.dumps(event, default=str, ensure_ascii=False)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as exc:
            logger.warning("Failed to write transcript event: %s", exc)


def _truncate(text: str, max_len: int) -> str:
    """Truncate text with ellipsis if needed."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
