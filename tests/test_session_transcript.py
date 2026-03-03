"""Unit tests for SessionTranscript — no API key required."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# Load module directly to avoid agent/__init__.py pulling in langchain_core
_BASE = Path(__file__).parent.parent
_spec = importlib.util.spec_from_file_location(
    "streamlit_app.agent.transcript", _BASE / "streamlit_app" / "agent" / "transcript.py"
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["streamlit_app.agent.transcript"] = _mod
_spec.loader.exec_module(_mod)
SessionTranscript = _mod.SessionTranscript


class TestSessionTranscriptBasics:
    """Basic event logging."""

    def test_log_tool_call(self, tmp_path):
        t = SessionTranscript("test", transcript_dir=tmp_path)
        t.log_tool_call("query_tree_dataset", {"query": "top species"})

        assert t.event_count == 1
        events = _read_events(t)
        assert events[0]["event"] == "tool_call"
        assert events[0]["tool"] == "query_tree_dataset"
        assert events[0]["data"]["args"]["query"] == "top species"

    def test_log_tool_result(self, tmp_path):
        t = SessionTranscript("test", transcript_dir=tmp_path)
        t.log_tool_result("query_tree_dataset", {"rows": 10}, duration_ms=42.5)

        events = _read_events(t)
        assert events[0]["event"] == "tool_result"
        assert events[0]["data"]["duration_ms"] == 42.5

    def test_log_llm_call_and_response(self, tmp_path):
        t = SessionTranscript("test", transcript_dir=tmp_path)
        t.log_llm_call("gpt-4o", 5, token_estimate=1200)
        t.log_llm_response("gpt-4o", duration_ms=350.0, tokens_used=800)

        events = _read_events(t)
        assert events[0]["event"] == "llm_call"
        assert events[0]["data"]["model"] == "gpt-4o"
        assert events[1]["event"] == "llm_response"
        assert events[1]["data"]["tokens_used"] == 800

    def test_log_error(self, tmp_path):
        t = SessionTranscript("test", transcript_dir=tmp_path)
        t.log_error("test_tool", "timeout error", recoverable=True)

        events = _read_events(t)
        assert events[0]["event"] == "error"
        assert events[0]["data"]["recoverable"] is True

    def test_log_loop_detected(self, tmp_path):
        t = SessionTranscript("test", transcript_dir=tmp_path)
        t.log_loop_detected("query_tree_dataset", "identical_call_repeated", 2, "stop")

        events = _read_events(t)
        assert events[0]["event"] == "loop_detected"
        assert events[0]["data"]["action"] == "stop"

    def test_log_user_message(self, tmp_path):
        t = SessionTranscript("test", transcript_dir=tmp_path)
        t.log_user_message("quanti alberi ci sono?")

        events = _read_events(t)
        assert events[0]["event"] == "user_message"

    def test_log_agent_response(self, tmp_path):
        t = SessionTranscript("test", transcript_dir=tmp_path)
        t.log_agent_response("Ci sono 33.612 alberi.")

        events = _read_events(t)
        assert events[0]["event"] == "agent_response"


class TestSessionTranscriptIntegrity:
    """JSONL format and sequencing."""

    def test_event_sequence_numbers(self, tmp_path):
        t = SessionTranscript("test", transcript_dir=tmp_path)
        t.log_user_message("msg1")
        t.log_llm_call("gpt-4o", 1)
        t.log_llm_response("gpt-4o")
        t.log_agent_response("response")

        events = _read_events(t)
        seqs = [e["seq"] for e in events]
        assert seqs == [1, 2, 3, 4]

    def test_jsonl_valid_json(self, tmp_path):
        t = SessionTranscript("test", transcript_dir=tmp_path)
        t.log_tool_call("tool_a", {"key": "value"})
        t.log_error("src", "err")
        t.log_replan("reason", 1)
        t.log_budget_exceeded({"total": 10})

        lines = t.path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 4
        for line in lines:
            parsed = json.loads(line)  # Should not raise
            assert "seq" in parsed
            assert "ts" in parsed
            assert "event" in parsed
            assert "session_id" in parsed

    def test_elapsed_ms_increases(self, tmp_path):
        t = SessionTranscript("test", transcript_dir=tmp_path)
        t.log_user_message("a")
        t.log_user_message("b")

        events = _read_events(t)
        assert events[1]["elapsed_ms"] >= events[0]["elapsed_ms"]


class TestSessionTranscriptTruncation:
    """Long text truncation."""

    def test_long_result_truncated(self, tmp_path):
        t = SessionTranscript("test", transcript_dir=tmp_path)
        long_result = "x" * 2000
        t.log_tool_result("tool_a", long_result, duration_ms=10)

        events = _read_events(t)
        preview = events[0]["data"]["result_preview"]
        assert len(preview) <= 503  # 500 + "..."
        assert preview.endswith("...")

    def test_long_user_message_truncated(self, tmp_path):
        t = SessionTranscript("test", transcript_dir=tmp_path)
        t.log_user_message("x" * 500)

        events = _read_events(t)
        preview = events[0]["data"]["preview"]
        assert len(preview) <= 203


class TestSessionTranscriptFileCreation:
    """File path and creation."""

    def test_jsonl_file_created(self, tmp_path):
        t = SessionTranscript("my-session", transcript_dir=tmp_path)
        t.log_user_message("hello")

        assert t.path.exists()
        assert t.path.name == "my-session.jsonl"

    def test_path_property(self, tmp_path):
        t = SessionTranscript("test", transcript_dir=tmp_path)
        assert t.path == tmp_path / "test.jsonl"


# ── Helpers ─────────────────────────────────────────

def _read_events(transcript: SessionTranscript) -> list[dict]:
    """Read all events from a transcript file."""
    lines = transcript.path.read_text(encoding="utf-8").strip().split("\n")
    return [json.loads(line) for line in lines]
