"""Unit tests for PersistentMemory — no API key required."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Load module directly to avoid agent/__init__.py pulling in langchain_core
_BASE = Path(__file__).parent.parent
_spec = importlib.util.spec_from_file_location(
    "streamlit_app.agent.memory", _BASE / "streamlit_app" / "agent" / "memory.py"
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["streamlit_app.agent.memory"] = _mod
_spec.loader.exec_module(_mod)
PersistentMemory = _mod.PersistentMemory


class TestPersistentMemorySaveLoad:
    """Save and load facts."""

    def test_save_and_load_fact(self, tmp_path):
        memory = PersistentMemory("test-session", memory_dir=tmp_path)
        memory.save_fact("The most common species is Acer", category="result")

        facts = memory.load_facts()
        assert len(facts) == 1
        assert "Acer" in facts[0]["fact"]
        assert facts[0]["category"] == "result"

    def test_multiple_facts_ordered(self, tmp_path):
        memory = PersistentMemory("test-session", memory_dir=tmp_path)
        memory.save_fact("Fact one", category="result")
        memory.save_fact("Fact two", category="user_preference")
        memory.save_fact("Fact three", category="dataset")

        facts = memory.load_facts()
        assert len(facts) == 3
        # load_facts returns DESC order (newest first)
        assert facts[0]["fact"] == "Fact three"
        assert facts[2]["fact"] == "Fact one"

    def test_load_facts_respects_limit(self, tmp_path):
        memory = PersistentMemory("test-session", memory_dir=tmp_path)
        for i in range(10):
            memory.save_fact(f"Fact {i}", category="result")

        facts = memory.load_facts(limit=3)
        assert len(facts) == 3


class TestPersistentMemoryPrompt:
    """Prompt section generation."""

    def test_empty_memory_returns_empty_string(self, tmp_path):
        memory = PersistentMemory("test-session", memory_dir=tmp_path)
        assert memory.get_memory_for_prompt() == ""

    def test_memory_for_prompt_with_facts(self, tmp_path):
        memory = PersistentMemory("test-session", memory_dir=tmp_path)
        memory.save_fact("Vienna has 33612 trees", category="result")

        prompt = memory.get_memory_for_prompt()
        assert "## Memory" in prompt
        assert "33612" in prompt
        assert "[result]" in prompt

    def test_max_chars_limit(self, tmp_path):
        memory = PersistentMemory("test-session", memory_dir=tmp_path)
        for i in range(50):
            memory.save_fact(f"This is fact number {i} with enough text to fill space", category="result")

        prompt = memory.get_memory_for_prompt(max_chars=200)
        assert len(prompt) <= 200


class TestPersistentMemoryCrossSession:
    """Cross-session fact loading."""

    def test_cross_session_load(self, tmp_path):
        m1 = PersistentMemory("session-1", memory_dir=tmp_path)
        m1.save_fact("Session 1 fact", category="result")

        m2 = PersistentMemory("session-2", memory_dir=tmp_path)
        m2.save_fact("Session 2 fact", category="result")

        all_facts = m2.load_all_facts()
        assert len(all_facts) == 2
        sessions = {f["session_id"] for f in all_facts}
        assert sessions == {"session-1", "session-2"}

    def test_session_isolation(self, tmp_path):
        m1 = PersistentMemory("session-1", memory_dir=tmp_path)
        m1.save_fact("Only in session 1", category="result")

        m2 = PersistentMemory("session-2", memory_dir=tmp_path)
        facts = m2.load_facts()
        assert len(facts) == 0


class TestPersistentMemoryFiles:
    """File creation and persistence."""

    def test_markdown_file_created(self, tmp_path):
        memory = PersistentMemory("test-session", memory_dir=tmp_path)
        memory.save_fact("A fact", category="general")

        md_path = tmp_path / "test-session_memory.md"
        assert md_path.exists()
        content = md_path.read_text(encoding="utf-8")
        assert "[general] A fact" in content

    def test_sqlite_db_created(self, tmp_path):
        memory = PersistentMemory("test-session", memory_dir=tmp_path)
        db_path = tmp_path / "memory.db"
        assert db_path.exists()
