"""Unit tests for FailureTracker — no API key required."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Load modules directly to avoid agent/__init__.py pulling in langchain_core
_BASE = Path(__file__).parent.parent


def _load_module(name: str, filepath: Path):
    """Load a Python module from file path, bypassing package __init__."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Pre-load registry (dependency of failure_tracker)
_reg = _load_module("streamlit_app.tools.registry", _BASE / "streamlit_app" / "tools" / "registry.py")
_ft = _load_module("streamlit_app.agent.failure_tracker", _BASE / "streamlit_app" / "agent" / "failure_tracker.py")

FailureTracker = _ft.FailureTracker
LoopDecision = _ft.LoopDecision


class TestFailureTrackerBasics:
    """Basic call tracking and counting."""

    def test_first_call_continues(self):
        tracker = FailureTracker()
        decision = tracker.record_and_check("query_tree_dataset", {"query": "top species"})
        assert decision.action == "continue"

    def test_get_counts_after_calls(self):
        tracker = FailureTracker()
        tracker.record_and_check("query_tree_dataset", {"query": "a"})
        tracker.record_and_check("generate_chart", {"chart_type": "bar"})
        tracker.record_and_check("query_tree_dataset", {"query": "b"})

        counts = tracker.get_counts()
        assert counts["query_tree_dataset"] == 2
        assert counts["generate_chart"] == 1
        assert tracker.get_total_calls() == 3

    def test_get_tool_count_unknown_tool(self):
        tracker = FailureTracker()
        assert tracker.get_tool_count("nonexistent") == 0


class TestIdenticalCallDetection:
    """MAX_IDENTICAL_CALLS = 2: stop after 2 identical (tool+args) calls."""

    def test_identical_call_stops_after_threshold(self):
        tracker = FailureTracker()
        args = {"query": "how many trees"}

        d1 = tracker.record_and_check("query_tree_dataset", args)
        assert d1.action == "continue"

        d2 = tracker.record_and_check("query_tree_dataset", args)
        assert d2.action == "stop"
        assert d2.reason == "identical_call_repeated"
        assert d2.call_count == 2

    def test_different_args_do_not_trigger_identical_stop(self):
        tracker = FailureTracker()

        d1 = tracker.record_and_check("query_tree_dataset", {"query": "top species"})
        assert d1.action == "continue"

        d2 = tracker.record_and_check("query_tree_dataset", {"query": "total trees"})
        assert d2.action == "continue"

    def test_different_tools_same_args_do_not_trigger(self):
        tracker = FailureTracker()
        args = {"query": "test"}

        tracker.record_and_check("query_tree_dataset", args)
        d2 = tracker.record_and_check("query_species_list", args)
        assert d2.action == "continue"


class TestPerToolLimits:
    """Per-tool threshold from ToolRegistry triggers replan then stop."""

    def test_per_tool_limit_triggers_replan(self):
        tracker = FailureTracker()
        # generate_chart has max_calls_per_session=3 in registry
        for i in range(3):
            d = tracker.record_and_check("generate_chart", {"chart_type": "bar", "i": i})

        # 3rd call should trigger replan (max_calls=3)
        assert d.action == "replan"
        assert d.reason == "tool_limit_reached"

    def test_per_tool_limit_plus2_triggers_stop(self):
        tracker = FailureTracker()
        # generate_chart: max_calls=3, stop at 3+2=5
        for i in range(5):
            d = tracker.record_and_check("generate_chart", {"chart_type": "bar", "i": i})

        assert d.action == "stop"
        assert d.reason == "tool_limit_exceeded"


class TestPatternDetection:
    """Sliding-window pattern detection (A-B-A-B)."""

    def test_abab_pattern_detected_on_fourth_call(self):
        tracker = FailureTracker()
        tracker.record_and_check("query_tree_dataset", {"q": "1"})
        tracker.record_and_check("generate_chart", {"t": "1"})
        tracker.record_and_check("query_tree_dataset", {"q": "2"})
        d = tracker.record_and_check("generate_chart", {"t": "2"})

        assert d.action == "replan"
        assert d.reason == "repeating_pattern"

    def test_no_pattern_with_varied_tools(self):
        tracker = FailureTracker()
        tracker.record_and_check("query_tree_dataset", {"q": "1"})
        tracker.record_and_check("generate_chart", {"t": "1"})
        tracker.record_and_check("calculate_co2_sequestration", {"species": "Acer"})
        d = tracker.record_and_check("query_tree_dataset", {"q": "2"})

        assert d.action == "continue"
