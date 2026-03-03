"""Unified tool-call tracking and loop detection.

Replaces three overlapping systems (ToolLoopManager, ToolLoopGuard,
SemanticToolLoopDetector) with a single, clean implementation inspired
by OpenClaw's failure tracking pattern:
  - Track (tool_name, args_hash) tuples
  - MAX_IDENTICAL_CALLS = 2  →  stop after 2 identical calls
  - Per-tool thresholds from ToolRegistry
  - Sliding-window pattern detection
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from streamlit_app.tools.registry import ToolRegistry


@dataclass(frozen=True)
class ToolCallSignature:
    """Normalized representation of a tool call for deduplication."""

    tool_name: str
    args_hash: str


@dataclass
class LoopDecision:
    """Decision from the failure tracker."""

    action: Literal["continue", "replan", "stop"]
    reason: str = ""
    tool_name: str = ""
    call_count: int = 0


class FailureTracker:
    """Unified tool-call tracking and loop detection.

    Usage::

        tracker = FailureTracker()
        decision = tracker.record_and_check("query_tree_dataset", {"query": "top species"})
        if decision.action == "stop":
            ...
    """

    MAX_IDENTICAL_CALLS: int = 2   # Stop after 2 identical (tool+args) calls
    WINDOW_SIZE: int = 10          # Sliding window for pattern detection

    def __init__(self) -> None:
        self._history: List[ToolCallSignature] = []
        self._tool_counts: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_and_check(self, tool_name: str, args: Dict[str, Any]) -> LoopDecision:
        """Record a tool call and check for loops.

        Args:
            tool_name: Name of the tool being called.
            args: Arguments passed to the tool.

        Returns:
            LoopDecision indicating what the agent should do.
        """
        sig = ToolCallSignature(
            tool_name=tool_name,
            args_hash=self._hash_args(args),
        )
        self._history.append(sig)
        self._tool_counts[tool_name] = self._tool_counts.get(tool_name, 0) + 1

        tool_count = self._tool_counts[tool_name]
        max_calls = ToolRegistry.get_max_calls(tool_name)

        # Check 1: Identical call repeated (same tool + same args)
        identical_count = sum(1 for s in self._history if s == sig)
        if identical_count >= self.MAX_IDENTICAL_CALLS:
            return LoopDecision(
                action="stop",
                reason="identical_call_repeated",
                tool_name=tool_name,
                call_count=identical_count,
            )

        # Check 2: Per-tool threshold exceeded significantly → stop
        if tool_count >= max_calls + 2:
            return LoopDecision(
                action="stop",
                reason="tool_limit_exceeded",
                tool_name=tool_name,
                call_count=tool_count,
            )

        # Check 3: Per-tool threshold reached → replan
        if tool_count >= max_calls:
            return LoopDecision(
                action="replan",
                reason="tool_limit_reached",
                tool_name=tool_name,
                call_count=tool_count,
            )

        # Check 4: Sliding-window pattern detection
        if self._detect_pattern():
            return LoopDecision(
                action="replan",
                reason="repeating_pattern",
                tool_name=tool_name,
                call_count=tool_count,
            )

        return LoopDecision(action="continue")

    def get_tool_count(self, tool_name: str) -> int:
        """Get how many times a tool has been called."""
        return self._tool_counts.get(tool_name, 0)

    def get_total_calls(self) -> int:
        """Get total number of tool calls tracked."""
        return len(self._history)

    def get_counts(self) -> Dict[str, int]:
        """Get per-tool call counts."""
        return dict(self._tool_counts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_args(args: Dict[str, Any]) -> str:
        """Create a stable hash of tool arguments."""
        try:
            canonical = json.dumps(args, sort_keys=True, default=str)
        except (TypeError, ValueError):
            canonical = str(args)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def _detect_pattern(self) -> bool:
        """Detect repeating patterns in the sliding window.

        Looks for A-B-A-B or A-A-B-A-A-B style repetitions.
        """
        window = self._history[-self.WINDOW_SIZE:]
        if len(window) < 4:
            return False

        # Check for 2-element repeating pattern (A-B-A-B)
        names = [s.tool_name for s in window]
        for pattern_len in (2, 3):
            if len(names) >= pattern_len * 2:
                pattern = names[-pattern_len:]
                preceding = names[-(pattern_len * 2):-pattern_len]
                if pattern == preceding:
                    return True

        return False
