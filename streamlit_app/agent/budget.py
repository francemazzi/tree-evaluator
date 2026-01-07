"""Budget management for agent execution to prevent infinite loops and runaway costs.

This module provides budget tracking and enforcement mechanisms to ensure
the agent operates within defined resource limits.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Sequence, Tuple

from langchain_core.messages import AIMessage, BaseMessage


@dataclass
class AgentBudget:
    """Budget constraints for agent execution to prevent infinite loops and runaway costs.
    
    This class tracks resource usage during agent execution and enforces limits on:
    - Total tool calls across all tools
    - Calls per individual tool
    - LLM invocations
    - Execution time
    - Replan attempts
    """
    
    # Budget limits (configurable)
    max_total_tool_calls: int = 20          # Hard limit on total tool calls
    max_calls_per_tool: int = 5             # Max calls to same tool (increased for viz tools)
    max_llm_calls: int = 15                 # Max LLM invocations
    max_execution_time_seconds: int = 120   # Timeout in seconds
    max_replans: int = 2                    # Max replan attempts
    
    # Runtime counters
    tool_calls: Dict[str, int] = field(default_factory=dict)
    total_tool_calls: int = 0
    llm_calls: int = 0
    start_time: float = field(default_factory=time.time)
    replans: int = 0
    
    def can_call_tool(self, tool_name: str) -> Tuple[bool, str]:
        """Check if we can call a specific tool.
        
        Args:
            tool_name: Name of the tool to check.
            
        Returns:
            Tuple of (can_call, reason_if_blocked).
        """
        # Check timeout
        elapsed = time.time() - self.start_time
        if elapsed > self.max_execution_time_seconds:
            return False, f"Timeout: {elapsed:.1f}s exceeded limit of {self.max_execution_time_seconds}s"
        
        # Check total tool calls
        if self.total_tool_calls >= self.max_total_tool_calls:
            return False, f"Total tool call limit reached: {self.total_tool_calls}/{self.max_total_tool_calls}"
        
        # Check per-tool limit
        current_count = self.tool_calls.get(tool_name, 0)
        if current_count >= self.max_calls_per_tool:
            return False, f"Tool '{tool_name}' limit reached: {current_count}/{self.max_calls_per_tool} calls"
        
        return True, ""
    
    def record_tool_call(self, tool_name: str) -> None:
        """Record a tool call."""
        self.tool_calls[tool_name] = self.tool_calls.get(tool_name, 0) + 1
        self.total_tool_calls += 1
    
    def can_call_llm(self) -> Tuple[bool, str]:
        """Check if we can make another LLM call."""
        if self.llm_calls >= self.max_llm_calls:
            return False, f"LLM call limit reached: {self.llm_calls}/{self.max_llm_calls}"
        return True, ""
    
    def record_llm_call(self) -> None:
        """Record an LLM call."""
        self.llm_calls += 1
    
    def can_replan(self) -> Tuple[bool, str]:
        """Check if we can attempt another replan."""
        if self.replans >= self.max_replans:
            return False, f"Replan limit reached: {self.replans}/{self.max_replans}"
        return True, ""
    
    def record_replan(self) -> None:
        """Record a replan attempt."""
        self.replans += 1
    
    def get_status(self) -> Dict[str, Any]:
        """Get current budget status for debugging/logging."""
        elapsed = time.time() - self.start_time
        return {
            "total_tool_calls": f"{self.total_tool_calls}/{self.max_total_tool_calls}",
            "llm_calls": f"{self.llm_calls}/{self.max_llm_calls}",
            "replans": f"{self.replans}/{self.max_replans}",
            "elapsed_time": f"{elapsed:.1f}s/{self.max_execution_time_seconds}s",
            "per_tool_calls": dict(self.tool_calls),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize budget to dict for state storage."""
        return {
            "max_total_tool_calls": self.max_total_tool_calls,
            "max_calls_per_tool": self.max_calls_per_tool,
            "max_llm_calls": self.max_llm_calls,
            "max_execution_time_seconds": self.max_execution_time_seconds,
            "max_replans": self.max_replans,
            "tool_calls": dict(self.tool_calls),
            "total_tool_calls": self.total_tool_calls,
            "llm_calls": self.llm_calls,
            "start_time": self.start_time,
            "replans": self.replans,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentBudget":
        """Deserialize budget from dict."""
        budget = cls(
            max_total_tool_calls=data.get("max_total_tool_calls", 20),
            max_calls_per_tool=data.get("max_calls_per_tool", 5),
            max_llm_calls=data.get("max_llm_calls", 15),
            max_execution_time_seconds=data.get("max_execution_time_seconds", 120),
            max_replans=data.get("max_replans", 2),
        )
        budget.tool_calls = dict(data.get("tool_calls", {}))
        budget.total_tool_calls = data.get("total_tool_calls", 0)
        budget.llm_calls = data.get("llm_calls", 0)
        budget.start_time = data.get("start_time", time.time())
        budget.replans = data.get("replans", 0)
        return budget


class BudgetAwareToolGuard:
    """Circuit breaker that enforces budget limits before tool execution.
    
    This guard checks budget constraints before each tool call and prevents
    execution if limits are exceeded, returning a user-friendly error message.
    """
    
    def __init__(self, budget: AgentBudget):
        self._budget = budget
    
    def check_before_tools(self, messages: Sequence[BaseMessage]) -> Tuple[bool, str, Dict[str, Any]]:
        """Check budget before tool execution.
        
        Args:
            messages: Current conversation messages.
            
        Returns:
            Tuple of (can_proceed, error_message, budget_status).
        """
        # Extract pending tool calls from last AI message
        last_ai = None
        for msg in reversed(list(messages)):
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                last_ai = msg
                break
        
        if not last_ai or not last_ai.tool_calls:
            return True, "", self._budget.get_status()
        
        # Check each tool call against budget
        for tc in last_ai.tool_calls:
            tool_name = tc.get("name", "unknown")
            can_call, reason = self._budget.can_call_tool(tool_name)
            
            if not can_call:
                error_msg = self._build_budget_error_response(reason, tool_name)
                return False, error_msg, self._budget.get_status()
            
            # Record the call (pre-emptively)
            self._budget.record_tool_call(tool_name)
        
        return True, "", self._budget.get_status()
    
    def check_before_llm(self) -> Tuple[bool, str]:
        """Check budget before LLM call."""
        can_call, reason = self._budget.can_call_llm()
        if can_call:
            self._budget.record_llm_call()
        return can_call, reason
    
    def _build_budget_error_response(self, reason: str, tool_name: str) -> str:
        """Build user-friendly error message when budget is exceeded."""
        status = self._budget.get_status()
        tools_used = ', '.join(status['per_tool_calls'].keys()) or 'Nessuno'
        
        return f"""⚠️ **Limite di esecuzione raggiunto**

{reason}

**Stato attuale:**
- Tool calls totali: {status['total_tool_calls']}
- Chiamate LLM: {status['llm_calls']}
- Tempo trascorso: {status['elapsed_time']}
- Tool più usato: {tool_name}

**Suggerimenti:**
- Riformula la domanda in modo più specifico
- Suddividi la richiesta in domande più semplici
- Chiedi direttamente il risultato senza elaborazioni complesse

Tool utilizzati: {tools_used}"""

