"""Custom exceptions for the Tree Evaluator Agent.

This module defines a hierarchy of exceptions for structured error handling
throughout the agent system, replacing string-based error detection.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class AgentException(Exception):
    """Base exception for all agent-related errors.

    Attributes:
        message: Human-readable error message.
        details: Optional dictionary with additional context.
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class RateLimitException(AgentException):
    """Raised when an API rate limit is hit.

    This exception should be caught and handled by falling back to a lighter model
    or implementing exponential backoff.
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        provider: str = "unknown",
        retry_after: Optional[int] = None,
    ):
        details = {"provider": provider}
        if retry_after:
            details["retry_after_seconds"] = retry_after
        super().__init__(message, details)
        self.provider = provider
        self.retry_after = retry_after


class BudgetExceededException(AgentException):
    """Raised when agent execution budget is exceeded.

    This includes limits on tool calls, LLM invocations, and execution time.
    """

    def __init__(
        self,
        message: str = "Budget exceeded",
        budget_type: str = "unknown",
        current_value: int = 0,
        limit_value: int = 0,
    ):
        details = {
            "budget_type": budget_type,
            "current": current_value,
            "limit": limit_value,
        }
        super().__init__(message, details)
        self.budget_type = budget_type
        self.current_value = current_value
        self.limit_value = limit_value


class ToolLoopException(AgentException):
    """Raised when a tool call loop is detected.

    This occurs when the agent repeatedly calls the same tool with similar
    arguments without making progress.
    """

    def __init__(
        self,
        message: str = "Tool loop detected",
        tool_name: str = "unknown",
        call_count: int = 0,
        fingerprint: Optional[str] = None,
    ):
        details = {
            "tool_name": tool_name,
            "call_count": call_count,
        }
        if fingerprint:
            details["fingerprint"] = fingerprint
        super().__init__(message, details)
        self.tool_name = tool_name
        self.call_count = call_count
        self.fingerprint = fingerprint


class ToolExecutionException(AgentException):
    """Raised when a tool fails to execute properly.

    This wraps errors from individual tool implementations.
    """

    def __init__(
        self,
        message: str = "Tool execution failed",
        tool_name: str = "unknown",
        original_error: Optional[Exception] = None,
    ):
        details = {"tool_name": tool_name}
        if original_error:
            details["original_error"] = str(original_error)
            details["error_type"] = type(original_error).__name__
        super().__init__(message, details)
        self.tool_name = tool_name
        self.original_error = original_error


class QueryOptimizationException(AgentException):
    """Raised when query optimization fails.

    This is a recoverable error - the agent should fall back to simple planning.
    """

    def __init__(
        self,
        message: str = "Query optimization failed",
        original_query: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        details = {}
        if original_query:
            details["query"] = original_query[:200]  # Truncate for safety
        if original_error:
            details["original_error"] = str(original_error)
        super().__init__(message, details)
        self.original_query = original_query
        self.original_error = original_error


class FormulaExecutionException(AgentException):
    """Raised when a dynamic formula fails to execute.

    This occurs in the DynamicToolLoader when formula parsing or execution fails.
    """

    def __init__(
        self,
        message: str = "Formula execution failed",
        formula: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        details = {}
        if formula:
            details["formula"] = formula
        if inputs:
            details["inputs"] = inputs
        if original_error:
            details["original_error"] = str(original_error)
        super().__init__(message, details)
        self.formula = formula
        self.inputs = inputs
        self.original_error = original_error


class ContextOverflowException(AgentException):
    """Raised when conversation context exceeds token limits.

    This should trigger context compression or message pruning.
    """

    def __init__(
        self,
        message: str = "Context overflow",
        message_count: int = 0,
        estimated_tokens: Optional[int] = None,
    ):
        details = {"message_count": message_count}
        if estimated_tokens:
            details["estimated_tokens"] = estimated_tokens
        super().__init__(message, details)
        self.message_count = message_count
        self.estimated_tokens = estimated_tokens
