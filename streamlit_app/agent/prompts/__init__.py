"""Modular prompt system with skill-based architecture."""

from __future__ import annotations

from typing import Literal

from streamlit_app.agent.prompts.assembler import PromptAssembler

# Singleton assembler instance
_assembler = PromptAssembler()


class SystemPrompts:
    """Backward-compatible wrapper that delegates to PromptAssembler.

    Legacy code can continue to call SystemPrompts.get_system_prompt().
    New code should use PromptAssembler directly.
    """

    @staticmethod
    def get_system_prompt(language: Literal["it", "en"] = "it") -> str:
        """Get the system prompt in the specified language."""
        return _assembler.assemble(language=language, mode="full")

    # Legacy class attribute — used by some old references
    MAIN_SYSTEM_PROMPT = _assembler.assemble(language="it", mode="full")


__all__ = ["PromptAssembler", "SystemPrompts"]
