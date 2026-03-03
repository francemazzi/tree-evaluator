"""PromptAssembler — dynamically assembles system prompt from modular Markdown files.

Inspired by OpenClaw's skill-based prompt injection pattern:
- SOUL.md defines agent identity
- skills/*.md inject tool-specific guidance
- rules/*.md inject behavioral rules
- Each file uses <!-- lang:it/en --> markers for bilingual content in a single file.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, Optional

from streamlit_app.constants import WOOD_DENSITIES

_PROMPTS_DIR = Path(__file__).parent

# Regex to extract language-specific sections from bilingual Markdown files.
# Matches content between <!-- lang:XX --> and the next <!-- lang: or <!-- /lang or EOF.
_LANG_PATTERN = re.compile(
    r"<!--\s*lang:(\w+)\s*-->(.*?)(?=<!--\s*(?:lang:|/lang)|\Z)",
    re.DOTALL,
)


class PromptAssembler:
    """Dynamically assembles system prompt from modular Markdown files."""

    MAX_PROMPT_CHARS: int = 150_000  # OpenClaw-inspired size cap

    def __init__(self, prompts_dir: Optional[Path] = None) -> None:
        self._dir = prompts_dir or _PROMPTS_DIR

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assemble(
        self,
        language: Literal["it", "en"] = "it",
        mode: Literal["full", "minimal", "none"] = "full",
    ) -> str:
        """Assemble the complete system prompt.

        Args:
            language: Target language for prompt content.
            mode: Prompt mode — "full" (all sections), "minimal" (SOUL only),
                  "none" (empty string).

        Returns:
            Assembled prompt string, capped at MAX_PROMPT_CHARS.
        """
        if mode == "none":
            return ""

        sections: list[str] = []

        # 1. SOUL — always included
        soul = self._load(self._dir / "SOUL.md", language)
        if soul:
            sections.append(soul)

        if mode == "minimal":
            return self._join(sections)

        # 2. Skills
        skills_dir = self._dir / "skills"
        if skills_dir.is_dir():
            for md in sorted(skills_dir.glob("*.md")):
                content = self._load(md, language)
                if content:
                    sections.append(content)

        # 3. Rules
        rules_dir = self._dir / "rules"
        if rules_dir.is_dir():
            for md in sorted(rules_dir.glob("*.md")):
                content = self._load(md, language)
                if content:
                    sections.append(content)

        # 4. Dynamic constants (wood densities table)
        sections.append(self._build_wood_densities_section(language))

        return self._join(sections)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load(path: Path, language: str) -> str:
        """Load a Markdown file and extract the section for *language*.

        If no language markers are found, the entire file content is returned.
        """
        if not path.exists():
            return ""

        raw = path.read_text(encoding="utf-8")

        # Try to extract language-specific section
        matches = {m.group(1): m.group(2).strip() for m in _LANG_PATTERN.finditer(raw)}

        if matches:
            return matches.get(language, matches.get("en", ""))

        # No markers — return raw content (language-agnostic file)
        return raw.strip()

    @staticmethod
    def _build_wood_densities_section(language: str) -> str:
        """Generate the wood densities table from constants.py."""
        if language == "it":
            header = "Densita' del legno comuni (g/cm^3):"
        else:
            header = "Common wood densities (g/cm^3):"

        lines = [header]
        for genus, density in WOOD_DENSITIES.items():
            lines.append(f"- {genus}: {density}")
        return "\n".join(lines)

    def _join(self, sections: list[str]) -> str:
        """Join sections with double newlines, enforcing MAX_PROMPT_CHARS."""
        result = "\n\n".join(s for s in sections if s)
        if len(result) > self.MAX_PROMPT_CHARS:
            result = result[: self.MAX_PROMPT_CHARS]
        return result
