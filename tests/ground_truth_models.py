from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class GroundTruthRecord:
    """Represents a single ground truth entry."""

    identifier: str
    question: str
    numeric_answer: Optional[float]
    text_answer: Optional[str]
    category: Optional[str]

    def has_numeric_answer(self) -> bool:
        """Return True when the record provides a numeric answer."""
        return self.numeric_answer is not None

    def has_text_answer(self) -> bool:
        """Return True when the record provides a text answer."""
        return bool(self.text_answer and self.text_answer.strip())


@dataclass
class LLMParsedResponse:
    """Container for parsed LLM outputs."""

    raw_text: str
    extracted_number: Optional[float]
    normalized_text: str


@dataclass
class EvaluationResult:
    """Maintains evaluation outcomes for a single record."""

    record: GroundTruthRecord
    response: LLMParsedResponse
    numeric_match: Optional[bool]
    numeric_error: Optional[float]
    text_similarity: Optional[float]
    error: Optional[str]
