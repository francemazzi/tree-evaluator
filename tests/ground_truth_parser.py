from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import List, Optional

from tests.ground_truth_models import LLMParsedResponse


class ResponseParser:
    """Extracts structured data from raw LLM responses."""

    NUMBER_PATTERN = re.compile(r"\d[\d\.,]*")

    def parse(self, response: str, expected_number: Optional[float]) -> LLMParsedResponse:
        numeric_value = self._extract_number(response, expected_number)
        normalized_text = self._normalize_text(response)
        return LLMParsedResponse(
            raw_text=response.strip(),
            extracted_number=numeric_value,
            normalized_text=normalized_text,
        )

    def _extract_number(self, response: str, expected_number: Optional[float]) -> Optional[float]:
        candidates: List[float] = []
        for match in self.NUMBER_PATTERN.findall(response):
            parsed = self._to_float(match)
            if parsed is not None:
                candidates.append(parsed)

        if not candidates:
            return None

        if expected_number is None:
            return candidates[0]

        return min(candidates, key=lambda value: abs(value - expected_number))

    @staticmethod
    def _to_float(value: str) -> Optional[float]:
        cleaned = value.strip()
        cleaned = cleaned.replace("\u00a0", " ")
        cleaned = cleaned.replace(" ", "")
        cleaned = cleaned.replace(".", "").replace(",", ".")
        if not cleaned:
            return None

        try:
            return float(cleaned)
        except ValueError:
            return None

    def normalize_reference(self, value: str) -> str:
        return self._normalize_text(value)

    @staticmethod
    def _normalize_text(value: str) -> str:
        normalized = value.strip().lower()
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized


class NumericAnswerMatcher:
    """Checks whether an expected numeric answer is textually contained within a response."""

    _SEPARATOR_CLASS = r"[.,\s\u00a0\u202f']"

    def __init__(self, expected_value: float) -> None:
        self._expected_value = expected_value
        self._pattern = self._compile_pattern(expected_value)

    def matches(self, text: str) -> bool:
        if self._pattern is None or not text:
            return False
        return bool(self._pattern.search(text))

    @classmethod
    def _compile_pattern(cls, value: float) -> Optional[re.Pattern]:
        try:
            decimal_value = Decimal(str(value))
        except (InvalidOperation, ValueError):
            return None

        sign = "-" if decimal_value.is_signed() else ""
        magnitude = decimal_value.copy_abs()
        digits_tuple = magnitude.as_tuple()

        digits = "".join(str(digit) for digit in digits_tuple.digits) or "0"
        exponent = digits_tuple.exponent

        if exponent >= 0:
            digits = digits + ("0" * exponent)
            integer_digits = digits or "0"
            fractional_digits = ""
        else:
            decimal_places = -exponent
            if len(digits) <= decimal_places:
                digits = digits.zfill(decimal_places + 1)
            integer_digits = digits[:-decimal_places] or "0"
            fractional_digits = digits[-decimal_places:]

        if fractional_digits and set(fractional_digits) == {"0"}:
            fractional_digits = ""

        integer_pattern = cls._build_digit_pattern(integer_digits, allow_group_separators=True)
        pattern_parts: List[str] = []

        if sign:
            pattern_parts.append(r"[-\u2212]?")

        pattern_parts.append(integer_pattern)

        if fractional_digits:
            fractional_pattern = cls._build_digit_pattern(fractional_digits, allow_group_separators=False)
            pattern_parts.append(r"(?:[.,]" + fractional_pattern + r")")

        full_pattern = "".join(pattern_parts)
        return re.compile(r"(?<!\d)" + full_pattern + r"(?!\d)")

    @classmethod
    def _build_digit_pattern(cls, digits: str, allow_group_separators: bool) -> str:
        if not digits:
            return ""
        if digits == "0":
            return "0"

        pattern_parts: List[str] = []
        for index, digit in enumerate(digits):
            pattern_parts.append(digit)
            if allow_group_separators and index != len(digits) - 1:
                pattern_parts.append(cls._SEPARATOR_CLASS + "?")

        return "".join(pattern_parts)
