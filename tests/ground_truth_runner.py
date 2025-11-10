from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from pathlib import Path
from statistics import median
from typing import Iterable, List, Optional, Sequence

from dotenv import load_dotenv


# Ensure project root is available in sys.path when executed as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_app.agent import TreeEvaluatorAgent  # noqa: E402  pylint: disable=wrong-import-position


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


class GroundTruthDataset:
    """Loads and exposes ground truth records from the CSV dataset."""

    def __init__(self, csv_path: Path) -> None:
        self._csv_path = csv_path
        self._records: List[GroundTruthRecord] = []
        self._load()

    def _load(self) -> None:
        if not self._csv_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {self._csv_path}")

        with self._csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter=";")
            for row in reader:
                question = (row.get("domanda") or "").strip()
                if not question:
                    continue

                identifier = (row.get("id") or str(len(self._records) + 1)).strip()
                numeric = self._parse_numeric(row.get("risposta numerica"))
                text = self._clean_text(row.get("risposta estesa"))
                category = self._clean_text(row.get("type"))

                record = GroundTruthRecord(
                    identifier=identifier,
                    question=question,
                    numeric_answer=numeric,
                    text_answer=text,
                    category=category,
                )
                self._records.append(record)

    def records(self) -> Sequence[GroundTruthRecord]:
        """Return the parsed ground truth records."""

        return tuple(self._records)

    def __iter__(self) -> Iterable[GroundTruthRecord]:
        return iter(self._records)

    @staticmethod
    def _parse_numeric(value: Optional[str]) -> Optional[float]:
        if not value:
            return None

        cleaned = value.strip().replace("\u00a0", " ")
        if not cleaned:
            return None

        normalized = cleaned.replace(".", "").replace(",", ".")
        normalized = normalized.replace(" ", "")

        try:
            return float(normalized)
        except ValueError:
            return None

    @staticmethod
    def _clean_text(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            return None
        return cleaned.strip("\"“”")


@dataclass
class LLMParsedResponse:
    """Container for parsed LLM outputs."""

    raw_text: str
    extracted_number: Optional[float]
    normalized_text: str


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


@dataclass
class EvaluationResult:
    """Maintains evaluation outcomes for a single record."""

    record: GroundTruthRecord
    response: LLMParsedResponse
    numeric_match: Optional[bool]
    numeric_error: Optional[float]
    text_similarity: Optional[float]
    error: Optional[str]


class AccuracyReport:
    """Aggregates evaluation results and exposes accuracy metrics."""

    def __init__(self, results: Sequence[EvaluationResult], text_threshold: float = 0.65) -> None:
        self._results = list(results)
        self._text_threshold = text_threshold

    @property
    def numeric_accuracy(self) -> Optional[float]:
        counts = self.numeric_counts
        if counts is None:
            return None
        total, correct = counts
        return correct / total if total else None

    @property
    def average_text_similarity(self) -> Optional[float]:
        similarities = [r.text_similarity for r in self._results if r.text_similarity is not None]
        if not similarities:
            return None
        return sum(similarities) / len(similarities)

    @property
    def median_text_similarity(self) -> Optional[float]:
        similarities = [r.text_similarity for r in self._results if r.text_similarity is not None]
        if not similarities:
            return None
        return median(similarities)

    @property
    def numeric_counts(self) -> Optional[tuple[int, int]]:
        numeric_results = [
            r
            for r in self._results
            if r.record.has_numeric_answer() and r.numeric_match is not None
        ]
        if not numeric_results:
            return None
        correct = sum(1 for result in numeric_results if result.numeric_match)
        return len(numeric_results), correct

    @property
    def text_counts(self) -> Optional[tuple[int, int]]:
        text_results = [
            r
            for r in self._results
            if r.record.has_text_answer() and r.text_similarity is not None
        ]
        if not text_results:
            return None
        passes = sum(1 for result in text_results if result.text_similarity >= self._text_threshold)
        return len(text_results), passes

    @property
    def full_pass_counts(self) -> Optional[tuple[int, int]]:
        if not self._results:
            return None
        passed = 0
        for result in self._results:
            if result.error:
                continue

            numeric_ok = True
            if result.record.has_numeric_answer():
                numeric_ok = result.numeric_match is True

            text_ok = True
            if result.record.has_text_answer() and result.text_similarity is not None:
                text_ok = result.text_similarity >= self._text_threshold

            if result.record.has_text_answer() and result.text_similarity is None:
                text_ok = False

            if numeric_ok and text_ok:
                passed += 1
        return passed, len(self._results)

    @property
    def full_pass_rate(self) -> Optional[float]:
        counts = self.full_pass_counts
        if counts is None:
            return None
        passed, total = counts
        return passed / total if total else None

    @property
    def total_records(self) -> int:
        return len(self._results)

    def failing_records(self) -> List[EvaluationResult]:
        failures: List[EvaluationResult] = []
        for result in self._results:
            if result.error:
                failures.append(result)
                continue
            if result.record.has_numeric_answer() and result.numeric_match is False:
                failures.append(result)
                continue
            if (
                result.record.has_text_answer()
                and result.text_similarity is not None
                and result.text_similarity < self._text_threshold
            ):
                failures.append(result)
        return failures

    def render(self, verbose: bool = False) -> str:
        lines: List[str] = []
        lines.append("=== Ground Truth Accuracy Report ===")
        lines.append(f"Records evaluated: {self.total_records}")

        numeric_counts = self.numeric_counts
        numeric_accuracy = self.numeric_accuracy
        if numeric_counts and numeric_accuracy is not None:
            total_numeric, correct_numeric = numeric_counts
            lines.append(
                f"Numeric accuracy: {correct_numeric}/{total_numeric} ({numeric_accuracy * 100:.1f}%)"
            )
        else:
            lines.append("Numeric accuracy: not available")

        text_counts = self.text_counts
        if text_counts:
            total_text, text_passes = text_counts
            text_pass_rate = text_passes / total_text if total_text else 0.0
            lines.append(
                f"Text pass rate (≥{self._text_threshold * 100:.0f}%): {text_passes}/{total_text} ({text_pass_rate * 100:.1f}%)"
            )
        else:
            lines.append("Text pass rate: not available")

        text_similarity = self.average_text_similarity
        if text_similarity is not None:
            lines.append(f"Average text similarity: {text_similarity * 100:.1f}%")
        else:
            lines.append("Average text similarity: not available")

        median_similarity = self.median_text_similarity
        if median_similarity is not None:
            lines.append(f"Median text similarity: {median_similarity * 100:.1f}%")

        full_pass_counts = self.full_pass_counts
        full_pass_rate = self.full_pass_rate
        if full_pass_counts and full_pass_rate is not None:
            passed, total = full_pass_counts
            lines.append(f"Full pass rate: {passed}/{total} ({full_pass_rate * 100:.1f}%)")

        failures = self.failing_records()
        if failures:
            lines.append("")
            lines.append("Failures:")
            for failure in failures:
                reason_parts: List[str] = []
                if failure.error:
                    reason_parts.append(f"Error: {failure.error}")
                if failure.numeric_match is False:
                    reason_parts.append(
                        f"Numeric mismatch (expected {failure.record.numeric_answer}, got {failure.response.extracted_number})"
                    )
                if (
                    failure.record.has_text_answer()
                    and failure.text_similarity is not None
                    and failure.text_similarity < self._text_threshold
                ):
                    reason_parts.append(f"Low text similarity ({failure.text_similarity:.2f})")

                lines.append(f"- ID {failure.record.identifier}: {'; '.join(reason_parts)}")

        # Verbose mode: show detailed comparison for all failures
        if verbose and failures:
            lines.append("\n" + "=" * 80)
            lines.append("DETAILED OUTLIER ANALYSIS")
            lines.append("=" * 80)
            
            for idx, failure in enumerate(failures, 1):
                lines.append(f"\n[OUTLIER {idx}] ID: {failure.record.identifier}")
                lines.append("-" * 80)
                
                # Question
                lines.append(f"❓ DOMANDA:")
                lines.append(f"   {failure.record.question}")
                lines.append("")
                
                # Expected answers
                if failure.record.has_numeric_answer():
                    lines.append(f"✅ RISPOSTA NUMERICA ATTESA:")
                    lines.append(f"   {failure.record.numeric_answer}")
                    lines.append("")
                
                if failure.record.has_text_answer():
                    lines.append(f"✅ RISPOSTA TESTUALE ATTESA:")
                    lines.append(f"   {failure.record.text_answer}")
                    lines.append("")
                
                # Agent response
                lines.append(f"🤖 RISPOSTA AGENTE:")
                if failure.response.raw_text:
                    # Wrap long responses
                    response_lines = failure.response.raw_text.split('\n')
                    for line in response_lines:
                        if len(line) > 75:
                            # Word wrap
                            words = line.split()
                            current_line = "   "
                            for word in words:
                                if len(current_line) + len(word) + 1 > 78:
                                    lines.append(current_line)
                                    current_line = "   " + word
                                else:
                                    current_line += (" " if current_line != "   " else "") + word
                            if current_line.strip():
                                lines.append(current_line)
                        else:
                            lines.append(f"   {line}")
                else:
                    lines.append("   [Nessuna risposta]")
                lines.append("")
                
                # Metrics
                lines.append(f"📊 METRICHE:")
                if failure.record.has_numeric_answer():
                    lines.append(f"   • Numero estratto: {failure.response.extracted_number}")
                    lines.append(f"   • Match numerico: {'✓ SI' if failure.numeric_match else '✗ NO'}")
                    if failure.numeric_error is not None:
                        lines.append(f"   • Errore assoluto: {failure.numeric_error:.2f}")
                
                if failure.text_similarity is not None:
                    lines.append(f"   • Similarità testuale: {failure.text_similarity:.2%} (soglia: {self._text_threshold:.2%})")
                    if failure.text_similarity < self._text_threshold:
                        lines.append(f"   • Status: ✗ SOTTO SOGLIA")
                    else:
                        lines.append(f"   • Status: ✓ SOPRA SOGLIA")
                
                if failure.error:
                    lines.append(f"   • Errore: {failure.error}")
                
                lines.append("")

        return "\n".join(lines)


class TreeAgentClient:
    """Wrapper around the TreeEvaluatorAgent to facilitate queries."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        load_dotenv()
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY is required to run ground truth evaluation")

        self._agent = TreeEvaluatorAgent(openai_api_key=self._api_key)

    def ask(self, question: str) -> str:
        return self._agent.chat(question)


class GroundTruthTestRunner:
    """Executes the ground truth evaluation workflow."""

    def __init__(
        self,
        csv_path: Path,
        agent_client: Optional[TreeAgentClient] = None,
        numeric_tolerance: float = 0.01,
        text_threshold: float = 0.65,
    ) -> None:
        self._dataset = GroundTruthDataset(csv_path)
        self._agent_client = agent_client or TreeAgentClient()
        self._parser = ResponseParser()
        self._numeric_tolerance = numeric_tolerance
        self._text_threshold = text_threshold

    def run(self, limit: Optional[int] = None) -> AccuracyReport:
        results: List[EvaluationResult] = []

        for index, record in enumerate(self._dataset, start=1):
            if limit is not None and index > limit:
                break

            try:
                response_text = self._agent_client.ask(record.question)
                parsed_response = self._parser.parse(response_text, record.numeric_answer)
                result = self._evaluate(record, parsed_response)
            except Exception as exc:  # pylint: disable=broad-except
                result = EvaluationResult(
                    record=record,
                    response=LLMParsedResponse(raw_text="", extracted_number=None, normalized_text=""),
                    numeric_match=None,
                    numeric_error=None,
                    text_similarity=None,
                    error=str(exc),
                )

            results.append(result)

        return AccuracyReport(results, text_threshold=self._text_threshold)

    def _evaluate(self, record: GroundTruthRecord, response: LLMParsedResponse) -> EvaluationResult:
        numeric_match: Optional[bool] = None
        numeric_error: Optional[float] = None

        if record.has_numeric_answer():
            matcher = NumericAnswerMatcher(float(record.numeric_answer))
            numeric_match = matcher.matches(response.raw_text)
            if response.extracted_number is not None:
                numeric_error = abs(response.extracted_number - float(record.numeric_answer))
            else:
                numeric_error = None

        text_similarity: Optional[float] = None
        if record.has_text_answer():
            expected_text = self._parser.normalize_reference(record.text_answer)
            text_similarity = SequenceMatcher(None, expected_text, response.normalized_text).ratio()

        return EvaluationResult(
            record=record,
            response=response,
            numeric_match=numeric_match,
            numeric_error=numeric_error,
            text_similarity=text_similarity,
            error=None,
        )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LLM ground truth evaluation against stored answers.")
    default_csv = PROJECT_ROOT / "dataset" / "ground_truth.csv"
    parser.add_argument(
        "--csv",
        type=Path,
        default=default_csv,
        help="Percorso del file CSV contenente il ground truth (default: dataset/ground_truth.csv).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Numero massimo di domande da valutare (default: tutte).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Tolleranza relativa ammessa per le risposte numeriche (default: 1%).",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.65,
        help="Soglia di similarità testuale considerata accettabile (default: 0.65).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Mostra analisi dettagliata degli outlier con domande, risposte attese e ottenute.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    csv_path = args.csv
    if not csv_path.is_absolute():
        csv_path = (PROJECT_ROOT / csv_path).resolve()

    runner = GroundTruthTestRunner(
        csv_path=csv_path,
        numeric_tolerance=args.tolerance,
        text_threshold=args.text_threshold,
    )
    report = runner.run(limit=args.limit)

    print(report.render(verbose=args.verbose))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

