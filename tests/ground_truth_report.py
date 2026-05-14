from __future__ import annotations

from statistics import median
from typing import List, Optional, Sequence

from tests.ground_truth_models import EvaluationResult


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

        if verbose and failures:
            self._append_verbose_failures(lines, failures)

        return "\n".join(lines)

    def _append_verbose_failures(self, lines: List[str], failures: List[EvaluationResult]) -> None:
        lines.append("\n" + "=" * 80)
        lines.append("DETAILED OUTLIER ANALYSIS")
        lines.append("=" * 80)

        for idx, failure in enumerate(failures, 1):
            lines.append(f"\n[OUTLIER {idx}] ID: {failure.record.identifier}")
            lines.append("-" * 80)

            lines.append("❓ DOMANDA:")
            lines.append(f"   {failure.record.question}")
            lines.append("")

            if failure.record.has_numeric_answer():
                lines.append("✅ RISPOSTA NUMERICA ATTESA:")
                lines.append(f"   {failure.record.numeric_answer}")
                lines.append("")

            if failure.record.has_text_answer():
                lines.append("✅ RISPOSTA TESTUALE ATTESA:")
                lines.append(f"   {failure.record.text_answer}")
                lines.append("")

            lines.append("🤖 RISPOSTA AGENTE:")
            if failure.response.raw_text:
                self._append_wrapped_response(lines, failure.response.raw_text)
            else:
                lines.append("   [Nessuna risposta]")
            lines.append("")

            lines.append("📊 METRICHE:")
            if failure.record.has_numeric_answer():
                lines.append(f"   • Numero estratto: {failure.response.extracted_number}")
                lines.append(f"   • Match numerico: {'✓ SI' if failure.numeric_match else '✗ NO'}")
                if failure.numeric_error is not None:
                    lines.append(f"   • Errore assoluto: {failure.numeric_error:.2f}")

            if failure.text_similarity is not None:
                lines.append(
                    f"   • Similarità testuale: {failure.text_similarity:.2%} (soglia: {self._text_threshold:.2%})"
                )
                if failure.text_similarity < self._text_threshold:
                    lines.append("   • Status: ✗ SOTTO SOGLIA")
                else:
                    lines.append("   • Status: ✓ SOPRA SOGLIA")

            if failure.error:
                lines.append(f"   • Errore: {failure.error}")

            lines.append("")

    @staticmethod
    def _append_wrapped_response(lines: List[str], response: str) -> None:
        for line in response.split("\n"):
            if len(line) <= 75:
                lines.append(f"   {line}")
                continue

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
