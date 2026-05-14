from __future__ import annotations

import argparse
import os
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional, Sequence

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_app.agent import TreeEvaluatorAgent  # noqa: E402  pylint: disable=wrong-import-position
from tests.ground_truth_dataset import GroundTruthDataset  # noqa: E402
from tests.ground_truth_models import EvaluationResult, GroundTruthRecord, LLMParsedResponse  # noqa: E402
from tests.ground_truth_parser import NumericAnswerMatcher, ResponseParser  # noqa: E402
from tests.ground_truth_report import AccuracyReport  # noqa: E402


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
        results: list[EvaluationResult] = []

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
