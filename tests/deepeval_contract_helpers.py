from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, Iterable, List, Tuple

import pytest

deepeval = pytest.importorskip("deepeval")
from deepeval import assert_test  # noqa: E402
from deepeval.metrics import BaseMetric  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeLLM:
    def __init__(self, content: str = "") -> None:
        self._content = content

    def bind_tools(self, _tools: list[Any]) -> "FakeLLM":
        return self

    def invoke(self, _prompt: Any) -> FakeResponse:
        return FakeResponse(self._content)


class FakeEmbeddings:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[float(len(text)), 0.0, 1.0] for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return [float(len(text)), 0.0, 1.0]


Check = Callable[[Dict[str, Any]], Tuple[bool, str]]


class ToolContractMetric(BaseMetric):
    """Small deterministic DeepEval metric for local tool contracts."""

    threshold = 1.0
    evaluation_model = None
    strict_mode = True
    async_mode = False
    verbose_mode = False
    include_reason = True

    def __init__(self, name: str, checks: Iterable[Check]) -> None:
        self.threshold = 1.0
        self.name = name
        self._checks = list(checks)
        self.score = 0.0
        self.success = False
        self.reason = "Not measured yet"
        self.error: str | None = None

    def measure(self, test_case: LLMTestCase, *_args: Any, **_kwargs: Any) -> float:
        try:
            payload = json.loads(test_case.actual_output)
        except json.JSONDecodeError as exc:
            self.success = False
            self.score = 0.0
            self.reason = f"actual_output is not valid JSON: {exc}"
            return self.score

        failures: list[str] = []
        for check in self._checks:
            ok, reason = check(payload)
            if not ok:
                failures.append(reason)

        self.success = not failures
        self.score = 1.0 if self.success else 0.0
        self.reason = "All contract checks passed" if self.success else "; ".join(failures)
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *_args: Any, **_kwargs: Any) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:
        return self.name


def _get_path(payload: Dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def path_exists(path: str) -> Check:
    def check(payload: Dict[str, Any]) -> Tuple[bool, str]:
        value = _get_path(payload, path)
        return value is not None, f"Missing required path: {path}"

    return check


def path_numeric_gt(path: str, minimum: float) -> Check:
    def check(payload: Dict[str, Any]) -> Tuple[bool, str]:
        value = _get_path(payload, path)
        ok = isinstance(value, (int, float)) and value > minimum
        return ok, f"Expected {path} to be numeric and > {minimum}, got {value!r}"

    return check


def output_has_no_error(payload: Dict[str, Any]) -> Tuple[bool, str]:
    output = payload.get("tool_output", {})
    has_error = isinstance(output, dict) and bool(output.get("error"))
    return not has_error, f"Tool returned error: {output.get('error') if isinstance(output, dict) else output!r}"


def output_success(path: str = "tool_output.success") -> Check:
    def check(payload: Dict[str, Any]) -> Tuple[bool, str]:
        value = _get_path(payload, path)
        return value is True, f"Expected {path} to be true, got {value!r}"

    return check


def assert_deepeval_contract(
    *,
    case_name: str,
    user_input: str,
    actual: Dict[str, Any],
    checks: Iterable[Check],
) -> None:
    os.environ.setdefault("DEEPEVAL_RESULTS_FOLDER", ".deepeval-results")
    test_case = LLMTestCase(
        input=user_input,
        actual_output=json.dumps(actual, default=str, ensure_ascii=False),
        expected_output=json.dumps({"case": case_name}, ensure_ascii=False),
    )
    metric = ToolContractMetric(name=f"{case_name}_contract", checks=checks)
    assert_test(test_case, [metric], run_async=False)
