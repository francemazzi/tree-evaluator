from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.ground_truth_runner import GroundTruthTestRunner, TreeAgentClient


@pytest.mark.slow
def test_tree_evaluator_agent_ground_truth() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set; skipping ground truth agent test")

    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "dataset" / "ground_truth.csv"

    runner = GroundTruthTestRunner(
        csv_path=csv_path,
        agent_client=TreeAgentClient(api_key=api_key),
        numeric_tolerance=0.01,
        text_threshold=0.65,
    )

    report = runner.run(limit=5)

    numeric_accuracy = report.numeric_accuracy
    if numeric_accuracy is not None:
        assert numeric_accuracy >= 0.5, "Numeric accuracy below acceptable threshold"

    assert report.total_records > 0, "No records evaluated"

