from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from tests.ground_truth_models import GroundTruthRecord


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
