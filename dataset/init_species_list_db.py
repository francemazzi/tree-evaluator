#!/usr/bin/env python3
"""
Initialize SQLite database for the species list dataset (speciesList.csv).

This script creates dataset/species_list.db with a normalized schema (see species_list.sql)
and imports all rows from speciesList.csv.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional


@dataclass(frozen=True)
class SpeciesListPaths:
    dataset_dir: Path

    @property
    def csv_path(self) -> Path:
        return self.dataset_dir / "speciesList.csv"

    @property
    def schema_sql_path(self) -> Path:
        return self.dataset_dir / "species_list.sql"

    @property
    def db_path(self) -> Path:
        return self.dataset_dir / "species_list.db"


class SpeciesListSchemaApplier:
    """Applies the SQLite schema from a .sql file."""

    def __init__(self, schema_sql_path: Path) -> None:
        self._schema_sql_path = schema_sql_path

    def apply(self, connection: sqlite3.Connection) -> None:
        if not self._schema_sql_path.exists():
            raise FileNotFoundError(f"Schema SQL not found: {self._schema_sql_path}")
        sql_script = self._schema_sql_path.read_text(encoding="utf-8")
        connection.executescript(sql_script)


class SpeciesListRowMapper:
    """Maps a CSV row (original columns) into the normalized DB schema."""

    def map_row(self, row: Dict[str, str]) -> Dict[str, Optional[str]]:
        genus = (row.get("Genus Name") or "").strip()
        species = (row.get("Species Name") or "").strip()
        genus_species = f"{genus} {species}".strip()

        mapped: Dict[str, Optional[str]] = {
            "genus_name": genus or None,
            "species_name": species or None,
            "genus_species": genus_species or None,
            "synonyms": self._clean_text(row.get("Synonyms")),
            "family": self._clean_text(row.get("Family")),
            "taxonomic_order": self._clean_text(row.get("Order")),
            "taxonomic_class": self._clean_text(row.get("Class")),
            "common_name": self._clean_text(row.get("Common Name")),
            "species_code": self._clean_text(row.get("Species Code")),
            "growth_form": self._clean_text(row.get("Growth Form")),
            "percent_leaf_type": self._clean_text(row.get("Percent Leaf Type")),
            "leaf_type": self._clean_text(row.get("Leaf Type")),
            "growth_rate": self._clean_text(row.get("Growth Rate")),
            "longevity": self._clean_text(row.get("Longevity")),
            "height_at_maturity_feet": self._parse_int_to_str(row.get("Height at Maturity (feet)")),
        }

        mapped["source_row_hash"] = self._hash_row(mapped)
        return mapped

    @staticmethod
    def _clean_text(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        v = value.strip()
        return v if v else None

    @staticmethod
    def _parse_int_to_str(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        v = value.strip()
        if not v:
            return None
        try:
            return str(int(float(v)))
        except ValueError:
            return None

    @staticmethod
    def _hash_row(mapped: Dict[str, Optional[str]]) -> str:
        # Stable hash to support potential dedup/debugging.
        payload = "|".join(
            [
                mapped.get("genus_name") or "",
                mapped.get("species_name") or "",
                mapped.get("species_code") or "",
                mapped.get("family") or "",
                mapped.get("taxonomic_order") or "",
                mapped.get("taxonomic_class") or "",
                mapped.get("common_name") or "",
                mapped.get("growth_form") or "",
                mapped.get("leaf_type") or "",
                mapped.get("growth_rate") or "",
                mapped.get("longevity") or "",
                mapped.get("height_at_maturity_feet") or "",
            ]
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()


class SpeciesListDatabaseInitializer:
    """Creates the SQLite DB and imports the CSV dataset."""

    def __init__(self, paths: SpeciesListPaths) -> None:
        self._paths = paths
        self._schema = SpeciesListSchemaApplier(paths.schema_sql_path)
        self._mapper = SpeciesListRowMapper()

    def create_or_replace(self, force: bool = False) -> Path:
        self._validate_inputs()
        if self._paths.db_path.exists():
            if not force:
                raise FileExistsError(
                    f"Database already exists: {self._paths.db_path}. Run with --force to overwrite."
                )
            self._paths.db_path.unlink()

        connection = sqlite3.connect(self._paths.db_path.as_posix())
        try:
            connection.row_factory = sqlite3.Row
            self._schema.apply(connection)
            count = self._import_csv(connection)
            connection.commit()
        finally:
            connection.close()

        print(f"✅ Created {self._paths.db_path} with {count:,} rows in table 'species_list'")
        return self._paths.db_path

    def _validate_inputs(self) -> None:
        if not self._paths.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self._paths.csv_path}")

    def _import_csv(self, connection: sqlite3.Connection) -> int:
        with self._paths.csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = (self._mapper.map_row(r) for r in reader)
            return self._bulk_insert(connection, rows)

    @staticmethod
    def _bulk_insert(connection: sqlite3.Connection, rows: Iterable[Dict[str, Optional[str]]]) -> int:
        sql = """
            INSERT INTO species_list (
                genus_name,
                species_name,
                genus_species,
                synonyms,
                family,
                taxonomic_order,
                taxonomic_class,
                common_name,
                species_code,
                growth_form,
                percent_leaf_type,
                leaf_type,
                growth_rate,
                longevity,
                height_at_maturity_feet,
                source_row_hash
            ) VALUES (
                :genus_name,
                :species_name,
                :genus_species,
                :synonyms,
                :family,
                :taxonomic_order,
                :taxonomic_class,
                :common_name,
                :species_code,
                :growth_form,
                :percent_leaf_type,
                :leaf_type,
                :growth_rate,
                :longevity,
                :height_at_maturity_feet,
                :source_row_hash
            )
        """
        cursor = connection.cursor()
        batch = []
        total = 0
        for r in rows:
            batch.append(r)
            if len(batch) >= 1000:
                cursor.executemany(sql, batch)
                total += len(batch)
                batch = []
        if batch:
            cursor.executemany(sql, batch)
            total += len(batch)
        return total


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize species_list.db from speciesList.csv")
    parser.add_argument("--force", action="store_true", help="Overwrite existing species_list.db if present")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_dir = Path(__file__).parent
    paths = SpeciesListPaths(dataset_dir=dataset_dir)
    initializer = SpeciesListDatabaseInitializer(paths)
    initializer.create_or_replace(force=bool(args.force))


if __name__ == "__main__":
    main()


