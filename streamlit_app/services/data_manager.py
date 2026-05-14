"""
Dynamic Data Manager Service

This module provides functionality to dynamically load CSV files,
convert them to SQLite databases, and make them available for the chat agent.
"""

from __future__ import annotations

import csv
import hashlib
import re
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import sqlite3

from streamlit_app.services.data_profile import (
    build_profile_summary,
    detect_column_roles,
    detect_semantic_type,
    json_safe_value,
    profile_dataframe,
)


class DatasetValidationError(ValueError):
    """Raised when an uploaded dataset cannot be safely imported."""


class DynamicDataManager:
    """
    Handles dynamic loading of CSV data into SQLite for the chat agent.
    
    This class provides methods to upload CSV files, automatically infer
    data types, and create SQLite databases that can be queried by the agent.
    """
    
    DEFAULT_MAX_UPLOAD_SIZE_MB = 200
    DEFAULT_MAX_ROWS = 1_000_000
    DEFAULT_MAX_COLUMNS = 500

    def __init__(
        self,
        upload_dir: Path | str = "temp_data",
        max_upload_size_mb: int = DEFAULT_MAX_UPLOAD_SIZE_MB,
        max_rows: int = DEFAULT_MAX_ROWS,
        max_columns: int = DEFAULT_MAX_COLUMNS,
    ):
        """
        Initialize the Data Manager.
        
        Args:
            upload_dir: Directory where uploaded files and databases will be stored
        """
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.max_upload_size_bytes = max_upload_size_mb * 1024 * 1024
        self.max_rows = max_rows
        self.max_columns = max_columns
    
    def _sanitize_column_name(self, column_name: str) -> str:
        """
        Sanitize column name for SQL compatibility.
        
        Args:
            column_name: Original column name
            
        Returns:
            Sanitized column name safe for SQL
        """
        # Replace spaces and special characters with underscores
        sanitized = re.sub(r'[^\w]', '_', column_name)
        # Remove consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        return sanitized.lower()

    def _get_file_bytes(self, uploaded_file) -> bytes:
        """Return uploaded file bytes regardless of Streamlit buffer type."""
        raw_buffer = uploaded_file.getbuffer()
        if hasattr(raw_buffer, "tobytes"):
            return raw_buffer.tobytes()
        if isinstance(raw_buffer, bytes):
            return raw_buffer
        return bytes(raw_buffer)

    def _safe_upload_path(self, filename: str) -> Path:
        """Resolve upload path from basename only, preventing path traversal."""
        safe_name = Path(str(filename or "")).name
        if not safe_name:
            raise DatasetValidationError("Nome file CSV mancante.")
        if not safe_name.lower().endswith(".csv"):
            raise DatasetValidationError("Sono supportati solo file CSV.")
        return self.upload_dir / safe_name

    def _validate_file_size(self, content: bytes) -> None:
        if not content:
            raise DatasetValidationError("Il file CSV è vuoto.")
        if len(content) > self.max_upload_size_bytes:
            max_mb = self.max_upload_size_bytes / (1024 * 1024)
            raise DatasetValidationError(
                f"Il file CSV supera il limite configurato di {max_mb:.0f} MB."
            )

    def _detect_encoding(self, content: bytes) -> str:
        """Detect a usable encoding with deterministic fallbacks."""
        for encoding in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                content.decode(encoding)
                return encoding
            except UnicodeDecodeError:
                continue
        raise DatasetValidationError("Encoding del CSV non supportato.")

    def _detect_delimiter(self, sample_text: str) -> str:
        """Detect CSV delimiter, falling back to a simple frequency heuristic."""
        candidate_delimiters = [",", ";", "\t", "|"]
        try:
            dialect = csv.Sniffer().sniff(sample_text, delimiters=candidate_delimiters)
            if dialect.delimiter in candidate_delimiters:
                return dialect.delimiter
        except csv.Error:
            pass

        first_lines = [line for line in sample_text.splitlines()[:10] if line.strip()]
        if not first_lines:
            raise DatasetValidationError("Il CSV non contiene righe leggibili.")

        scores = {
            delimiter: sum(line.count(delimiter) for line in first_lines)
            for delimiter in candidate_delimiters
        }
        delimiter, score = max(scores.items(), key=lambda item: item[1])
        return delimiter if score > 0 else ","

    def _make_unique_column_names(self, original_columns: list[str]) -> tuple[list[str], list[dict[str, Any]], list[str]]:
        """Sanitize column names and make duplicates unique instead of failing SQLite."""
        seen: dict[str, int] = {}
        sanitized_columns: list[str] = []
        mappings: list[dict[str, Any]] = []
        warnings: list[str] = []

        for index, original in enumerate(original_columns):
            original_str = str(original).strip()
            base_name = self._sanitize_column_name(original_str)
            if not base_name or base_name.startswith("unnamed"):
                base_name = f"column_{index + 1}"
                warnings.append(f"Colonna {index + 1} senza nome rinominata in {base_name}.")

            count = seen.get(base_name, 0) + 1
            seen[base_name] = count
            final_name = base_name if count == 1 else f"{base_name}_{count}"
            if final_name != base_name:
                warnings.append(
                    f"Nome colonna duplicato dopo sanitizzazione: {base_name} rinominato in {final_name}."
                )

            sanitized_columns.append(final_name)
            mappings.append(
                {
                    "index": index,
                    "original": original_str,
                    "sanitized": final_name,
                }
            )

        return sanitized_columns, mappings, warnings

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise DatasetValidationError("Il CSV non contiene righe dati.")
        if len(df.columns) == 0:
            raise DatasetValidationError("Il CSV non contiene colonne.")
        if len(df.columns) > self.max_columns:
            raise DatasetValidationError(
                f"Il CSV contiene {len(df.columns)} colonne, oltre il limite di {self.max_columns}."
            )
        if len(df) > self.max_rows:
            raise DatasetValidationError(
                f"Il CSV contiene {len(df):,} righe, oltre il limite di {self.max_rows:,}."
            )

    def _sample_records(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Return JSON-friendly preview rows."""
        sample = df.head(3).astype(object).where(pd.notna(df.head(3)), None)
        return sample.to_dict("records")

    def _json_safe_value(self, value: Any) -> Any:
        """Convert pandas/numpy scalar values into plain JSON-friendly values."""
        return json_safe_value(value)

    def _detect_semantic_type(self, series: pd.Series, unique_count: int) -> str:
        """Infer a coarse semantic type useful for prompting and UI hints."""
        return detect_semantic_type(series, unique_count)

    def _detect_column_roles(self, columns: list[str]) -> dict[str, list[str]]:
        """Detect common analytical roles from normalized column names."""
        return detect_column_roles(columns)

    def _profile_dataframe(
        self,
        df: pd.DataFrame,
        column_mappings: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build a compact dataset profile for UI and LLM context."""
        return profile_dataframe(df, column_mappings)

    def _build_profile_summary(self, profile: dict[str, Any], max_columns: int = 20) -> str:
        """Create a concise human/LLM-readable profile summary."""
        return build_profile_summary(profile, max_columns=max_columns)

    def process_uploaded_file(
        self, 
        uploaded_file, 
        table_name: Optional[str] = None
    ) -> Tuple[Path, str, Dict[str, Any]]:
        """
        Processes an uploaded CSV file and converts it to SQLite database.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            table_name: Optional custom table name (default: 'uploaded_data')
            
        Returns:
            Tuple containing:
                - Path to the created SQLite database
                - Table name used in the database
                - Dictionary with metadata (row_count, columns, etc.)
        """
        content = self._get_file_bytes(uploaded_file)
        self._validate_file_size(content)

        file_path = self._safe_upload_path(uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(content)

        encoding = self._detect_encoding(content)
        sample_text = content[:64 * 1024].decode(encoding, errors="replace")
        delimiter = self._detect_delimiter(sample_text)
        decimal = "," if delimiter != "," and re.search(r"\d+,\d+", sample_text) else "."

        try:
            df = pd.read_csv(
                StringIO(content.decode(encoding)),
                sep=delimiter,
                decimal=decimal,
                encoding=encoding,
                low_memory=False,
            )
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            raise DatasetValidationError(f"CSV non importabile: {exc}") from exc

        self._validate_dataframe(df)
        
        original_columns = df.columns.tolist()
        sanitized_columns, column_mappings, warnings = self._make_unique_column_names(original_columns)
        df.columns = sanitized_columns
        profile = self._profile_dataframe(df, column_mappings)
        
        metadata = {
            "original_filename": uploaded_file.name,
            "stored_filename": file_path.name,
            "file_size_bytes": len(content),
            "file_hash": hashlib.sha256(content).hexdigest(),
            "detected_encoding": encoding,
            "detected_delimiter": delimiter,
            "decimal_separator": decimal,
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "original_columns": original_columns,
            "column_mapping": dict(zip([str(c) for c in original_columns], df.columns)),
            "column_mappings": column_mappings,
            "dtypes": df.dtypes.astype(str).to_dict(),
            "sample_data": self._sample_records(df),
            "profile": profile,
            "profile_summary": self._build_profile_summary(profile),
            "warnings": warnings,
        }
        
        db_name = file_path.stem + ".db"
        db_path = self.upload_dir / db_name
        
        if table_name is None:
            table_name = "uploaded_data"
        else:
            table_name = self._sanitize_column_name(table_name)
        
        with sqlite3.connect(db_path) as conn:
            df.to_sql(table_name, conn, if_exists="replace", index=False)
        
        return db_path, table_name, metadata
    
    def get_schema_info(self, db_path: Path, table_name: str) -> str:
        """
        Get schema information from a SQLite database.
        
        Args:
            db_path: Path to the SQLite database
            table_name: Name of the table
            
        Returns:
            Schema information as a string
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            )
            schema = cursor.fetchone()
            return schema[0] if schema else "Schema not found"
        finally:
            conn.close()
    
    def cleanup_old_files(self, keep_latest: int = 5) -> None:
        """
        Clean up old uploaded files to save disk space.
        
        Args:
            keep_latest: Number of most recent files to keep
        """
        # Get all files sorted by modification time
        all_files = sorted(
            self.upload_dir.glob("*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        # Remove old files beyond the keep_latest limit
        for file_path in all_files[keep_latest:]:
            try:
                file_path.unlink()
            except Exception:
                pass  # Ignore errors during cleanup
