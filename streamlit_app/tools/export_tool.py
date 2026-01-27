"""Export tool for exporting data to CSV or Excel format."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type, Union

import pandas as pd
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class ExportDataInput(BaseModel):
    """Input schema for export data tool."""

    data: Union[List[Dict[str, Any]], str] = Field(
        description="""Data to export. Can be:
        - A list of dictionaries (query results)
        - A JSON string representing the data

        Examples:
        - [{"species": "Acer", "count": 100}, {"species": "Tilia", "count": 50}]
        - '{"results": [{"id": 1, "name": "Tree1"}]}'
        """
    )

    format: Literal["csv", "excel"] = Field(
        default="csv",
        description="Export format: 'csv' or 'excel'"
    )

    filename: Optional[str] = Field(
        default=None,
        description="Optional custom filename (without extension). If not provided, a timestamp-based name will be generated."
    )


class ExportDataTool(BaseTool):
    """Tool to export data to CSV or Excel format."""

    name: str = "export_data"
    description: str = """
    Export data to CSV or Excel file for download.

    Use this tool when the user wants to:
    - Export query results to a file
    - Save data as CSV or Excel
    - Download data for external use

    The tool accepts:
    - Query results (list of dictionaries)
    - JSON data strings

    Returns the path to the exported file.

    Examples:
    - "Esporta questi dati in CSV"
    - "Scarica i risultati in Excel"
    - "Export the data to spreadsheet"
    """
    args_schema: Type[BaseModel] = ExportDataInput

    _output_dir: Path
    _language: str = "it"

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        language: str = "it",
        **kwargs
    ):
        super().__init__(**kwargs)

        # Default output directory: project_root/exports
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / "exports"

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        object.__setattr__(self, "_output_dir", output_dir)
        object.__setattr__(self, "_language", language)

    def _parse_data(self, data: Union[List[Dict[str, Any]], str]) -> pd.DataFrame:
        """Parse input data into a DataFrame."""
        if isinstance(data, str):
            try:
                parsed = json.loads(data)
                # Handle nested results format
                if isinstance(parsed, dict):
                    if "results" in parsed:
                        parsed = parsed["results"]
                    elif "data" in parsed:
                        parsed = parsed["data"]
                data = parsed
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON data: {e}")

        if not isinstance(data, list):
            raise ValueError("Data must be a list of dictionaries")

        if len(data) == 0:
            raise ValueError("Data is empty, nothing to export")

        return pd.DataFrame(data)

    def _generate_filename(self, custom_name: Optional[str], extension: str) -> str:
        """Generate a filename with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if custom_name:
            # Clean custom name
            clean_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in custom_name)
            return f"{clean_name}_{timestamp}.{extension}"

        return f"export_{timestamp}.{extension}"

    def _run(
        self,
        data: Union[List[Dict[str, Any]], str],
        format: Literal["csv", "excel"] = "csv",
        filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """Export data to the specified format."""
        try:
            # Parse data into DataFrame
            df = self._parse_data(data)

            # Determine file extension
            extension = "xlsx" if format == "excel" else "csv"

            # Generate filename
            output_filename = self._generate_filename(filename, extension)
            output_path = self._output_dir / output_filename

            # Export based on format
            if format == "excel":
                df.to_excel(output_path, index=False, engine="openpyxl")
            else:
                df.to_csv(output_path, index=False, encoding="utf-8-sig")

            # Build response based on language
            if self._language == "en":
                return {
                    "success": True,
                    "message": f"Data exported successfully to {format.upper()}",
                    "file_path": str(output_path),
                    "filename": output_filename,
                    "rows_exported": len(df),
                    "columns": list(df.columns),
                    "format": format,
                    "instruction": f"The file has been saved to: {output_path}"
                }
            else:
                return {
                    "success": True,
                    "message": f"Dati esportati con successo in {format.upper()}",
                    "file_path": str(output_path),
                    "filename": output_filename,
                    "rows_exported": len(df),
                    "columns": list(df.columns),
                    "format": format,
                    "instruction": f"Il file e' stato salvato in: {output_path}"
                }

        except ValueError as e:
            error_msg = str(e)
            if self._language == "en":
                return {
                    "success": False,
                    "error": f"Data validation error: {error_msg}",
                    "hint": "Ensure data is a valid list of dictionaries or JSON string."
                }
            else:
                return {
                    "success": False,
                    "error": f"Errore di validazione dati: {error_msg}",
                    "hint": "Assicurati che i dati siano una lista di dizionari valida o una stringa JSON."
                }

        except Exception as e:
            error_msg = str(e)
            if self._language == "en":
                return {
                    "success": False,
                    "error": f"Export error: {error_msg}",
                    "hint": "Check that the data format is correct and try again."
                }
            else:
                return {
                    "success": False,
                    "error": f"Errore di esportazione: {error_msg}",
                    "hint": "Verifica che il formato dei dati sia corretto e riprova."
                }
