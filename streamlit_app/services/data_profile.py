from __future__ import annotations

from typing import Any

import pandas as pd


def json_safe_value(value: Any) -> Any:
    """Convert pandas/numpy scalar values into plain JSON-friendly values."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return value


def detect_semantic_type(series: pd.Series, unique_count: int) -> str:
    """Infer a coarse semantic type useful for prompting and UI hints."""
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    non_null_sample = series.dropna().astype(str).head(1000)
    if not non_null_sample.empty:
        date_like = non_null_sample.str.match(
            r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}"
            r"|^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"
            r"|^\d{4}-\d{1,2}$"
        )
        if date_like.mean() < 0.5:
            row_count = len(series)
            if unique_count <= max(20, int(row_count * 0.2)):
                return "categorical"
            return "text"
        parsed_dates = pd.to_datetime(non_null_sample, errors="coerce", format="mixed")
        parse_ratio = parsed_dates.notna().mean()
        if parse_ratio >= 0.8:
            return "datetime"

    row_count = len(series)
    if unique_count <= max(20, int(row_count * 0.2)):
        return "categorical"
    return "text"


def detect_column_roles(columns: list[str]) -> dict[str, list[str]]:
    """Detect common analytical roles from normalized column names."""
    roles = {
        "latitude_candidates": [],
        "longitude_candidates": [],
        "date_candidates": [],
        "species_candidates": [],
        "diameter_candidates": [],
        "circumference_candidates": [],
        "height_candidates": [],
    }
    for column in columns:
        col = column.lower()
        if col in {"lat", "latitude", "latitudine", "y"} or "latitude" in col or "latitudine" in col:
            roles["latitude_candidates"].append(column)
        if col in {"lon", "lng", "longitude", "longitudine", "x"} or "longitude" in col or "longitudine" in col:
            roles["longitude_candidates"].append(column)
        if any(token in col for token in ("date", "data", "year", "anno", "mese", "month")):
            roles["date_candidates"].append(column)
        if any(token in col for token in ("species", "specie", "genus", "genere")):
            roles["species_candidates"].append(column)
        if any(token in col for token in ("diameter", "diametro", "dbh")):
            roles["diameter_candidates"].append(column)
        if any(token in col for token in ("circumference", "circonferenza", "circ")):
            roles["circumference_candidates"].append(column)
        if any(token in col for token in ("height", "altezza")):
            roles["height_candidates"].append(column)
    return roles


def profile_dataframe(
    df: pd.DataFrame,
    column_mappings: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a compact dataset profile for UI and LLM context."""
    mappings_by_sanitized = {
        mapping["sanitized"]: mapping["original"]
        for mapping in column_mappings
    }
    column_profiles: dict[str, dict[str, Any]] = {}
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    datetime_columns: list[str] = []
    text_columns: list[str] = []

    for column in df.columns:
        series = df[column]
        non_null = series.dropna()
        unique_count = int(series.nunique(dropna=True))
        semantic_type = detect_semantic_type(series, unique_count)
        if semantic_type == "numeric":
            numeric_columns.append(column)
        elif semantic_type == "datetime":
            datetime_columns.append(column)
        elif semantic_type == "categorical":
            categorical_columns.append(column)
        elif semantic_type == "text":
            text_columns.append(column)

        sample_values = [
            json_safe_value(value)
            for value in non_null.drop_duplicates().head(5).tolist()
        ]

        profile = {
            "original_name": mappings_by_sanitized.get(column, column),
            "dtype": str(series.dtype),
            "semantic_type": semantic_type,
            "null_count": int(series.isna().sum()),
            "null_ratio": round(float(series.isna().mean()), 4),
            "non_null_count": int(series.notna().sum()),
            "unique_count": unique_count,
            "sample_values": sample_values,
        }

        if semantic_type == "numeric":
            numeric_series = pd.to_numeric(series, errors="coerce")
            profile["min"] = json_safe_value(numeric_series.min())
            profile["max"] = json_safe_value(numeric_series.max())
            profile["mean"] = json_safe_value(round(float(numeric_series.mean()), 6))
        elif semantic_type == "datetime" and not non_null.empty:
            parsed_dates = pd.to_datetime(non_null.astype(str), errors="coerce", format="mixed")
            profile["min"] = json_safe_value(parsed_dates.min())
            profile["max"] = json_safe_value(parsed_dates.max())

        column_profiles[column] = profile

    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": column_profiles,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "text_columns": text_columns,
        "roles": detect_column_roles(list(df.columns)),
    }


def build_profile_summary(profile: dict[str, Any], max_columns: int = 20) -> str:
    """Create a concise human/LLM-readable profile summary."""
    lines = [
        f"Righe: {profile['row_count']:,}; colonne: {profile['column_count']}.",
    ]
    if profile["numeric_columns"]:
        lines.append("Colonne numeriche: " + ", ".join(profile["numeric_columns"][:20]) + ".")
    if profile["categorical_columns"]:
        lines.append("Colonne categoriche: " + ", ".join(profile["categorical_columns"][:20]) + ".")
    if profile["datetime_columns"]:
        lines.append("Colonne data/tempo: " + ", ".join(profile["datetime_columns"][:20]) + ".")

    roles = profile.get("roles", {})
    role_lines = []
    for label, columns in roles.items():
        if columns:
            role_lines.append(f"{label}: {', '.join(columns)}")
    if role_lines:
        lines.append("Ruoli rilevati: " + "; ".join(role_lines) + ".")

    lines.append("Dettaglio colonne principali:")
    for column, column_profile in list(profile["columns"].items())[:max_columns]:
        examples = ", ".join(str(v) for v in column_profile.get("sample_values", [])[:3])
        detail = (
            f"- {column} (originale: {column_profile['original_name']}): "
            f"{column_profile['semantic_type']}, null {column_profile['null_count']}, "
            f"valori unici {column_profile['unique_count']}"
        )
        if examples:
            detail += f", esempi: {examples}"
        lines.append(detail)

    if profile["column_count"] > max_columns:
        lines.append(f"... altre {profile['column_count'] - max_columns} colonne omesse dal riepilogo.")
    return "\n".join(lines)
