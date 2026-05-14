from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, Optional


def extract_requested_limit(data_query: str, default: int = 10, maximum: int = 50) -> int:
    """Extract an explicit top-N limit from a natural-language chart request."""
    query = data_query.lower()
    patterns = [
        r"\btop\s+(\d+)\b",
        r"\bprim(?:e|i)\s+(\d+)\b",
        r"\b(\d+)\s+(?:specie|species|categorie|distretti|district|municipi)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            return max(1, min(int(match.group(1)), maximum))
    return default


def get_predefined_chart_query(
    data_query: str,
    chart_type: str,
    table_name: str,
    columns: set[str],
) -> Optional[Dict[str, Any]]:
    """Return a predefined SQL/query metadata bundle for common chart requests."""
    data_query_lower = data_query.lower()
    requested_limit = extract_requested_limit(data_query)

    if (
        "genus_species" in columns
        and chart_type == "pie"
        and any(keyword in data_query_lower for keyword in ["specie", "species", "composizione", "distribuzione"])
    ):
        return _species_pie_query(table_name, requested_limit)

    if (
        "genus_species" in columns
        and chart_type == "bar"
        and any(keyword in data_query_lower for keyword in ["specie", "species", "top", "comuni", "diffuse"])
    ):
        return {
            "sql": f"""SELECT genus_species AS category, COUNT(*) AS count
                     FROM {table_name}
                     WHERE genus_species IS NOT NULL AND genus_species <> ''
                     GROUP BY genus_species
                     ORDER BY count DESC
                     LIMIT {requested_limit}""",
            "x_column": "category",
            "y_column": "count",
            "suggested_title": f"Top {requested_limit} Specie Piu' Comuni",
            "x_label": "Specie",
            "y_label": "Numero di Alberi",
        }

    if "district" in columns and any(keyword in data_query_lower for keyword in ["distretto", "district", "quartiere", "municipio"]):
        return {
            "sql": f"""SELECT district AS category, COUNT(*) as count
                     FROM {table_name}
                     WHERE district IS NOT NULL AND district <> ''
                     GROUP BY district
                     ORDER BY count DESC""",
            "x_column": "category",
            "y_column": "count",
            "suggested_title": "Distribuzione Alberi per Distretto",
            "x_label": "Distretto",
            "y_label": "Numero di Alberi",
        }

    if "plant_year" in columns and chart_type == "histogram" and any(keyword in data_query_lower for keyword in ["eta", "età", "age"]):
        current_year = datetime.now().year
        return {
            "sql": f"""SELECT ({current_year} - plant_year) as age
                     FROM {table_name}
                     WHERE plant_year IS NOT NULL AND plant_year > 0 AND plant_year < {current_year}""",
            "x_column": "age",
            "y_column": None,
            "suggested_title": "Distribuzione dell'Eta' degli Alberi",
            "x_label": "Eta' (anni)",
            "y_label": "Frequenza",
        }

    return None


def _species_pie_query(table_name: str, requested_limit: int) -> Dict[str, Any]:
    return {
        "sql": f"""WITH species_counts AS (
            SELECT genus_species, COUNT(*) AS count
            FROM {table_name}
            WHERE genus_species IS NOT NULL AND genus_species <> ''
            GROUP BY genus_species
        ),
        ranked AS (
            SELECT genus_species, count, ROW_NUMBER() OVER (ORDER BY count DESC) AS rn
            FROM species_counts
        )
        SELECT
            CASE WHEN rn <= {requested_limit} THEN genus_species ELSE 'Altro' END AS category,
            SUM(count) AS count
        FROM ranked
        GROUP BY CASE WHEN rn <= {requested_limit} THEN genus_species ELSE 'Altro' END
        ORDER BY
            CASE WHEN category = 'Altro' THEN 1 ELSE 0 END,
            count DESC""",
        "x_column": "category",
        "y_column": "count",
        "suggested_title": "Composizione delle Specie di Alberi",
        "x_label": "Specie",
        "y_label": "Numero di Alberi",
    }
