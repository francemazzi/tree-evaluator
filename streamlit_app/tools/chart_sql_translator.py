from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from typing import Any, Dict

logger = logging.getLogger(__name__)


def translate_to_chart_sql(
    *,
    llm: Any,
    fallback_llm: Any,
    table_name: str,
    data_query: str,
    chart_type: str,
    schema_info: str,
) -> Dict[str, Any]:
    """Translate a natural-language chart request to SQL metadata."""
    if not llm:
        raise ValueError("LLM is required. Initialize ChartGenerationTool with an LLM instance.")

    prompt = build_chart_sql_prompt(
        table_name=table_name,
        data_query=data_query,
        chart_type=chart_type,
        schema_info=schema_info,
    )
    response = _invoke_with_fallback(llm, fallback_llm, prompt)
    parsed = _parse_chart_sql_response(response)
    missing_fields = [
        field
        for field in ["sql", "x_column", "suggested_title", "x_label", "y_label"]
        if field not in parsed
    ]
    if missing_fields:
        raise KeyError(f"Missing required fields: {', '.join(missing_fields)}")
    return parsed


def build_chart_sql_prompt(
    *,
    table_name: str,
    data_query: str,
    chart_type: str,
    schema_info: str,
) -> str:
    current_year = datetime.now().year
    return f"""You are a SQL expert for data visualization. Generate a SQL query for creating a {chart_type} chart.

DATABASE SCHEMA:
{schema_info}

IMPORTANT RULES:
1. Current year is {current_year}
2. Table name is: {table_name}
3. DBH = trunk_circumference / {math.pi} when trunk_circumference exists; otherwise use a diameter column if present.
4. Age = {current_year} - plant_year when plant_year exists.
5. Return data optimized for {chart_type} chart
6. For bar/pie charts: return category and count/value columns named exactly "category" and "count"
7. For line charts: return time-based x-axis and y-axis values
8. For scatter: return two numeric columns
9. For histogram: return the raw values to be binned
10. For box plots: return category and numeric value columns
11. For pie charts showing species: ALWAYS use TOP 15 + "Altro" pattern when genus_species exists
12. ALWAYS filter out NULL and empty string values with: WHERE column IS NOT NULL AND column <> ''
13. For pie/bar charts: limit to max 15-20 main categories, group rest as "Altro"
14. Use ONLY the table {table_name}; do not reference other tables.

USER REQUEST: {data_query}
CHART TYPE: {chart_type}

Return ONLY a valid JSON object with:
{{
    "sql": "the SQL query",
    "x_column": "name of x-axis column",
    "y_column": "name of y-axis column (or null for histogram)",
    "suggested_title": "suggested chart title in Italian",
    "x_label": "suggested x-axis label in Italian",
    "y_label": "suggested y-axis label in Italian"
}}

CRITICAL EXAMPLES:

Request: "Composizione specie" OR "distribuzione specie" OR "specie di piante"
Chart: pie
Response:
{{
    "sql": "WITH species_counts AS (SELECT genus_species, COUNT(*) AS count FROM {table_name} WHERE genus_species IS NOT NULL AND genus_species <> '' GROUP BY genus_species), ranked AS (SELECT genus_species, count, ROW_NUMBER() OVER (ORDER BY count DESC) AS rn FROM species_counts) SELECT CASE WHEN rn <= 15 THEN genus_species ELSE 'Altro' END AS category, SUM(count) AS count FROM ranked GROUP BY CASE WHEN rn <= 15 THEN genus_species ELSE 'Altro' END ORDER BY CASE WHEN category = 'Altro' THEN 1 ELSE 0 END, count DESC",
    "x_column": "category",
    "y_column": "count",
    "suggested_title": "Composizione delle Specie di Alberi",
    "x_label": "Specie",
    "y_label": "Numero di Alberi"
}}

Request: "Numero di alberi per distretto"
Chart: bar
Response:
{{
    "sql": "SELECT district AS category, COUNT(*) as count FROM {table_name} WHERE district IS NOT NULL AND district <> '' GROUP BY district ORDER BY count DESC",
    "x_column": "category",
    "y_column": "count",
    "suggested_title": "Numero di Alberi per Distretto",
    "x_label": "Distretto",
    "y_label": "Numero di Alberi"
}}

Request: "Distribuzione età degli alberi"
Chart: histogram
Response:
{{
    "sql": "SELECT ({current_year} - plant_year) as age FROM {table_name} WHERE plant_year > 0 AND plant_year < {current_year}",
    "x_column": "age",
    "y_column": null,
    "suggested_title": "Distribuzione dell'Età degli Alberi",
    "x_label": "Età (anni)",
    "y_label": "Frequenza"
}}

Now generate the query for: {data_query}

Remember: Return ONLY valid JSON, no markdown, no explanation."""


def _invoke_with_fallback(llm: Any, fallback_llm: Any, prompt: str) -> Any:
    try:
        return llm.invoke(prompt)
    except Exception as exc:
        if "rate_limit" in str(exc).lower() or "429" in str(exc) or "Request too large" in str(exc):
            if fallback_llm:
                try:
                    return fallback_llm.invoke(prompt)
                except Exception:
                    pass
        raise


def _parse_chart_sql_response(response: Any) -> Dict[str, Any]:
    response_text = response.content if hasattr(response, "content") else str(response)
    response_text = _strip_markdown_json(response_text.strip())
    if "{" in response_text:
        response_text = response_text[response_text.index("{"):response_text.rindex("}") + 1]
    try:
        return json.loads(response_text)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse LLM response: %s...", response_text[:200])
        logger.warning("JSON error: %s", exc)
        raise


def _strip_markdown_json(response_text: str) -> str:
    if "```json" in response_text:
        return response_text.split("```json")[1].split("```")[0].strip()
    if "```" not in response_text:
        return response_text
    for part in response_text.split("```"):
        part = part.strip()
        if part and (part.startswith("{") or part.startswith("[")):
            return part
    return response_text
