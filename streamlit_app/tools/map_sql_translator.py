from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any, Dict


def translate_to_map_sql(
    *,
    llm: Any,
    fallback_llm: Any,
    table_name: str,
    lat_column: str,
    lon_column: str,
    data_query: str,
    max_points: int,
    schema_info: str,
) -> Dict[str, Any]:
    """Translate natural language data query to SQL for map visualization."""
    if not llm:
        raise ValueError("LLM is required. Initialize MapGenerationTool with an LLM instance.")

    prompt = build_map_sql_prompt(
        table_name=table_name,
        lat_column=lat_column,
        lon_column=lon_column,
        data_query=data_query,
        max_points=max_points,
        schema_info=schema_info,
    )
    response = _invoke_with_fallback(llm, fallback_llm, prompt)
    response_text = response.content if hasattr(response, "content") else str(response)
    return json.loads(_clean_json_response(response_text))


def build_map_sql_prompt(
    *,
    table_name: str,
    lat_column: str,
    lon_column: str,
    data_query: str,
    max_points: int,
    schema_info: str,
) -> str:
    current_year = datetime.now().year
    return f"""You are a SQL expert for map visualization. Generate a SQL query to get tree locations for a map.

DATABASE SCHEMA:
{schema_info}

IMPORTANT:
1. Current year is {current_year}
2. Table name is: {table_name}
3. ALWAYS select {lat_column} and {lon_column} columns - these are REQUIRED for the map
4. Filter out NULL coordinates: WHERE {lat_column} IS NOT NULL AND {lon_column} IS NOT NULL
5. Limit results to {max_points} points maximum
6. For species searches, use LIKE with % wildcards (case-insensitive)
7. Use only columns that appear in the schema above.
8. Use ONLY the table {table_name}; do not reference other tables.

USER REQUEST: {data_query}

Return a JSON object with:
{{
    "sql": "the SQL query - MUST include latitude and longitude columns",
    "suggested_title": "suggested map title in Italian",
    "popup_columns": ["list", "of", "columns", "to", "show", "in", "popup"],
    "center_lat": estimated center latitude (default: 45.4642 for Milano),
    "center_lon": estimated center longitude (default: 9.19 for Milano),
    "zoom": suggested zoom level (10-15, higher = more zoomed in)
}}

Examples:

Request: "Mostra i tigli sulla mappa"
Response:
{{
    "sql": "SELECT {lat_column}, {lon_column}, genus_species, trunk_diameter_cm, street FROM {table_name} WHERE genus_species LIKE '%Tilia%' AND {lat_column} IS NOT NULL AND {lon_column} IS NOT NULL LIMIT {max_points}",
    "suggested_title": "Distribuzione dei Tigli (Tilia) a Milano",
    "popup_columns": ["genus_species", "trunk_diameter_cm", "street"],
    "center_lat": 45.4642,
    "center_lon": 9.19,
    "zoom": 12
}}

Request: "Alberi del municipio 3"
Response:
{{
    "sql": "SELECT {lat_column}, {lon_column}, genus_species, trunk_diameter_cm, street FROM {table_name} WHERE district = 3 AND {lat_column} IS NOT NULL AND {lon_column} IS NOT NULL LIMIT {max_points}",
    "suggested_title": "Alberi del Municipio 3 di Milano",
    "popup_columns": ["genus_species", "trunk_diameter_cm", "street"],
    "center_lat": 45.48,
    "center_lon": 9.22,
    "zoom": 13
}}

Request: "Distribuzione degli alberi"
Response:
{{
    "sql": "SELECT {lat_column}, {lon_column}, genus_species FROM {table_name} WHERE {lat_column} IS NOT NULL AND {lon_column} IS NOT NULL LIMIT {max_points}",
    "suggested_title": "Distribuzione degli Alberi a Milano",
    "popup_columns": ["genere", "specie"],
    "center_lat": 45.4642,
    "center_lon": 9.19,
    "zoom": 11
}}

Now generate the query for: {data_query}"""


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


def _clean_json_response(response_text: str) -> str:
    response_text = response_text.strip()
    if response_text.startswith("```json"):
        response_text = response_text.split("```json")[1].split("```")[0].strip()
    elif response_text.startswith("```"):
        response_text = response_text.split("```")[1].split("```")[0].strip()
    if "{" in response_text:
        response_text = response_text[response_text.index("{"):response_text.rindex("}") + 1]
    return response_text
