from __future__ import annotations

import math
from datetime import datetime
from typing import Any


def translate_to_sql(
    *,
    llm: Any,
    fallback_llm: Any,
    table_name: str,
    user_description: str,
    natural_query: str,
    schema_info: str,
) -> str:
    """Translate a natural-language dataset question to SQLite."""
    if not llm:
        raise ValueError(
            "LLM is required for natural language to SQL translation. "
            "Please initialize DatasetQueryTool with an LLM instance."
        )

    prompt = build_dataset_sql_prompt(
        table_name=table_name,
        user_description=user_description,
        natural_query=natural_query,
        schema_info=schema_info,
    )
    response = _invoke_with_fallback(llm, fallback_llm, prompt)
    sql = response.content if hasattr(response, "content") else str(response)
    return clean_sql_response(sql)


def build_dataset_sql_prompt(
    *,
    table_name: str,
    user_description: str,
    natural_query: str,
    schema_info: str,
) -> str:
    current_year = datetime.now().year
    return f"""You are a SQL expert. Translate the user's natural language question into a SQLite query.

DATABASE SCHEMA:
{schema_info}

USER-PROVIDED CONTEXT ABOUT THE DATA:
{user_description if user_description else "No additional context provided - infer from schema"}

IMPORTANT NOTES:
1. Table name is: {table_name}
2. Current year is {current_year} (use for age calculations)
3. DBH (diameter) = trunk_circumference / {math.pi}
4. Age = {current_year} - plant_year
5. **ALWAYS USE LIMIT** - NEVER return all rows without LIMIT (max 100 for SELECT *, max 20 for aggregations, LIMIT 1 for single results)
6. For "mostrami" or "dammi" queries, use SELECT with LIMIT
7. For species searches, use LIKE with % wildcards (case-insensitive)
8. Common species keywords: Acer (acero), Tilia (tiglio), Quercus (quercia), Fraxinus (frassino), Pinus (pino)
9. For "oldest/newest/largest/smallest" queries, use ORDER BY with LIMIT 1 or LIMIT 10
10. NEVER use SELECT * without LIMIT - always specify columns and LIMIT

CRITICAL FOR COMPOSITE QUERIES:
- If the question mentions "distretto con più alberi/piante" or similar, use a SUBQUERY to find that district first
- Example: "specie del distretto con più piante" should be translated to:
  SELECT genus_species, COUNT(*) as count FROM {table_name}
  WHERE district = (SELECT district FROM {table_name} GROUP BY district ORDER BY COUNT(*) DESC LIMIT 1)
  GROUP BY genus_species ORDER BY count DESC LIMIT 20

USER QUESTION: {natural_query}

Return ONLY the SQL query, nothing else. No explanations, no markdown, just the SQL.
Examples:

Question: "Quanti alberi ci sono?"
SQL: SELECT COUNT(*) as total FROM baumkatogd

Question: "Quanti alberi nel distretto 19?"
SQL: SELECT COUNT(*) as total FROM baumkatogd WHERE district = 19

Question: "Mostra gli Acer piantati dopo 2000"
SQL: SELECT objectid, genus_species, plant_year, district, trunk_circumference FROM baumkatogd WHERE genus_species LIKE '%Acer%' AND plant_year > 2000 LIMIT 20

Question: "Qual è la specie più comune?"
SQL: SELECT genus_species, COUNT(*) as count FROM baumkatogd GROUP BY genus_species ORDER BY count DESC LIMIT 1

Question: "Top 5 specie"
SQL: SELECT genus_species, COUNT(*) as count FROM baumkatogd WHERE genus_species IS NOT NULL GROUP BY genus_species ORDER BY count DESC LIMIT 5

Question: "Statistiche per distretto"
SQL: SELECT district, COUNT(*) as count, ROUND(AVG(trunk_circumference / {math.pi}), 1) as avg_dbh_cm, ROUND(AVG({current_year} - plant_year), 1) as avg_age FROM baumkatogd WHERE district IS NOT NULL GROUP BY district ORDER BY count DESC LIMIT 20

Question: "Alberi con circonferenza > 100"
SQL: SELECT objectid, genus_species, trunk_circumference, district FROM baumkatogd WHERE trunk_circumference > 100 ORDER BY trunk_circumference DESC LIMIT 20

Question: "Età media alberi distretto 10"
SQL: SELECT ROUND(AVG({current_year} - plant_year), 1) as avg_age FROM baumkatogd WHERE district = 10 AND plant_year > 0

Question: "Qual è l'albero più vecchio?"
SQL: SELECT objectid, genus_species, plant_year, district, ({current_year} - plant_year) as age FROM baumkatogd WHERE plant_year > 0 ORDER BY plant_year ASC LIMIT 1

Question: "Mostra i 10 alberi più vecchi"
SQL: SELECT objectid, genus_species, plant_year, district, ({current_year} - plant_year) as age FROM baumkatogd WHERE plant_year > 0 ORDER BY plant_year ASC LIMIT 10

Question: "Quali sono le specie del distretto con più piante?"
SQL: SELECT genus_species, COUNT(*) as count FROM baumkatogd WHERE district = (SELECT district FROM baumkatogd WHERE district IS NOT NULL GROUP BY district ORDER BY COUNT(*) DESC LIMIT 1) GROUP BY genus_species ORDER BY count DESC LIMIT 20

Question: "Specie nel distretto 22"
SQL: SELECT genus_species, COUNT(*) as count FROM baumkatogd WHERE district = 22 GROUP BY genus_species ORDER BY count DESC LIMIT 20

Now translate this question:
{natural_query}"""


def clean_sql_response(sql: str) -> str:
    sql = sql.strip()
    if sql.startswith("```"):
        lines = sql.split("\n")
        if lines[0].startswith("```"):
            sql = "\n".join(lines[1:])
    if sql.endswith("```"):
        sql = sql.rsplit("\n```", 1)[0]
    return sql.strip()


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
