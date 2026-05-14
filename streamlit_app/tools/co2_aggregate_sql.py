from __future__ import annotations

from typing import Any

CO2_AGGREGATE_DESCRIPTION = """
Calculate TOTAL STOCK (not annual absorption!) of CO2, biomass, and carbon for a group of trees.

IMPORTANT - This tool calculates STOCK (how much carbon/CO2 is STORED in trees right now).
This tool does NOT calculate ANNUAL ABSORPTION (how much carbon trees absorb per year).

Use this tool ONLY for:
- "stock di carbonio" / "carbon stock" / "carbonio immagazzinato"
- "CO2 totale immagazzinata" / "total stored CO2"
- "biomassa totale" / "total biomass"

Do NOT use this tool for:
- "carbonio assorbito annualmente" / "annual carbon absorption" - requires annual increment data not available
- "quanto carbonio assorbe all'anno" - this is annual rate, not stock
- Single tree calculations (use calculate_co2_sequestration instead)
- Carbon content/fraction per species (use lookup_carbon_content instead)

If user asks for ANNUAL absorption, explain that this data is not available in the dataset.
"""


def generate_sql_query(
    *,
    llm: Any,
    table_name: str,
    dataset_type: str,
    natural_query: str,
) -> str:
    """Generate a SELECT query based on the natural-language CO2 filter."""
    if not llm:
        raise ValueError("LLM is required for query generation")

    if dataset_type == "vienna":
        schema_hint = """
        Table: baumkatogd
        Columns: objectid, district (1-23), genus_species (e.g. 'Acer platanoides'),
        plant_year, trunk_circumference (cm), tree_height (0-8 cat), object_street
        """
    elif dataset_type == "milano":
        schema_hint = """
        Table: milano_trees
        Columns: _id, district, genus_species, plant_year,
        trunk_diameter_cm (cm), height_m (m), street
        """
    else:
        schema_hint = f"Table: {table_name} (Generic)"

    prompt = f"""You are a SQL expert. Generate a SELECT query to retrieve tree data for CO2 calculation.

    {schema_hint}

    User Query: "{natural_query}"

    Rules:
    1. Select ONLY the columns needed for calculation + identifiers:
       - Vienna: objectid, trunk_circumference, tree_height, genus_species
       - Milano: _id, trunk_diameter_cm, height_m, genus_species
    2. Apply the filtering logic requested (WHERE clause).
    3. Do NOT use LIMIT unless explicitly asked (we need all rows for total stock).
    4. If the user asks for "all trees" or "total", do not add a WHERE clause.
    5. For species, use LIKE with % (e.g. genus_species LIKE '%Pinus%').

    Return ONLY the SQL string.
    """

    response = llm.invoke(prompt)
    sql = response.content if hasattr(response, "content") else str(response)
    sql = sql.strip()
    if sql.startswith("```"):
        sql = sql.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    if sql.lower().startswith("sql"):
        sql = sql[3:].strip()
    return sql
