from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class DatasetQueryInput(BaseModel):
    """Input schema for dataset query tool."""

    natural_query: str = Field(
        description="""Natural language question about the tree dataset. 
        The system will automatically translate this into SQL.
        
        Examples:
        - "Quanti alberi ci sono nel distretto 19?"
        - "Mostrami gli alberi di tipo Acer piantati dopo il 2000"
        - "Qual è la specie più comune nel dataset?"
        - "Dammi le statistiche per distretto"
        - "Trova tutti gli alberi con circonferenza maggiore di 100 cm"
        - "Qual è l'età media degli alberi nel distretto 10?"
        """
    )


class DatasetQueryTool(BaseTool):
    """Tool to query tree dataset using natural language that gets translated to SQL."""

    name: str = "query_tree_dataset"
    description: str = """
    Query the Vienna trees dataset (229,298 trees) using natural language.
    The system automatically translates your question into SQL and executes it.
    
    Database schema:
    - Table: baumkatogd
    - Key columns:
      * objectid (INTEGER): Unique tree ID
      * district (INTEGER): District number (1-23)
      * genus_species (VARCHAR): Tree species name (e.g., "Acer platanoides")
      * plant_year (INTEGER): Year the tree was planted
      * trunk_circumference (INTEGER): Trunk circumference in cm
      * tree_height (INTEGER): Height category (encoded)
      * crown_diameter (INTEGER): Crown diameter category (encoded)
      * object_street (VARCHAR): Street name
      * area_group (VARCHAR): Area management group
    
    Calculated fields:
    - DBH (diameter at breast height) = trunk_circumference / π
    - Age = current_year - plant_year
    
    Examples of questions you can ask:
    - "Quanti alberi ci sono in totale?"
    - "Quanti alberi nel distretto 19?"
    - "Mostra gli Acer piantati dopo il 2000"
    - "Qual è la specie più comune?"
    - "Statistiche per distretto"
    - "Alberi con circonferenza > 100 cm"
    - "Età media degli alberi nel distretto 10"
    - "Top 5 specie per numero"
    - "Dammi 10 alberi casuali del distretto 15"
    
    Use this tool whenever the user asks about trees, statistics, counts, or wants to explore the dataset.
    """
    args_schema: Type[BaseModel] = DatasetQueryInput

    _db_path: Path
    _llm: Any = None

    def __init__(self, db_path: Optional[Path] = None, llm: Any = None, **kwargs):
        super().__init__(**kwargs)
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent / "dataset" / "BAUMKATOGD.db"
        object.__setattr__(self, "_db_path", db_path)
        object.__setattr__(self, "_llm", llm)

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        if not self._db_path.exists():
            raise FileNotFoundError(
                f"Database not found at {self._db_path}. "
                f"Run 'python dataset/init_db.py' to create it."
            )
        
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _get_schema_info(self, conn: sqlite3.Connection) -> str:
        """Get database schema information."""
        cursor = conn.cursor()
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='baumkatogd'")
        schema = cursor.fetchone()
        return schema[0] if schema else "Schema not found"
    
    def _translate_to_sql(self, natural_query: str, schema_info: str) -> str:
        """Translate natural language query to SQL using LLM."""
        from datetime import datetime
        current_year = datetime.now().year
        
        prompt = f"""You are a SQL expert. Translate the user's natural language question into a SQLite query.

DATABASE SCHEMA:
{schema_info}

IMPORTANT NOTES:
1. Table name is: baumkatogd
2. Current year is {current_year} (use for age calculations)
3. DBH (diameter) = trunk_circumference / {math.pi}
4. Age = {current_year} - plant_year
5. Use LIMIT to avoid returning too many rows (max 100 for SELECT *, max 20 for aggregations)
6. For "mostrami" or "dammi" queries, use SELECT with LIMIT
7. For species searches, use LIKE with % wildcards (case-insensitive)
8. Common species keywords: Acer (acero), Tilia (tiglio), Quercus (quercia), Fraxinus (frassino)

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

Now translate this question:
{natural_query}"""
        
        if not self._llm:
            raise ValueError(
                "LLM is required for natural language to SQL translation. "
                "Please initialize DatasetQueryTool with an LLM instance."
            )
        
        response = self._llm.invoke(prompt)
        sql = response.content if hasattr(response, 'content') else str(response)
        
        # Clean up response
        sql = sql.strip()
        # Remove markdown code blocks if present
        if sql.startswith('```'):
            lines = sql.split('\n')
            # Remove first line if it's ```sql or ```
            if lines[0].startswith('```'):
                sql = '\n'.join(lines[1:])
        if sql.endswith('```'):
            sql = sql.rsplit('\n```', 1)[0]
        
        sql = sql.strip()
        return sql
    

    def _execute_sql(self, conn: sqlite3.Connection, sql: str) -> Dict[str, Any]:
        """Execute SQL query and format results."""
        try:
            cursor = conn.cursor()
            cursor.execute(sql)
            
            # Get column names
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                
                # Format results based on query type
                if len(rows) == 0:
                    return {
                        "sql_executed": sql,
                        "result": "No results found",
                        "row_count": 0
                    }
                
                # Single value result (COUNT, AVG, etc.)
                if len(columns) == 1 and len(rows) == 1:
                    return {
                        "sql_executed": sql,
                        "result": rows[0][0],
                        "column": columns[0]
                    }
                
                # Multiple rows
                results = []
                for row in rows:
                    result_dict = {}
                    for i, col in enumerate(columns):
                        result_dict[col] = row[i]
                    results.append(result_dict)
                
                return {
                    "sql_executed": sql,
                    "results": results,
                    "row_count": len(results),
                    "columns": columns
                }
            else:
                # Query executed but no results (INSERT, UPDATE, etc.)
                return {
                    "sql_executed": sql,
                    "result": "Query executed successfully",
                    "rows_affected": cursor.rowcount
                }
                
        except sqlite3.Error as e:
            return {
                "error": f"SQL execution error: {str(e)}",
                "sql_attempted": sql
            }

    def _run(self, natural_query: str) -> dict:
        """Execute natural language query by translating to SQL."""
        try:
            # Connect to database
            conn = self._get_connection()
            
            # Get schema information
            schema_info = self._get_schema_info(conn)
            
            # Translate natural language to SQL
            sql = self._translate_to_sql(natural_query, schema_info)
            
            # Execute SQL and get results
            result = self._execute_sql(conn, sql)
            
            # Close connection
            conn.close()
            
            # Add the original query to the result
            result["natural_query"] = natural_query
            
            return result
            
        except FileNotFoundError as e:
            return {"error": str(e), "natural_query": natural_query}
        except Exception as e:
            return {
                "error": f"Error processing query: {str(e)}",
                "natural_query": natural_query
            }

