from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
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
    _embeddings: Optional[OpenAIEmbeddings] = None

    def __init__(self, db_path: Optional[Path] = None, llm: Any = None, **kwargs):
        super().__init__(**kwargs)
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent / "dataset" / "BAUMKATOGD.db"
        object.__setattr__(self, "_db_path", db_path)
        object.__setattr__(self, "_llm", llm)
        
        # Initialize embeddings for vector search (lazy initialization)
        object.__setattr__(self, "_embeddings", None)

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
5. **ALWAYS USE LIMIT** - NEVER return all rows without LIMIT (max 100 for SELECT *, max 20 for aggregations, LIMIT 1 for single results)
6. For "mostrami" or "dammi" queries, use SELECT with LIMIT
7. For species searches, use LIKE with % wildcards (case-insensitive)
8. Common species keywords: Acer (acero), Tilia (tiglio), Quercus (quercia), Fraxinus (frassino)
9. For "oldest/newest/largest/smallest" queries, use ORDER BY with LIMIT 1 or LIMIT 10
10. NEVER use SELECT * without LIMIT - always specify columns and LIMIT

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
    

    def _init_embeddings(self) -> OpenAIEmbeddings:
        """Initialize embeddings (lazy initialization)."""
        if self._embeddings is None:
            object.__setattr__(self, "_embeddings", OpenAIEmbeddings(model="text-embedding-3-small"))
        return self._embeddings
    
    def _semantic_filter_results(
        self, 
        rows: List[tuple], 
        columns: List[str], 
        natural_query: str, 
        top_k: int = 50
    ) -> List[Dict[str, Any]]:
        """Use LangChain InMemoryVectorStore to filter large result sets to most relevant items."""
        try:
            # Initialize embeddings
            embeddings = self._init_embeddings()
            
            # Convert rows to LangChain Documents with metadata
            documents = []
            
            for idx, row in enumerate(rows):
                # Create a dict representation of the row
                row_dict = {columns[i]: row[i] for i in range(len(columns))}
                
                # Create searchable text from row
                text_parts = []
                for col, val in row_dict.items():
                    if val is not None:
                        text_parts.append(f"{col}: {val}")
                page_content = " | ".join(text_parts)
                
                # Create LangChain Document with metadata
                doc = Document(
                    page_content=page_content,
                    metadata=row_dict
                )
                documents.append(doc)
            
            # Create InMemoryVectorStore with documents
            vectorstore = InMemoryVectorStore.from_documents(
                documents=documents,
                embedding=embeddings
            )
            
            # Perform similarity search with natural language query
            similar_docs = vectorstore.similarity_search(
                query=natural_query,
                k=min(top_k, len(rows))
            )
            
            # Extract metadata (which contains the actual row data)
            filtered_results = [doc.metadata for doc in similar_docs]
            
            return filtered_results
            
        except Exception as e:
            # If vector search fails, fall back to simple truncation
            print(f"Vector search failed: {e}, falling back to truncation")
            return [{columns[i]: row[i] for i in range(len(columns))} for row in rows[:top_k]]
    
    def _execute_sql(self, conn: sqlite3.Connection, sql: str, natural_query: str = "") -> Dict[str, Any]:
        """Execute SQL query and format results."""
        # Thresholds
        DIRECT_LIMIT = 100  # Direct return if <= this
        VECTOR_SEARCH_LIMIT = 50  # Return top N via vector search if > DIRECT_LIMIT
        
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
                
                # Multiple rows - check if we need vector search
                total_rows = len(rows)
                
                if total_rows <= DIRECT_LIMIT:
                    # Direct return for small result sets
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
                    # Use vector search for large result sets
                    filtered_results = self._semantic_filter_results(
                        rows, columns, natural_query, top_k=VECTOR_SEARCH_LIMIT
                    )
                    
                    return {
                        "sql_executed": sql,
                        "results": filtered_results,
                        "row_count": len(filtered_results),
                        "columns": columns,
                        "vector_search_applied": True,
                        "total_rows_found": total_rows,
                        "info": f"Vector search applied: showing top {len(filtered_results)} most relevant results out of {total_rows} total rows"
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
            
            # Execute SQL and get results (pass natural query for semantic filtering)
            result = self._execute_sql(conn, sql, natural_query=natural_query)
            
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

