from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from langchain_core.vectorstores import InMemoryVectorStore
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
    _fallback_llm: Any = None
    _embeddings: Any = None
    _table_name: str = "baumkatogd"
    _user_description: str = ""
    _dataset_source: Dict[str, Any] = {}
    
    # Known dataset sources for open data
    DATASET_SOURCES: Dict[str, Dict[str, Any]] = {
        "baumkatogd": {
            "name": "Vienna Tree Cadastre (Baumkataster der Stadt Wien)",
            "provider": "City of Vienna - Open Data",
            "url": "https://www.data.gv.at/katalog/dataset/stadt-wien_baumkatasterderstadtwien",
            "license": "Creative Commons Attribution 4.0 (CC BY 4.0)",
            "description": "Public tree inventory of the City of Vienna"
        },
        "dataset_milano": {
            "name": "Censimento Alberi del Comune di Milano",
            "provider": "Comune di Milano - Open Data",
            "url": "https://dati.comune.milano.it/dataset/ds447-infogeo-aree-verdi-alberi-702eb2e7",
            "license": "Creative Commons Attribution 4.0 (CC BY 4.0)",
            "description": "Inventario degli alberi pubblici del Comune di Milano"
        }
    }

    def __init__(
        self, 
        db_path: Optional[Path] = None, 
        table_name: Optional[str] = None,
        user_description: str = "",
        llm: Any = None, 
        fallback_llm: Any = None,
        embeddings: Any = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent / "dataset" / "BAUMKATOGD.db"
        object.__setattr__(self, "_db_path", db_path)
        object.__setattr__(self, "_llm", llm)
        object.__setattr__(self, "_fallback_llm", fallback_llm)
        
        # Set table name (default: baumkatogd for Vienna trees)
        if table_name:
            object.__setattr__(self, "_table_name", table_name)
        
        # Set user-provided description for context
        object.__setattr__(self, "_user_description", user_description)
        
        # Initialize embeddings for vector search (lazy initialization)
        object.__setattr__(self, "_embeddings", embeddings)
        
        # Set dataset source based on table name or db path
        dataset_source = self._determine_dataset_source(db_path, table_name)
        object.__setattr__(self, "_dataset_source", dataset_source)
        
        # Update description if custom dataset is used
        if table_name and table_name != "baumkatogd":
            self._update_description_for_custom_dataset()
    
    def _determine_dataset_source(self, db_path: Path, table_name: Optional[str]) -> Dict[str, Any]:
        """Determine the data source based on database path or table name."""
        # Check if it's the Milano dataset
        if db_path and "milano" in str(db_path).lower():
            return self.DATASET_SOURCES.get("dataset_milano", {})
        
        # Check table name for known datasets
        if table_name:
            table_lower = table_name.lower()
            if "milano" in table_lower or table_lower == "milano_trees":
                return self.DATASET_SOURCES.get("dataset_milano", {})
            elif table_lower == "baumkatogd":
                return self.DATASET_SOURCES.get("baumkatogd", {})
        
        # Default to Vienna if using default db
        if db_path and "BAUMKATOGD" in str(db_path):
            return self.DATASET_SOURCES.get("baumkatogd", {})
        
        # Custom dataset - no known source
        return {
            "name": "Custom Dataset",
            "provider": "User uploaded",
            "description": "Custom dataset uploaded by user"
        }

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
        cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{self._table_name}'")
        schema = cursor.fetchone()
        return schema[0] if schema else "Schema not found"
    
    def _update_description_for_custom_dataset(self) -> None:
        """Update tool description for custom uploaded datasets."""
        custom_description = f"""
    Query a custom uploaded dataset using natural language.
    The system automatically translates your question into SQL and executes it.
    
    Database table: {self._table_name}
    
    User-provided context:
    {self._user_description if self._user_description else "No additional context provided"}
    
    You can ask questions like:
    - "Quanti record ci sono in totale?"
    - "Mostrami i primi 10 record"
    - "Qual è il valore medio di [colonna]?"
    - "Raggruppa i dati per [colonna]"
    - "Trova i record dove [condizione]"
    
    Use this tool whenever the user asks about the dataset, statistics, counts, or wants to explore the data.
    """
        object.__setattr__(self, "description", custom_description)
    
    def _translate_to_sql(self, natural_query: str, schema_info: str) -> str:
        """Translate natural language query to SQL using LLM."""
        from datetime import datetime
        current_year = datetime.now().year
        
        prompt = f"""You are a SQL expert. Translate the user's natural language question into a SQLite query.

DATABASE SCHEMA:
{schema_info}

USER-PROVIDED CONTEXT ABOUT THE DATA:
{self._user_description if self._user_description else "No additional context provided - infer from schema"}

IMPORTANT NOTES:
1. Table name is: {self._table_name}
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
  SELECT genus_species, COUNT(*) as count FROM {self._table_name} 
  WHERE district = (SELECT district FROM {self._table_name} GROUP BY district ORDER BY COUNT(*) DESC LIMIT 1)
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
        
        if not self._llm:
            raise ValueError(
                "LLM is required for natural language to SQL translation. "
                "Please initialize DatasetQueryTool with an LLM instance."
            )

        def _invoke_with_fallback() -> Any:
            try:
                return self._llm.invoke(prompt)
            except Exception as e:
                # Fallback to a lighter model on rate-limit or size errors
                if "rate_limit" in str(e).lower() or "429" in str(e) or "Request too large" in str(e):
                    try:
                        if self._fallback_llm:
                            return self._fallback_llm.invoke(prompt)
                    except Exception:
                        pass
                raise

        response = _invoke_with_fallback()
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
    

    def _init_embeddings(self) -> Any:
        """Return embeddings instance if available; otherwise raise to trigger truncation fallback."""
        if self._embeddings is None:
            raise RuntimeError("Embeddings not configured for DatasetQueryTool")
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
                        "columns": columns,
                        "instruction": f"IMPORTANT: These are ALL {len(results)} results. Use them to formulate your response. Do NOT call this tool again for the same query."
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
            
            # Validate SQL before execution (SQL injection prevention)
            from streamlit_app.tools.sql_validator import SQLValidator
            
            validator = SQLValidator(allowed_tables=[self._table_name])
            is_valid, sanitized_sql, error_msg = validator.validate(sql)
            
            if not is_valid:
                conn.close()
                return {
                    "error": f"SQL validation failed: {error_msg}",
                    "sql_attempted": sql,
                    "natural_query": natural_query,
                    "hint": "Please rephrase your question to request data retrieval only (SELECT queries)."
                }
            
            # Execute validated and sanitized SQL
            result = self._execute_sql(conn, sanitized_sql, natural_query=natural_query)
            
            # Close connection
            conn.close()
            
            # Add the original query to the result
            result["natural_query"] = natural_query
            
            # Add data source information
            if self._dataset_source:
                result["data_source"] = self._dataset_source
            
            return result
            
        except FileNotFoundError as e:
            return {"error": str(e), "natural_query": natural_query}
        except Exception as e:
            return {
                "error": f"Error processing query: {str(e)}",
                "natural_query": natural_query
            }

