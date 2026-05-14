from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from streamlit_app.tools.dataset_results import (
    execute_sql,
    format_result_row,
    semantic_filter_results,
)
from streamlit_app.tools.dataset_sql_translator import translate_to_sql

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
        conn.execute("PRAGMA query_only = ON")
        conn.row_factory = sqlite3.Row
        return conn
    
    def _get_schema_info(self, conn: sqlite3.Connection) -> str:
        """Get database schema information."""
        cursor = conn.cursor()
        cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            (self._table_name,),
        )
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
        return translate_to_sql(
            llm=self._llm,
            fallback_llm=self._fallback_llm,
            table_name=self._table_name,
            user_description=self._user_description,
            natural_query=natural_query,
            schema_info=schema_info,
        )

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
        top_k: int = 50,
    ) -> List[Dict[str, Any]]:
        """Use vector search to filter large result sets to most relevant items."""
        return semantic_filter_results(rows, columns, natural_query, self._embeddings, top_k)

    def _format_result_row(
        self,
        columns: List[str],
        row: tuple,
        natural_query: str = "",
    ) -> Dict[str, Any]:
        """Format a SQLite row and add stable aliases for common aggregate outputs."""
        return format_result_row(columns, row, natural_query)

    def _execute_sql(self, conn: sqlite3.Connection, sql: str, natural_query: str = "") -> Dict[str, Any]:
        """Execute SQL query and format results."""
        return execute_sql(conn, sql, natural_query, self._embeddings)

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
