from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from langchain_core.vectorstores import InMemoryVectorStore
from pydantic import BaseModel, Field


class SpeciesListQueryInput(BaseModel):
    """Input schema for species list query tool."""

    natural_query: str = Field(
        description="""Natural language question about plant species and taxonomy.

Examples:
- "Dimmi la famiglia e l'ordine di Abelia triflora"
- "Cerca la specie con species_code ABTR"
- "Mostrami 10 specie del genere Abies"
- "Quali specie sono della famiglia Pinaceae?"
- "Che significa leaf_type e quali valori ci sono per Abies?"
"""
    )


class SpeciesListQueryTool(BaseTool):
    """Tool to query the species_list dataset using natural language translated to SQLite SQL."""

    name: str = "query_species_list"
    description: str = """
Query a plant species list dataset (taxonomy + traits) using natural language.
The system translates your question into SQLite SQL and executes it.

Database schema (table: species_list):
- genus_name (TEXT): Genus (e.g., "Abelia")
- species_name (TEXT): Species epithet (e.g., "triflora")
- genus_species (TEXT): Full binomial (genus + species)
- synonyms (TEXT): Synonyms (optional)
- family (TEXT): Botanical family
- taxonomic_order (TEXT): Botanical order
- taxonomic_class (TEXT): Botanical class
- common_name (TEXT): Common name (optional)
- species_code (TEXT): Short code (often unique)
- growth_form (TEXT): Growth form (e.g., Tree, Shrub)
- percent_leaf_type (TEXT): Leaf type percent/category (as in source)
- leaf_type (TEXT): Leaf type / group (as in source)
- growth_rate (TEXT): Growth rate (optional)
- longevity (TEXT): Longevity category (optional)
- height_at_maturity_feet (INTEGER): Height at maturity (feet), if available

Use this tool to provide botanical context: taxonomy, traits, lookups by genus/species/code/common name.
Always ask for a smaller subset (LIMIT) if returning many rows.
"""

    args_schema: Type[BaseModel] = SpeciesListQueryInput

    _db_path: Path
    _table_name: str = "species_list"
    _llm: Any = None
    _fallback_llm: Any = None
    _embeddings: Any = None

    def __init__(
        self,
        db_path: Optional[Path] = None,
        table_name: str = "species_list",
        llm: Any = None,
        fallback_llm: Any = None,
        embeddings: Any = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent / "dataset" / "species_list.db"
        object.__setattr__(self, "_db_path", db_path)
        object.__setattr__(self, "_table_name", table_name)
        object.__setattr__(self, "_llm", llm)
        object.__setattr__(self, "_fallback_llm", fallback_llm)
        object.__setattr__(self, "_embeddings", embeddings)

    def _get_connection(self) -> sqlite3.Connection:
        if not self._db_path.exists():
            raise FileNotFoundError(
                f"Database not found at {self._db_path}. "
                f"Run 'python dataset/init_species_list_db.py --force' to create it."
            )
        conn = sqlite3.connect(self._db_path.as_posix())
        conn.row_factory = sqlite3.Row
        return conn

    def _get_schema_info(self, conn: sqlite3.Connection) -> str:
        cursor = conn.cursor()
        cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{self._table_name}'")
        schema = cursor.fetchone()
        return schema[0] if schema else "Schema not found"

    def _translate_to_sql(self, natural_query: str, schema_info: str) -> str:
        prompt = f"""You are a SQL expert. Translate the user's natural language question into a SQLite query.

DATABASE SCHEMA:
{schema_info}

IMPORTANT NOTES:
1. Table name is: {self._table_name}
2. The full binomial name is stored in genus_species, but you can also combine genus_name and species_name.
3. Use case-insensitive LIKE for text searches (e.g., genus_species LIKE '%Abies%').
4. **ALWAYS USE LIMIT** - never return all rows. Use LIMIT 20 for lists, LIMIT 1 for single lookups, LIMIT 50 max.
5. Prefer selecting only the needed columns (avoid SELECT *).
6. If the user asks "quali valori possibili", you can query DISTINCT values with LIMIT.

Return ONLY the SQL query, nothing else. No explanations, no markdown, just the SQL.

Examples:
Question: "Dimmi la famiglia e l'ordine di Abelia triflora"
SQL: SELECT genus_species, family, taxonomic_order, taxonomic_class FROM {self._table_name} WHERE genus_species LIKE '%Abelia triflora%' LIMIT 5

Question: "Cerca la specie con species_code ABTR"
SQL: SELECT genus_species, family, taxonomic_order, common_name, growth_form FROM {self._table_name} WHERE species_code = 'ABTR' LIMIT 1

Question: "Mostrami 10 specie del genere Abies"
SQL: SELECT genus_species, species_code, family, leaf_type, growth_form FROM {self._table_name} WHERE genus_name = 'Abies' ORDER BY species_name ASC LIMIT 10

Now translate this question:
{natural_query}"""

        if not self._llm:
            raise ValueError("LLM is required for natural language to SQL translation.")

        def _invoke_with_fallback() -> Any:
            try:
                return self._llm.invoke(prompt)
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e) or "request too large" in str(e).lower():
                    if self._fallback_llm:
                        return self._fallback_llm.invoke(prompt)
                raise

        response = _invoke_with_fallback()
        sql = response.content if hasattr(response, "content") else str(response)
        sql = sql.strip()
        if sql.startswith("```"):
            lines = sql.split("\n")
            if lines and lines[0].startswith("```"):
                sql = "\n".join(lines[1:])
        if sql.endswith("```"):
            sql = sql.rsplit("\n```", 1)[0]
        return sql.strip()

    def _init_embeddings(self) -> Any:
        if self._embeddings is None:
            raise RuntimeError("Embeddings not configured for SpeciesListQueryTool")
        return self._embeddings

    def _semantic_filter_results(
        self,
        rows: List[tuple],
        columns: List[str],
        natural_query: str,
        top_k: int = 50,
    ) -> List[Dict[str, Any]]:
        try:
            embeddings = self._init_embeddings()
            documents: List[Document] = []
            for row in rows:
                row_dict = {columns[i]: row[i] for i in range(len(columns))}
                text_parts = [f"{k}: {v}" for k, v in row_dict.items() if v is not None]
                documents.append(Document(page_content=" | ".join(text_parts), metadata=row_dict))

            vectorstore = InMemoryVectorStore.from_documents(documents=documents, embedding=embeddings)
            similar_docs = vectorstore.similarity_search(query=natural_query, k=min(top_k, len(rows)))
            return [doc.metadata for doc in similar_docs]
        except Exception:
            return [{columns[i]: row[i] for i in range(len(columns))} for row in rows[:top_k]]

    def _execute_sql(self, conn: sqlite3.Connection, sql: str, natural_query: str = "") -> Dict[str, Any]:
        DIRECT_LIMIT = 100
        VECTOR_SEARCH_LIMIT = 50

        cursor = conn.cursor()
        cursor.execute(sql)

        if cursor.description:
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()

            if not rows:
                return {"sql_executed": sql, "result": "No results found", "row_count": 0}

            if len(columns) == 1 and len(rows) == 1:
                return {"sql_executed": sql, "result": rows[0][0], "column": columns[0]}

            total_rows = len(rows)
            if total_rows <= DIRECT_LIMIT:
                results: List[Dict[str, Any]] = []
                for row in rows:
                    results.append({columns[i]: row[i] for i in range(len(columns))})
                return {"sql_executed": sql, "results": results, "row_count": len(results), "columns": columns}

            filtered = self._semantic_filter_results(rows, columns, natural_query, top_k=VECTOR_SEARCH_LIMIT)
            return {
                "sql_executed": sql,
                "results": filtered,
                "row_count": len(filtered),
                "columns": columns,
                "vector_search_applied": True,
                "total_rows_found": total_rows,
                "info": f"Vector search applied: showing top {len(filtered)} most relevant results out of {total_rows} total rows",
            }

        return {"sql_executed": sql, "result": "Query executed successfully", "rows_affected": cursor.rowcount}

    def _run(self, natural_query: str) -> dict:
        try:
            conn = self._get_connection()
            schema_info = self._get_schema_info(conn)
            sql = self._translate_to_sql(natural_query, schema_info)
            result = self._execute_sql(conn, sql, natural_query=natural_query)
            conn.close()
            result["natural_query"] = natural_query
            return result
        except FileNotFoundError as e:
            return {"error": str(e), "natural_query": natural_query}
        except sqlite3.Error as e:
            return {"error": f"SQL execution error: {str(e)}", "natural_query": natural_query}
        except Exception as e:
            return {"error": f"Error processing query: {str(e)}", "natural_query": natural_query}


