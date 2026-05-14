from __future__ import annotations

import logging
import sqlite3
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

logger = logging.getLogger(__name__)


def semantic_filter_results(
    rows: List[tuple],
    columns: List[str],
    natural_query: str,
    embeddings: Any,
    top_k: int = 50,
) -> List[Dict[str, Any]]:
    """Filter large result sets to the most relevant rows using vector search."""
    try:
        if embeddings is None:
            raise RuntimeError("Embeddings not configured for DatasetQueryTool")
        documents = [_row_to_document(row, columns) for row in rows]
        vectorstore = InMemoryVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
        )
        similar_docs = vectorstore.similarity_search(
            query=natural_query,
            k=min(top_k, len(rows)),
        )
        return [doc.metadata for doc in similar_docs]
    except Exception as exc:
        logger.warning("Vector search failed: %s, falling back to truncation", exc)
        return [{columns[i]: row[i] for i in range(len(columns))} for row in rows[:top_k]]


def execute_sql(
    conn: sqlite3.Connection,
    sql: str,
    natural_query: str,
    embeddings: Any,
) -> Dict[str, Any]:
    """Execute SQL query and format results."""
    direct_limit = 100
    vector_search_limit = 50

    try:
        cursor = conn.cursor()
        cursor.execute(sql)

        if not cursor.description:
            return {
                "sql_executed": sql,
                "result": "Query executed successfully",
                "rows_affected": cursor.rowcount,
            }

        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        if len(rows) == 0:
            return {"sql_executed": sql, "result": "No results found", "row_count": 0}

        if len(columns) == 1 and len(rows) == 1:
            return {"sql_executed": sql, "result": rows[0][0], "column": columns[0]}

        total_rows = len(rows)
        if total_rows <= direct_limit:
            results = [format_result_row(columns, row, natural_query) for row in rows]
            return {
                "sql_executed": sql,
                "results": results,
                "row_count": len(results),
                "columns": columns,
                "instruction": (
                    f"IMPORTANT: These are ALL {len(results)} results. Use them to formulate your response. "
                    "Do NOT call this tool again for the same query."
                ),
            }

        filtered_results = semantic_filter_results(
            rows,
            columns,
            natural_query,
            embeddings,
            top_k=vector_search_limit,
        )
        return {
            "sql_executed": sql,
            "results": filtered_results,
            "row_count": len(filtered_results),
            "columns": columns,
            "vector_search_applied": True,
            "total_rows_found": total_rows,
            "info": f"Vector search applied: showing top {len(filtered_results)} most relevant results out of {total_rows} total rows",
        }
    except sqlite3.Error as exc:
        return {"error": f"SQL execution error: {str(exc)}", "sql_attempted": sql}


def format_result_row(
    columns: List[str],
    row: tuple,
    natural_query: str = "",
) -> Dict[str, Any]:
    """Format a SQLite row and add stable aliases for common aggregate outputs."""
    result_dict = {col: row[index] for index, col in enumerate(columns)}

    for key, value in list(result_dict.items()):
        lower_key = key.lower()
        if lower_key not in result_dict:
            result_dict[lower_key] = value

    query_lower = natural_query.lower()
    if "totale" in query_lower and "totale" not in result_dict:
        _add_total_alias(result_dict)

    if "totale" in query_lower and "totale" not in result_dict and len(columns) == 2:
        group_column = columns[0].lower()
        for key, value in result_dict.items():
            if key.lower() != group_column and isinstance(value, (int, float)):
                result_dict["totale"] = value
                break

    return result_dict


def _row_to_document(row: tuple, columns: List[str]) -> Document:
    row_dict = {columns[i]: row[i] for i in range(len(columns))}
    page_content = " | ".join(
        f"{column}: {value}" for column, value in row_dict.items() if value is not None
    )
    return Document(page_content=page_content, metadata=row_dict)


def _add_total_alias(result_dict: Dict[str, Any]) -> None:
    lowered_keys = {key.lower(): key for key in result_dict}
    for candidate in (
        "total",
        "total_sales",
        "total_vendite",
        "totale_vendite",
        "sum_vendite",
        "sum(vendite)",
    ):
        source_key = lowered_keys.get(candidate)
        if source_key is not None:
            result_dict["totale"] = result_dict[source_key]
            return
