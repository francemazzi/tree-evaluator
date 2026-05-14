from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from streamlit_app.tools.chart_queries import get_predefined_chart_query
from streamlit_app.tools.chart_rendering import create_chart
from streamlit_app.tools.chart_sql_translator import translate_to_chart_sql
from streamlit_app.tools.sql_validator import SQLValidator, quote_sql_identifier

ALLOWED_CHART_TYPES = {"bar", "pie", "line", "scatter", "histogram", "box"}


class ChartGenerationInput(BaseModel):
    """Input schema for chart generation tool."""

    chart_type: Literal["bar", "pie", "line", "scatter", "histogram", "box"] = Field(
        description="""Type of chart to generate:
        - bar: for comparing categories (e.g., trees per district)
        - pie: for showing proportions (e.g., species distribution)
        - line: for trends over time (e.g., plantings by year)
        - scatter: for relationships between variables (e.g., height vs circumference)
        - histogram: for distribution of continuous data (e.g., age distribution)
        - box: for statistical summaries (e.g., DBH distribution by species)
        """
    )
    
    data_query: str = Field(
        description="""Natural language description of what data to visualize.
        The system will automatically generate the appropriate SQL query.
        
        Examples:
        - "Numero di alberi per distretto"
        - "Top 10 specie più comuni"
        - "Distribuzione dell'età degli alberi"
        - "Andamento delle piantumazioni negli anni"
        - "Confronto circonferenza per le specie principali"
        """
    )
    
    title: Optional[str] = Field(
        default=None,
        description="Optional custom title for the chart"
    )
    
    x_label: Optional[str] = Field(
        default=None,
        description="Optional label for x-axis"
    )
    
    y_label: Optional[str] = Field(
        default=None,
        description="Optional label for y-axis"
    )


class ChartGenerationTool(BaseTool):
    """Tool to generate interactive charts from tree dataset using natural language."""

    name: str = "generate_chart"
    description: str = """
    Generate interactive charts and visualizations from the Vienna trees dataset.
    
    Automatically generates appropriate SQL queries and creates beautiful visualizations.
    
    Chart types available:
    - bar: comparing categories (e.g., trees per district, top species)
    - pie: showing proportions (e.g., species distribution)
    - line: showing trends over time (e.g., plantings by year)
    - scatter: showing relationships (e.g., height vs circumference)
    - histogram: showing distributions (e.g., age distribution)
    - box: showing statistical summaries (e.g., DBH by species)
    
    Examples:
    - "Crea un grafico a barre dei distretti con più alberi"
    - "Mostra un grafico a torta delle 5 specie più comuni"
    - "Fai un istogramma dell'età degli alberi"
    - "Crea un grafico a linee delle piantumazioni per anno dal 1950"
    - "Mostra un box plot della circonferenza per le specie principali"
    
    Use this tool whenever the user asks to create, generate, visualize, or show charts/graphs.
    """
    args_schema: Type[BaseModel] = ChartGenerationInput

    _db_path: Path
    _table_name: str = "baumkatogd"
    _llm: Any = None
    _fallback_llm: Any = None
    _column_cache: Optional[set[str]] = None

    def __init__(
        self,
        db_path: Optional[Path] = None,
        table_name: Optional[str] = None,
        llm: Any = None,
        fallback_llm: Any = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent / "dataset" / "BAUMKATOGD.db"
        object.__setattr__(self, "_db_path", db_path)
        if table_name:
            object.__setattr__(self, "_table_name", table_name)
        object.__setattr__(self, "_llm", llm)
        object.__setattr__(self, "_fallback_llm", fallback_llm)
        object.__setattr__(self, "_column_cache", None)
    
    def _get_predefined_query(self, data_query: str, chart_type: str) -> Optional[Dict[str, Any]]:
        """Get predefined query for common requests as fallback."""
        return get_predefined_chart_query(
            data_query=data_query,
            chart_type=chart_type,
            table_name=self._table_name,
            columns=self._get_table_columns(),
        )

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

    def _get_table_columns(self) -> set[str]:
        """Return available columns for the configured table."""
        cached = self._column_cache
        if cached is not None:
            return cached

        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            table_identifier = quote_sql_identifier(self._table_name)
            cursor.execute(f"PRAGMA table_info({table_identifier})")
            columns = {str(row[1]) for row in cursor.fetchall()}
            conn.close()
        except Exception:
            columns = set()

        object.__setattr__(self, "_column_cache", columns)
        return columns

    def _get_schema_info(self, conn: sqlite3.Connection) -> str:
        """Get database schema information for the configured table."""
        cursor = conn.cursor()
        cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            (self._table_name,),
        )
        schema = cursor.fetchone()
        return schema[0] if schema else f"Table: {self._table_name}"
    
    def _translate_to_chart_sql(
        self,
        data_query: str,
        chart_type: str,
        schema_info: str,
    ) -> Dict[str, Any]:
        """Translate natural language data query to SQL optimized for chart type."""
        return translate_to_chart_sql(
            llm=self._llm,
            fallback_llm=self._fallback_llm,
            table_name=self._table_name,
            data_query=data_query,
            chart_type=chart_type,
            schema_info=schema_info,
        )

    def _execute_query(self, conn: sqlite3.Connection, sql: str) -> list:
        """Execute SQL query and return results."""
        cursor = conn.cursor()
        cursor.execute(sql)
        
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            result_dict = {}
            for i, col in enumerate(columns):
                result_dict[col] = row[i]
            results.append(result_dict)
        
        return results
    
    def _create_chart(
        self,
        chart_type: str,
        data: list,
        x_column: str,
        y_column: Optional[str],
        title: str,
        x_label: str,
        y_label: str,
    ):
        """Create Plotly chart based on type and data."""
        return create_chart(chart_type, data, x_column, y_column, title, x_label, y_label)

    def _run(
        self,
        chart_type: str,
        data_query: str,
        title: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
    ) -> dict:
        """Generate chart from natural language query."""
        if chart_type not in ALLOWED_CHART_TYPES:
            raise ValueError(f"Unsupported chart type: {chart_type}")

        try:
            # Connect to database
            conn = self._get_connection()
            schema_info = self._get_schema_info(conn)
            
            # Try predefined query first for common patterns
            query_info = self._get_predefined_query(data_query, chart_type)
            
            # If no predefined query, use LLM to translate
            if not query_info:
                try:
                    query_info = self._translate_to_chart_sql(data_query, chart_type, schema_info)
                except json.JSONDecodeError as e:
                    # Try predefined query as fallback
                    query_info = self._get_predefined_query(data_query, chart_type)
                    if not query_info:
                        conn.close()
                        return {
                            "success": False,
                            "error": f"Errore nel parsing della risposta LLM: {str(e)}. Il modello non ha generato un JSON valido.",
                            "data_query": data_query,
                            "chart_type": chart_type,
                            "suggestion": "Riprova con una descrizione più semplice o usa un chart_type diverso."
                        }
                except KeyError as e:
                    # Try predefined query as fallback
                    query_info = self._get_predefined_query(data_query, chart_type)
                    if not query_info:
                        conn.close()
                        return {
                            "success": False,
                            "error": f"Risposta LLM incompleta: manca il campo {str(e)}",
                            "data_query": data_query,
                            "chart_type": chart_type,
                            "suggestion": "La query generata dall'LLM è incompleta. Riprova."
                        }
            
            sql = query_info["sql"]
            x_column = query_info["x_column"]
            y_column = query_info.get("y_column")
            validator = SQLValidator(
                allowed_tables=[self._table_name],
                default_limit=5000,
            )
            is_valid, sanitized_sql, error_msg = validator.validate(sql)
            if not is_valid:
                conn.close()
                return {
                    "success": False,
                    "error": f"SQL validation failed: {error_msg}",
                    "sql_attempted": sql,
                    "data_query": data_query,
                    "suggestion": "Riprova con una richiesta di visualizzazione basata solo sul dataset selezionato."
                }
            
            # Use custom labels or fall back to suggestions
            final_title = title or query_info.get("suggested_title", "Grafico")
            final_x_label = x_label or query_info.get("x_label", "")
            final_y_label = y_label or query_info.get("y_label", "")
            
            # Execute query
            try:
                data = self._execute_query(conn, sanitized_sql)
            except sqlite3.Error as e:
                conn.close()
                return {
                    "success": False,
                    "error": f"Errore nell'esecuzione della query SQL: {str(e)}",
                    "sql_executed": sanitized_sql,
                    "data_query": data_query,
                    "suggestion": "La query SQL generata non è valida. Riprova con una descrizione diversa."
                }
            
            conn.close()
            
            if not data:
                return {
                    "success": False,
                    "error": "Nessun dato trovato per la query specificata",
                    "sql_executed": sanitized_sql,
                    "data_query": data_query,
                    "suggestion": "Prova a modificare i filtri o la query."
                }

            if self._is_effectively_empty_chart_data(data, y_column):
                return {
                    "success": False,
                    "error": "Nessun dato trovato per la query specificata",
                    "sql_executed": sanitized_sql,
                    "data_query": data_query,
                    "suggestion": "La query ha prodotto solo valori nulli o pari a zero."
                }
            
            # Create chart
            fig = self._create_chart(
                chart_type=chart_type,
                data=data,
                x_column=x_column,
                y_column=y_column,
                title=final_title,
                x_label=final_x_label,
                y_label=final_y_label
            )
            
            # Return chart as JSON (Plotly's native format)
            return {
                "success": True,
                "chart_json": fig.to_json(),
                "chart_type": chart_type,
                "data_points": len(data),
                "sql_executed": sanitized_sql,
                "title": final_title,
                "description": f"Grafico {chart_type} creato con successo con {len(data)} punti dati"
            }
            
        except FileNotFoundError as e:
            return {
                "success": False, 
                "error": f"Database non trovato: {str(e)}",
                "suggestion": "Verifica che il database esista ed esegui init_db.py se necessario."
            }
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return {
                "success": False,
                "error": f"Errore imprevisto nella generazione del grafico: {str(e)}",
                "error_type": type(e).__name__,
                "details": error_details,
                "suggestion": "Contatta il supporto tecnico con questi dettagli."
            }

    def _is_effectively_empty_chart_data(
        self,
        data: list[Dict[str, Any]],
        y_column: Optional[str],
    ) -> bool:
        """Treat aggregate-only zero results as empty for visualization purposes."""
        if not data or not y_column:
            return False

        values = [row.get(y_column) for row in data]
        if not values:
            return False

        numeric_values = []
        for value in values:
            if value is None:
                numeric_values.append(0.0)
                continue
            if isinstance(value, (int, float)):
                numeric_values.append(float(value))
                continue
            try:
                numeric_values.append(float(value))
            except (TypeError, ValueError):
                return False

        return all(value == 0 for value in numeric_values)
