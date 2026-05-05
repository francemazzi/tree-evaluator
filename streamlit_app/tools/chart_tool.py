from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Type

import plotly.express as px
import plotly.graph_objects as go
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from streamlit_app.tools.sql_validator import SQLValidator, quote_sql_identifier

logger = logging.getLogger(__name__)

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
        data_query_lower = data_query.lower()
        table = self._table_name
        columns = self._get_table_columns()
        requested_limit = self._extract_requested_limit(data_query)
        
        # Species composition/distribution queries
        if (
            "genus_species" in columns
            and chart_type == "pie"
            and any(keyword in data_query_lower for keyword in ["specie", "species", "composizione", "distribuzione"])
        ):
            return {
                "sql": f"""WITH species_counts AS (
                    SELECT genus_species, COUNT(*) AS count
                    FROM {table}
                    WHERE genus_species IS NOT NULL AND genus_species <> ''
                    GROUP BY genus_species
                ),
                ranked AS (
                    SELECT genus_species, count, ROW_NUMBER() OVER (ORDER BY count DESC) AS rn
                    FROM species_counts
                )
                SELECT
                    CASE WHEN rn <= {requested_limit} THEN genus_species ELSE 'Altro' END AS category,
                    SUM(count) AS count
                FROM ranked
                GROUP BY CASE WHEN rn <= {requested_limit} THEN genus_species ELSE 'Altro' END
                ORDER BY
                    CASE WHEN category = 'Altro' THEN 1 ELSE 0 END,
                    count DESC""",
                "x_column": "category",
                "y_column": "count",
                "suggested_title": "Composizione delle Specie di Alberi",
                "x_label": "Specie",
                "y_label": "Numero di Alberi"
            }

        if (
            "genus_species" in columns
            and chart_type == "bar"
            and any(keyword in data_query_lower for keyword in ["specie", "species", "top", "comuni", "diffuse"])
        ):
            return {
                "sql": f"""SELECT genus_species AS category, COUNT(*) AS count
                         FROM {table}
                         WHERE genus_species IS NOT NULL AND genus_species <> ''
                         GROUP BY genus_species
                         ORDER BY count DESC
                         LIMIT {requested_limit}""",
                "x_column": "category",
                "y_column": "count",
                "suggested_title": f"Top {requested_limit} Specie Piu' Comuni",
                "x_label": "Specie",
                "y_label": "Numero di Alberi"
            }
        
        # District distribution
        if "district" in columns and any(keyword in data_query_lower for keyword in ["distretto", "district", "quartiere", "municipio"]):
            if chart_type == "pie":
                sql = f"""SELECT district AS category, COUNT(*) as count
                         FROM {table}
                         WHERE district IS NOT NULL AND district <> ''
                         GROUP BY district
                         ORDER BY count DESC"""
            else:  # bar
                sql = f"""SELECT district AS category, COUNT(*) as count
                         FROM {table}
                         WHERE district IS NOT NULL AND district <> ''
                         GROUP BY district
                         ORDER BY count DESC"""
            
            return {
                "sql": sql,
                "x_column": "category",
                "y_column": "count",
                "suggested_title": "Distribuzione Alberi per Distretto",
                "x_label": "Distretto",
                "y_label": "Numero di Alberi"
            }

        if "plant_year" in columns and chart_type == "histogram" and any(keyword in data_query_lower for keyword in ["eta", "età", "age"]):
            from datetime import datetime
            current_year = datetime.now().year
            return {
                "sql": f"""SELECT ({current_year} - plant_year) as age
                         FROM {table}
                         WHERE plant_year IS NOT NULL AND plant_year > 0 AND plant_year < {current_year}""",
                "x_column": "age",
                "y_column": None,
                "suggested_title": "Distribuzione dell'Eta' degli Alberi",
                "x_label": "Eta' (anni)",
                "y_label": "Frequenza"
            }
        
        return None

    def _extract_requested_limit(self, data_query: str, default: int = 10, maximum: int = 50) -> int:
        """Extract an explicit top-N limit from a natural-language chart request."""
        query = data_query.lower()
        patterns = [
            r"\btop\s+(\d+)\b",
            r"\bprim(?:e|i)\s+(\d+)\b",
            r"\b(\d+)\s+(?:specie|species|categorie|distretti|district|municipi)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                return max(1, min(int(match.group(1)), maximum))
        return default

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
        from datetime import datetime
        import math
        
        current_year = datetime.now().year
        
        prompt = f"""You are a SQL expert for data visualization. Generate a SQL query for creating a {chart_type} chart.

DATABASE SCHEMA:
{schema_info}

IMPORTANT RULES:
1. Current year is {current_year}
2. Table name is: {self._table_name}
3. DBH = trunk_circumference / {math.pi} when trunk_circumference exists; otherwise use a diameter column if present.
4. Age = {current_year} - plant_year when plant_year exists.
5. Return data optimized for {chart_type} chart
6. For bar/pie charts: return category and count/value columns named exactly "category" and "count"
7. For line charts: return time-based x-axis and y-axis values
8. For scatter: return two numeric columns
9. For histogram: return the raw values to be binned
10. For box plots: return category and numeric value columns
11. For pie charts showing species: ALWAYS use TOP 15 + "Altro" pattern when genus_species exists
12. ALWAYS filter out NULL and empty string values with: WHERE column IS NOT NULL AND column <> ''
13. For pie/bar charts: limit to max 15-20 main categories, group rest as "Altro"
14. Use ONLY the table {self._table_name}; do not reference other tables.

USER REQUEST: {data_query}
CHART TYPE: {chart_type}

Return ONLY a valid JSON object with:
{{
    "sql": "the SQL query",
    "x_column": "name of x-axis column",
    "y_column": "name of y-axis column (or null for histogram)",
    "suggested_title": "suggested chart title in Italian",
    "x_label": "suggested x-axis label in Italian",
    "y_label": "suggested y-axis label in Italian"
}}

CRITICAL EXAMPLES:

Request: "Composizione specie" OR "distribuzione specie" OR "specie di piante"
Chart: pie
Response:
{{
    "sql": "WITH species_counts AS (SELECT genus_species, COUNT(*) AS count FROM {self._table_name} WHERE genus_species IS NOT NULL AND genus_species <> '' GROUP BY genus_species), ranked AS (SELECT genus_species, count, ROW_NUMBER() OVER (ORDER BY count DESC) AS rn FROM species_counts) SELECT CASE WHEN rn <= 15 THEN genus_species ELSE 'Altro' END AS category, SUM(count) AS count FROM ranked GROUP BY CASE WHEN rn <= 15 THEN genus_species ELSE 'Altro' END ORDER BY CASE WHEN category = 'Altro' THEN 1 ELSE 0 END, count DESC",
    "x_column": "category",
    "y_column": "count",
    "suggested_title": "Composizione delle Specie di Alberi",
    "x_label": "Specie",
    "y_label": "Numero di Alberi"
}}

Request: "Numero di alberi per distretto"
Chart: bar
Response:
{{
    "sql": "SELECT district AS category, COUNT(*) as count FROM {self._table_name} WHERE district IS NOT NULL AND district <> '' GROUP BY district ORDER BY count DESC",
    "x_column": "category",
    "y_column": "count",
    "suggested_title": "Numero di Alberi per Distretto",
    "x_label": "Distretto",
    "y_label": "Numero di Alberi"
}}

Request: "Top 10 specie più comuni"
Chart: bar
Response:
{{
    "sql": "SELECT genus_species AS category, COUNT(*) as count FROM {self._table_name} WHERE genus_species IS NOT NULL AND genus_species <> '' GROUP BY genus_species ORDER BY count DESC LIMIT 10",
    "x_column": "category",
    "y_column": "count",
    "suggested_title": "Top 10 Specie Più Comuni",
    "x_label": "Specie",
    "y_label": "Numero di Alberi"
}}

Request: "Distribuzione età degli alberi"
Chart: histogram
Response:
{{
    "sql": "SELECT ({current_year} - plant_year) as age FROM {self._table_name} WHERE plant_year > 0 AND plant_year < {current_year}",
    "x_column": "age",
    "y_column": null,
    "suggested_title": "Distribuzione dell'Età degli Alberi",
    "x_label": "Età (anni)",
    "y_label": "Frequenza"
}}

Now generate the query for: {data_query}

Remember: Return ONLY valid JSON, no markdown, no explanation."""
        
        if not self._llm:
            raise ValueError("LLM is required. Initialize ChartGenerationTool with an LLM instance.")

        def _invoke_with_fallback() -> Any:
            try:
                return self._llm.invoke(prompt)
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e) or "Request too large" in str(e):
                    try:
                        if self._fallback_llm:
                            return self._fallback_llm.invoke(prompt)
                    except Exception:
                        pass
                raise

        response = _invoke_with_fallback()
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Clean up and parse JSON - improved robustness
        response_text = response_text.strip()
        
        # Remove markdown code blocks
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0].strip()
        elif '```' in response_text:
            # Try to extract JSON between any code blocks
            parts = response_text.split('```')
            for part in parts:
                part = part.strip()
                if part and (part.startswith('{') or part.startswith('[')):
                    response_text = part
                    break
        
        # Remove any leading/trailing text that's not JSON
        if '{' in response_text:
            start_idx = response_text.index('{')
            end_idx = response_text.rindex('}') + 1
            response_text = response_text[start_idx:end_idx]
        
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError as e:
            # Log the problematic response for debugging
            logger.warning("Failed to parse LLM response: %s...", response_text[:200])
            logger.warning("JSON error: %s", e)
            raise
        
        # Validate required fields
        required_fields = ["sql", "x_column", "suggested_title", "x_label", "y_label"]
        missing_fields = [f for f in required_fields if f not in parsed]
        if missing_fields:
            raise KeyError(f"Missing required fields: {', '.join(missing_fields)}")
        
        return parsed
    
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
        y_label: str
    ) -> go.Figure:
        """Create Plotly chart based on type and data."""
        
        if not data:
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="Nessun dato disponibile per questo grafico",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Extract data for plotting
        x_data = [row[x_column] for row in data]
        y_data = [row[y_column] for row in data] if y_column else None
        
        # Create appropriate chart
        if chart_type == "bar":
            fig = go.Figure(data=[
                go.Bar(x=x_data, y=y_data, marker_color='#2E7D32')
            ])
            
        elif chart_type == "pie":
            fig = go.Figure(data=[
                go.Pie(labels=x_data, values=y_data, hole=0.3)
            ])
            
        elif chart_type == "line":
            fig = go.Figure(data=[
                go.Scatter(x=x_data, y=y_data, mode='lines+markers', 
                          line=dict(color='#2E7D32', width=2),
                          marker=dict(size=6))
            ])
            
        elif chart_type == "scatter":
            fig = go.Figure(data=[
                go.Scatter(x=x_data, y=y_data, mode='markers',
                          marker=dict(size=8, color='#2E7D32', opacity=0.6))
            ])
            
        elif chart_type == "histogram":
            fig = go.Figure(data=[
                go.Histogram(x=x_data, marker_color='#2E7D32', nbinsx=30)
            ])
            
        elif chart_type == "box":
            # For box plot, we need to group by category
            # x_column is the category, y_column is the value
            categories = list(set(x_data))
            fig = go.Figure()
            for cat in categories:
                values = [row[y_column] for row in data if row[x_column] == cat]
                fig.add_trace(go.Box(y=values, name=str(cat)))
        
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")
        
        # Update layout
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center', font=dict(size=18)),
            xaxis_title=x_label,
            yaxis_title=y_label,
            template="plotly_white",
            hovermode='closest',
            height=500
        )
        
        return fig

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
