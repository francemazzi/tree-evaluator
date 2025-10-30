from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Type

import plotly.express as px
import plotly.graph_objects as go
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


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
    
    def _translate_to_chart_sql(
        self, 
        data_query: str, 
        chart_type: str
    ) -> Dict[str, Any]:
        """Translate natural language data query to SQL optimized for chart type."""
        from datetime import datetime
        import math
        
        current_year = datetime.now().year
        
        prompt = f"""You are a SQL expert for data visualization. Generate a SQL query for creating a {chart_type} chart.

DATABASE SCHEMA:
Table: baumkatogd
Columns: objectid, district, genus_species, plant_year, trunk_circumference, tree_height, crown_diameter, object_street, area_group

IMPORTANT:
1. Current year is {current_year}
2. DBH = trunk_circumference / {math.pi}
3. Age = {current_year} - plant_year
4. Return data optimized for {chart_type} chart
5. For bar/pie charts: return category and count/value columns
6. For line charts: return time-based x-axis and y-axis values
7. For scatter: return two numeric columns
8. For histogram: return the raw values to be binned
9. For box plots: return category and numeric value columns
10. Limit results appropriately (max 50 categories for bar/pie, no limit for distributions)

USER REQUEST: {data_query}
CHART TYPE: {chart_type}

Return a JSON object with:
{{
    "sql": "the SQL query",
    "x_column": "name of x-axis column",
    "y_column": "name of y-axis column (or null for histogram)",
    "suggested_title": "suggested chart title in Italian",
    "x_label": "suggested x-axis label in Italian",
    "y_label": "suggested y-axis label in Italian"
}}

Examples:

Request: "Numero di alberi per distretto"
Chart: bar
Response:
{{
    "sql": "SELECT district, COUNT(*) as count FROM baumkatogd WHERE district IS NOT NULL GROUP BY district ORDER BY district",
    "x_column": "district",
    "y_column": "count",
    "suggested_title": "Numero di Alberi per Distretto",
    "x_label": "Distretto",
    "y_label": "Numero di Alberi"
}}

Request: "Top 10 specie più comuni"
Chart: bar
Response:
{{
    "sql": "SELECT genus_species, COUNT(*) as count FROM baumkatogd WHERE genus_species IS NOT NULL GROUP BY genus_species ORDER BY count DESC LIMIT 10",
    "x_column": "genus_species",
    "y_column": "count",
    "suggested_title": "Top 10 Specie Più Comuni",
    "x_label": "Specie",
    "y_label": "Numero di Alberi"
}}

Request: "Distribuzione età degli alberi"
Chart: histogram
Response:
{{
    "sql": "SELECT ({current_year} - plant_year) as age FROM baumkatogd WHERE plant_year > 0 AND plant_year < {current_year}",
    "x_column": "age",
    "y_column": null,
    "suggested_title": "Distribuzione dell'Età degli Alberi",
    "x_label": "Età (anni)",
    "y_label": "Frequenza"
}}

Now generate the query for: {data_query}"""
        
        if not self._llm:
            raise ValueError("LLM is required. Initialize ChartGenerationTool with an LLM instance.")
        
        response = self._llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Clean up and parse JSON
        response_text = response_text.strip()
        if response_text.startswith('```json'):
            response_text = response_text.split('```json')[1].split('```')[0].strip()
        elif response_text.startswith('```'):
            response_text = response_text.split('```')[1].split('```')[0].strip()
        
        return json.loads(response_text)
    
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
        try:
            # Connect to database
            conn = self._get_connection()
            
            # Translate natural language to SQL
            query_info = self._translate_to_chart_sql(data_query, chart_type)
            
            sql = query_info["sql"]
            x_column = query_info["x_column"]
            y_column = query_info.get("y_column")
            
            # Use custom labels or fall back to suggestions
            final_title = title or query_info["suggested_title"]
            final_x_label = x_label or query_info["x_label"]
            final_y_label = y_label or query_info["y_label"]
            
            # Execute query
            data = self._execute_query(conn, sql)
            conn.close()
            
            if not data:
                return {
                    "success": False,
                    "error": "Nessun dato trovato per la query specificata",
                    "sql_executed": sql
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
                "sql_executed": sql,
                "title": final_title,
                "description": f"Grafico {chart_type} creato con {len(data)} punti dati"
            }
            
        except FileNotFoundError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {
                "success": False,
                "error": f"Errore nella generazione del grafico: {str(e)}"
            }

