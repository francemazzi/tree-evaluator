from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Type

import folium
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from streamlit_app.tools.map_rendering import create_map, create_popup_html
from streamlit_app.tools.map_sql_translator import translate_to_map_sql
from streamlit_app.tools.sql_validator import SQLValidator, quote_sql_identifier


class MapGenerationInput(BaseModel):
    """Input schema for map generation tool."""

    map_type: Literal["markers", "cluster", "heatmap"] = Field(
        description="""Type of map to generate:
        - markers: Individual markers for each point (best for <500 points)
        - cluster: Clustered markers that group nearby points (best for large datasets)
        - heatmap: Heat map showing density of points
        """
    )
    
    data_query: str = Field(
        description="""Natural language description of what data to show on the map.
        The system will automatically generate the appropriate SQL query to get coordinates.
        
        Examples:
        - "Mostra tutti i tigli (Tilia) sulla mappa"
        - "Mappa degli alberi nel distretto 5"
        - "Visualizza la distribuzione del genere Acer"
        - "Mostra gli alberi con diametro maggiore di 50cm"
        - "Mappa degli alberi piantati dopo il 2010"
        """
    )
    
    title: Optional[str] = Field(
        default=None,
        description="Optional custom title for the map"
    )
    
    color: Optional[str] = Field(
        default="#2E7D32",
        description="Color for markers/heatmap (hex code like #2E7D32 or color name)"
    )
    
    max_points: Optional[int] = Field(
        default=5000,
        description="Maximum number of points to display (to avoid browser slowdown)"
    )


class MapGenerationTool(BaseTool):
    """Tool to generate interactive maps from tree dataset using natural language."""

    name: str = "generate_map"
    description: str = """
    Generate interactive maps showing tree locations from the dataset.
    
    Automatically queries the database and creates beautiful map visualizations.
    
    Map types available:
    - markers: Individual markers for each tree (best for small selections)
    - cluster: Clustered markers that group nearby trees (best for large datasets)
    - heatmap: Heat map showing density/distribution of trees
    
    Examples:
    - "Mostra una mappa con tutti i tigli (Tilia)"
    - "Crea una heatmap della distribuzione degli alberi a Milano"
    - "Visualizza su mappa gli alberi del municipio 3"
    - "Mostra i cluster degli alberi di Acer platanoides"
    - "Crea una mappa con gli alberi più grandi (diametro > 60cm)"
    
    Use this tool whenever the user asks to:
    - Show trees on a map
    - Visualize tree distribution geographically
    - Create a map of specific tree species/genera
    - Display spatial distribution of trees
    
    IMPORTANT: This tool requires GPS coordinates (latitude, longitude) in the dataset.
    The Milano dataset has coordinates, Vienna dataset does NOT have coordinates.
    """
    args_schema: Type[BaseModel] = MapGenerationInput

    _db_path: Path
    _llm: Any = None
    _fallback_llm: Any = None
    _table_name: str = "milano_trees"  # Default to Milano which has GPS
    _lat_column: str = "latitude"
    _lon_column: str = "longitude"

    def __init__(
        self, 
        db_path: Optional[Path] = None, 
        table_name: Optional[str] = None,
        lat_column: str = "latitude",
        lon_column: str = "longitude",
        llm: Any = None,
        fallback_llm: Any = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent / "dataset" / "dataset_milano.db"
        object.__setattr__(self, "_db_path", db_path)
        object.__setattr__(self, "_llm", llm)
        object.__setattr__(self, "_fallback_llm", fallback_llm)
        
        if table_name:
            object.__setattr__(self, "_table_name", table_name)
        
        object.__setattr__(self, "_lat_column", lat_column)
        object.__setattr__(self, "_lon_column", lon_column)

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        if not self._db_path.exists():
            raise FileNotFoundError(
                f"Database not found at {self._db_path}. "
            )
        
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA query_only = ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _get_schema_info(self, conn: sqlite3.Connection) -> str:
        """Get database schema information for the configured table."""
        cursor = conn.cursor()
        cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            (self._table_name,),
        )
        schema = cursor.fetchone()
        return schema[0] if schema else f"Table: {self._table_name}"

    def _translate_to_map_sql(
        self,
        data_query: str,
        max_points: int,
        schema_info: str,
    ) -> Dict[str, Any]:
        """Translate natural language data query to SQL for map visualization."""
        return translate_to_map_sql(
            llm=self._llm,
            fallback_llm=self._fallback_llm,
            table_name=self._table_name,
            lat_column=self._lat_column,
            lon_column=self._lon_column,
            data_query=data_query,
            max_points=max_points,
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

    def _create_popup_html(self, row: Dict[str, Any], popup_columns: list) -> str:
        """Create HTML content for marker popup."""
        return create_popup_html(row, popup_columns)

    def _create_map(
        self,
        map_type: str,
        data: list,
        title: str,
        color: str,
        popup_columns: list,
        center_lat: float,
        center_lon: float,
        zoom: int,
    ) -> folium.Map:
        """Create Folium map based on type and data."""
        return create_map(
            map_type=map_type,
            data=data,
            title=title,
            color=color,
            popup_columns=popup_columns,
            center_lat=center_lat,
            center_lon=center_lon,
            zoom=zoom,
            lat_column=self._lat_column,
            lon_column=self._lon_column,
        )

    def _run(
        self,
        map_type: str,
        data_query: str,
        title: Optional[str] = None,
        color: Optional[str] = "#2E7D32",
        max_points: Optional[int] = 5000,
    ) -> dict:
        """Generate map from natural language query."""
        try:
            # Check if database has coordinates
            conn = self._get_connection()
            schema_info = self._get_schema_info(conn)
            
            # Verify coordinate columns exist
            cursor = conn.cursor()
            table_identifier = quote_sql_identifier(self._table_name)
            cursor.execute(f"PRAGMA table_info({table_identifier})")
            columns = [row[1] for row in cursor.fetchall()]
            
            if self._lat_column not in columns or self._lon_column not in columns:
                conn.close()
                return {
                    "success": False,
                    "error": f"Il dataset selezionato non contiene coordinate GPS ({self._lat_column}, {self._lon_column}). "
                            "Le mappe sono disponibili solo per il dataset Milano che include coordinate GPS."
                }
            
            # Translate natural language to SQL
            effective_max_points = min(max_points or 5000, 5000)
            query_info = self._translate_to_map_sql(data_query, effective_max_points, schema_info)
            
            sql = query_info["sql"]
            popup_columns = query_info.get("popup_columns", ["genus_species"])
            center_lat = query_info.get("center_lat", 45.4642)
            center_lon = query_info.get("center_lon", 9.19)
            zoom = query_info.get("zoom", 12)
            
            # Use custom title or fall back to suggestion
            final_title = title or query_info["suggested_title"]

            validator = SQLValidator(
                allowed_tables=[self._table_name],
                default_limit=effective_max_points,
            )
            is_valid, sanitized_sql, error_msg = validator.validate(sql)
            if not is_valid:
                conn.close()
                return {
                    "success": False,
                    "error": f"SQL validation failed: {error_msg}",
                    "sql_attempted": sql,
                    "data_query": data_query,
                }
            
            # Execute query
            data = self._execute_query(conn, sanitized_sql)
            conn.close()
            
            if not data:
                return {
                    "success": False,
                    "error": "Nessun dato trovato per la query specificata",
                    "sql_executed": sanitized_sql
                }
            
            # Create map
            m = self._create_map(
                map_type=map_type,
                data=data,
                title=final_title,
                color=color or "#2E7D32",
                popup_columns=popup_columns,
                center_lat=center_lat,
                center_lon=center_lon,
                zoom=zoom
            )
            
            # Get map HTML
            map_html = m._repr_html_()
            
            return {
                "success": True,
                "map_html": map_html,
                "map_type": map_type,
                "data_points": len(data),
                "sql_executed": sanitized_sql,
                "title": final_title,
                "center": {"lat": center_lat, "lon": center_lon},
                "zoom": zoom,
                "description": f"Mappa {map_type} creata con {len(data)} punti"
            }
            
        except FileNotFoundError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {
                "success": False,
                "error": f"Errore nella generazione della mappa: {str(e)}"
            }
