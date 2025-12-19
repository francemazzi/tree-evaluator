from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Type

import folium
from folium.plugins import HeatMap, MarkerCluster
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


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
        conn.row_factory = sqlite3.Row
        return conn

    def _translate_to_map_sql(
        self, 
        data_query: str, 
        max_points: int
    ) -> Dict[str, Any]:
        """Translate natural language data query to SQL for map visualization."""
        from datetime import datetime
        import math
        
        current_year = datetime.now().year
        
        prompt = f"""You are a SQL expert for map visualization. Generate a SQL query to get tree locations for a map.

DATABASE SCHEMA:
Table: {self._table_name}
Columns: _id, district, genere, specie, varieta, genus_species, trunk_diameter_cm, crown_diameter_m, height_m, street, plant_year, longitude, latitude

IMPORTANT:
1. Current year is {current_year}
2. ALWAYS select {self._lat_column} and {self._lon_column} columns - these are REQUIRED for the map
3. Filter out NULL coordinates: WHERE {self._lat_column} IS NOT NULL AND {self._lon_column} IS NOT NULL
4. Limit results to {max_points} points maximum
5. For species searches, use LIKE with % wildcards (case-insensitive)
6. Common genus keywords: Acer (acero), Tilia (tiglio), Quercus (quercia), Fraxinus (frassino), Prunus
7. The column 'genere' contains genus (e.g., 'Tilia', 'Acer'), 'specie' contains species name
8. For filtering, prefer using 'genere' column for genus and 'genus_species' for full name

USER REQUEST: {data_query}

Return a JSON object with:
{{
    "sql": "the SQL query - MUST include latitude and longitude columns",
    "suggested_title": "suggested map title in Italian",
    "popup_columns": ["list", "of", "columns", "to", "show", "in", "popup"],
    "center_lat": estimated center latitude (default: 45.4642 for Milano),
    "center_lon": estimated center longitude (default: 9.19 for Milano),
    "zoom": suggested zoom level (10-15, higher = more zoomed in)
}}

Examples:

Request: "Mostra i tigli sulla mappa"
Response:
{{
    "sql": "SELECT {self._lat_column}, {self._lon_column}, genere, specie, genus_species, trunk_diameter_cm, street FROM {self._table_name} WHERE genere LIKE '%Tilia%' AND {self._lat_column} IS NOT NULL AND {self._lon_column} IS NOT NULL LIMIT {max_points}",
    "suggested_title": "Distribuzione dei Tigli (Tilia) a Milano",
    "popup_columns": ["genus_species", "trunk_diameter_cm", "street"],
    "center_lat": 45.4642,
    "center_lon": 9.19,
    "zoom": 12
}}

Request: "Alberi del municipio 3"
Response:
{{
    "sql": "SELECT {self._lat_column}, {self._lon_column}, genere, specie, genus_species, trunk_diameter_cm, street FROM {self._table_name} WHERE district = 3 AND {self._lat_column} IS NOT NULL AND {self._lon_column} IS NOT NULL LIMIT {max_points}",
    "suggested_title": "Alberi del Municipio 3 di Milano",
    "popup_columns": ["genus_species", "trunk_diameter_cm", "street"],
    "center_lat": 45.48,
    "center_lon": 9.22,
    "zoom": 13
}}

Request: "Distribuzione degli alberi"
Response:
{{
    "sql": "SELECT {self._lat_column}, {self._lon_column}, genere, specie FROM {self._table_name} WHERE {self._lat_column} IS NOT NULL AND {self._lon_column} IS NOT NULL LIMIT {max_points}",
    "suggested_title": "Distribuzione degli Alberi a Milano",
    "popup_columns": ["genere", "specie"],
    "center_lat": 45.4642,
    "center_lon": 9.19,
    "zoom": 11
}}

Now generate the query for: {data_query}"""
        
        if not self._llm:
            raise ValueError("LLM is required. Initialize MapGenerationTool with an LLM instance.")

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

    def _create_popup_html(self, row: Dict[str, Any], popup_columns: list) -> str:
        """Create HTML content for marker popup."""
        html_parts = []
        for col in popup_columns:
            if col in row and row[col] is not None:
                # Format column name nicely
                col_name = col.replace('_', ' ').title()
                value = row[col]
                html_parts.append(f"<b>{col_name}:</b> {value}")
        
        return "<br>".join(html_parts) if html_parts else "Albero"

    def _create_map(
        self,
        map_type: str,
        data: list,
        title: str,
        color: str,
        popup_columns: list,
        center_lat: float,
        center_lon: float,
        zoom: int
    ) -> folium.Map:
        """Create Folium map based on type and data."""
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom,
            tiles='CartoDB positron'
        )
        
        # Add title
        title_html = f'''
        <div style="position: fixed; 
                    top: 10px; left: 50px; width: auto; 
                    background-color: white; 
                    border-radius: 8px;
                    border: 2px solid #2E7D32;
                    padding: 10px 20px;
                    z-index: 9999;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.3);">
            <h4 style="margin: 0; color: #2E7D32;">🌳 {title}</h4>
            <p style="margin: 5px 0 0 0; font-size: 12px; color: #666;">
                Punti visualizzati: {len(data):,}
            </p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        if not data:
            # Empty map with message
            folium.Marker(
                location=[center_lat, center_lon],
                popup="Nessun dato trovato",
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
            return m
        
        # Extract coordinates
        lat_col = self._lat_column
        lon_col = self._lon_column
        
        if map_type == "markers":
            # Individual markers
            for row in data:
                lat = row.get(lat_col)
                lon = row.get(lon_col)
                if lat is not None and lon is not None:
                    popup_html = self._create_popup_html(row, popup_columns)
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=5,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.7,
                        popup=folium.Popup(popup_html, max_width=300)
                    ).add_to(m)
                    
        elif map_type == "cluster":
            # Clustered markers
            cluster = MarkerCluster()
            for row in data:
                lat = row.get(lat_col)
                lon = row.get(lon_col)
                if lat is not None and lon is not None:
                    popup_html = self._create_popup_html(row, popup_columns)
                    folium.Marker(
                        location=[lat, lon],
                        popup=folium.Popup(popup_html, max_width=300),
                        icon=folium.Icon(color='green', icon='tree-deciduous', prefix='glyphicon')
                    ).add_to(cluster)
            cluster.add_to(m)
            
        elif map_type == "heatmap":
            # Heatmap
            heat_data = []
            for row in data:
                lat = row.get(lat_col)
                lon = row.get(lon_col)
                if lat is not None and lon is not None:
                    heat_data.append([lat, lon])
            
            if heat_data:
                HeatMap(
                    heat_data,
                    radius=15,
                    blur=20,
                    min_opacity=0.3,
                    max_zoom=18
                ).add_to(m)
        
        return m

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
            
            # Verify coordinate columns exist
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({self._table_name})")
            columns = [row[1] for row in cursor.fetchall()]
            
            if self._lat_column not in columns or self._lon_column not in columns:
                return {
                    "success": False,
                    "error": f"Il dataset selezionato non contiene coordinate GPS ({self._lat_column}, {self._lon_column}). "
                            "Le mappe sono disponibili solo per il dataset Milano che include coordinate GPS."
                }
            
            # Translate natural language to SQL
            query_info = self._translate_to_map_sql(data_query, max_points or 5000)
            
            sql = query_info["sql"]
            popup_columns = query_info.get("popup_columns", ["genus_species"])
            center_lat = query_info.get("center_lat", 45.4642)
            center_lon = query_info.get("center_lon", 9.19)
            zoom = query_info.get("zoom", 12)
            
            # Use custom title or fall back to suggestion
            final_title = title or query_info["suggested_title"]
            
            # Execute query
            data = self._execute_query(conn, sql)
            conn.close()
            
            if not data:
                return {
                    "success": False,
                    "error": "Nessun dato trovato per la query specificata",
                    "sql_executed": sql
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
                "sql_executed": sql,
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

