from __future__ import annotations

from typing import Any, Dict

import folium
from folium.plugins import HeatMap, MarkerCluster


def create_popup_html(row: Dict[str, Any], popup_columns: list) -> str:
    """Create HTML content for marker popup."""
    html_parts = []
    for col in popup_columns:
        if col in row and row[col] is not None:
            col_name = col.replace("_", " ").title()
            html_parts.append(f"<b>{col_name}:</b> {row[col]}")
    return "<br>".join(html_parts) if html_parts else "Albero"


def create_map(
    *,
    map_type: str,
    data: list,
    title: str,
    color: str,
    popup_columns: list,
    center_lat: float,
    center_lon: float,
    zoom: int,
    lat_column: str,
    lon_column: str,
) -> folium.Map:
    """Create Folium map based on type and data."""
    tree_map = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles="CartoDB positron",
    )
    tree_map.get_root().html.add_child(folium.Element(_title_html(title, len(data))))

    if not data:
        folium.Marker(
            location=[center_lat, center_lon],
            popup="Nessun dato trovato",
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(tree_map)
        return tree_map

    if map_type == "markers":
        _add_markers(tree_map, data, color, popup_columns, lat_column, lon_column)
    elif map_type == "cluster":
        _add_cluster(tree_map, data, popup_columns, lat_column, lon_column)
    elif map_type == "heatmap":
        _add_heatmap(tree_map, data, lat_column, lon_column)

    return tree_map


def _title_html(title: str, points: int) -> str:
    return f'''
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
            Punti visualizzati: {points:,}
        </p>
    </div>
    '''


def _add_markers(tree_map, data: list, color: str, popup_columns: list, lat_col: str, lon_col: str) -> None:
    for row in data:
        lat = row.get(lat_col)
        lon = row.get(lon_col)
        if lat is not None and lon is not None:
            popup_html = create_popup_html(row, popup_columns)
            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=300),
            ).add_to(tree_map)


def _add_cluster(tree_map, data: list, popup_columns: list, lat_col: str, lon_col: str) -> None:
    cluster = MarkerCluster()
    for row in data:
        lat = row.get(lat_col)
        lon = row.get(lon_col)
        if lat is not None and lon is not None:
            popup_html = create_popup_html(row, popup_columns)
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(color="green", icon="tree-deciduous", prefix="glyphicon"),
            ).add_to(cluster)
    cluster.add_to(tree_map)


def _add_heatmap(tree_map, data: list, lat_col: str, lon_col: str) -> None:
    heat_data = [
        [row.get(lat_col), row.get(lon_col)]
        for row in data
        if row.get(lat_col) is not None and row.get(lon_col) is not None
    ]
    if heat_data:
        HeatMap(
            heat_data,
            radius=15,
            blur=20,
            min_opacity=0.3,
            max_zoom=18,
        ).add_to(tree_map)
