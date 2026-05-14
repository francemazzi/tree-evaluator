from __future__ import annotations

import json
from typing import Optional, Tuple

import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components


def extract_chart_from_response(content: str) -> tuple[str, Optional[dict]]:
    """Extract chart JSON from assistant response if present."""
    if "CHART_DATA_START" in content and "CHART_DATA_END" in content:
        try:
            text_content, chart_data = _extract_marked_json(
                content,
                "CHART_DATA_START",
                "CHART_DATA_END",
            )
            if chart_data.get("success") and "chart_json" in chart_data:
                return text_content, chart_data
        except (json.JSONDecodeError, ValueError):
            pass

    if "chart_json" in content.lower() or '"success": true' in content:
        try:
            text_content, chart_data = _extract_inline_json(content)
            if chart_data.get("success") and "chart_json" in chart_data:
                return text_content, chart_data
        except (json.JSONDecodeError, ValueError):
            pass

    return content, None


def extract_map_from_response(content: str) -> Tuple[str, Optional[dict]]:
    """Extract map JSON from assistant response if present."""
    if "MAP_DATA_START" in content and "MAP_DATA_END" in content:
        try:
            text_content, map_data = _extract_marked_json(
                content,
                "MAP_DATA_START",
                "MAP_DATA_END",
            )
            if map_data.get("success") and "map_html" in map_data:
                return text_content, map_data
        except (json.JSONDecodeError, ValueError):
            pass

    if "map_html" in content.lower() or ('"success": true' in content and "map_type" in content):
        try:
            text_content, map_data = _extract_inline_json(content)
            if map_data.get("success") and "map_html" in map_data:
                return text_content, map_data
        except (json.JSONDecodeError, ValueError):
            pass

    return content, None


def _extract_marked_json(content: str, start_marker: str, end_marker: str) -> tuple[str, dict]:
    start_idx = content.find(start_marker) + len(start_marker)
    end_idx = content.find(end_marker)
    if start_idx <= len(start_marker) or end_idx <= start_idx:
        raise ValueError("markers not found")

    json_str = content[start_idx:end_idx].strip()
    payload = json.loads(json_str)
    text_before = content[:content.find(start_marker)].strip()
    text_after = content[content.find(end_marker) + len(end_marker):].strip()
    text_content = (text_before + " " + text_after).strip()
    return text_content, payload


def _extract_inline_json(content: str) -> tuple[str, dict]:
    start_idx = content.find("{")
    end_idx = content.rfind("}")
    if start_idx == -1 or end_idx == -1:
        raise ValueError("json object not found")
    json_str = content[start_idx:end_idx + 1]
    payload = json.loads(json_str)
    text_before = content[:start_idx].strip()
    text_after = content[end_idx + 1:].strip()
    text_content = (text_before + " " + text_after).strip()
    return text_content, payload


def render_map(map_data: dict, placeholder=None) -> None:
    """Render a Folium map from map data."""
    container = placeholder if placeholder else st

    try:
        map_html = map_data["map_html"]
        container.markdown(f"### 🗺️ {map_data.get('title', 'Mappa')}")
        components.html(map_html, height=500, scrolling=True)

        with container.expander("ℹ️ Dettagli mappa"):
            container.write(f"**Tipo:** {map_data.get('map_type', 'N/A')}")
            container.write(f"**Punti visualizzati:** {map_data.get('data_points', 'N/A'):,}")
            if "sql_executed" in map_data:
                container.code(map_data["sql_executed"], language="sql")
            if "center" in map_data:
                center = map_data["center"]
                container.write(f"**Centro:** {center.get('lat', 'N/A')}, {center.get('lon', 'N/A')}")
            container.write(f"**Zoom:** {map_data.get('zoom', 'N/A')}")
    except Exception as exc:
        container.error(f"Errore nella visualizzazione della mappa: {exc}")


def render_plotly_figure(fig: go.Figure, height: int = 520) -> None:
    """Render a Plotly figure robustly inside Streamlit chat."""
    try:
        html = fig.to_html(include_plotlyjs="inline", full_html=False)
        components.html(html, height=height, scrolling=False)
    except Exception as exc:
        st.error(f"Errore nella visualizzazione del grafico (Plotly HTML): {exc}")


def render_chart_data(chart_data: dict) -> None:
    """Render a chart payload embedded in an assistant response."""
    chart_json = chart_data["chart_json"]
    fig = go.Figure(json.loads(chart_json))
    render_plotly_figure(fig)

    with st.expander("ℹ️ Dettagli grafico"):
        st.write(f"**Tipo:** {chart_data.get('chart_type', 'N/A')}")
        st.write(f"**Punti dati:** {chart_data.get('data_points', 'N/A')}")
        if "sql_executed" in chart_data:
            st.code(chart_data["sql_executed"], language="sql")
