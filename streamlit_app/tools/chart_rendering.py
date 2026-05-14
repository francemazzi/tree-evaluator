from __future__ import annotations

from typing import Optional

import plotly.graph_objects as go


def create_chart(
    chart_type: str,
    data: list,
    x_column: str,
    y_column: Optional[str],
    title: str,
    x_label: str,
    y_label: str,
) -> go.Figure:
    """Create a Plotly chart based on type and data."""
    if not data:
        fig = go.Figure()
        fig.add_annotation(
            text="Nessun dato disponibile per questo grafico",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        return fig

    x_data = [row[x_column] for row in data]
    y_data = [row[y_column] for row in data] if y_column else None

    if chart_type == "bar":
        fig = go.Figure(data=[go.Bar(x=x_data, y=y_data, marker_color="#2E7D32")])
    elif chart_type == "pie":
        fig = go.Figure(data=[go.Pie(labels=x_data, values=y_data, hole=0.3)])
    elif chart_type == "line":
        fig = go.Figure(data=[
            go.Scatter(
                x=x_data,
                y=y_data,
                mode="lines+markers",
                line=dict(color="#2E7D32", width=2),
                marker=dict(size=6),
            )
        ])
    elif chart_type == "scatter":
        fig = go.Figure(data=[
            go.Scatter(
                x=x_data,
                y=y_data,
                mode="markers",
                marker=dict(size=8, color="#2E7D32", opacity=0.6),
            )
        ])
    elif chart_type == "histogram":
        fig = go.Figure(data=[go.Histogram(x=x_data, marker_color="#2E7D32", nbinsx=30)])
    elif chart_type == "box":
        fig = go.Figure()
        for category in set(x_data):
            values = [row[y_column] for row in data if row[x_column] == category]
            fig.add_trace(go.Box(y=values, name=str(category)))
    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=18)),
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
        hovermode="closest",
        height=500,
    )
    return fig
