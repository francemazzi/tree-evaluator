from __future__ import annotations

import streamlit as st

from streamlit_app.ui_visuals import (
    extract_chart_from_response,
    extract_map_from_response,
    render_chart_data,
    render_map,
)


def render_messages(ui) -> None:
    """Render all messages in the current conversation."""
    for message in st.session_state.messages:
        with st.chat_message(message.role):
            if message.role == "assistant":
                _render_assistant_message(ui, message)
            else:
                st.markdown(message.content)


def _render_assistant_message(ui, message) -> None:
    if hasattr(message, "reasoning") and message.reasoning:
        with st.expander("🧠 Processo di ragionamento", expanded=False):
            for step in message.reasoning.split("\n"):
                if step.strip():
                    st.markdown(step)

    text_content, chart_data = extract_chart_from_response(message.content)
    text_content, map_data = extract_map_from_response(text_content)

    has_visualization = False
    if text_content:
        st.markdown(text_content)

    if chart_data and chart_data.get("success"):
        has_visualization = True
        try:
            render_chart_data(chart_data)
        except Exception as exc:
            st.error(f"Errore nella visualizzazione del grafico: {exc}")

    if map_data and map_data.get("success"):
        has_visualization = True
        render_map(map_data)

    if not has_visualization and not text_content:
        st.markdown(message.content)
