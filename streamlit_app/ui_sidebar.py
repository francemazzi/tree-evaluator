from __future__ import annotations

import streamlit as st

from streamlit_app.ui_conversations import render_conversations
from streamlit_app.ui_dataset import render_dataset_settings
from streamlit_app.ui_llm_settings import render_llm_settings


def render_sidebar(ui) -> None:
    """Render the Streamlit sidebar."""
    with st.sidebar:
        settings = st.session_state.llm_settings
        render_llm_settings(ui, settings)
        render_dataset_settings(ui)
        render_conversations(ui)
