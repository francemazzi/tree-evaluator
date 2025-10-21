from __future__ import annotations

from typing import List

import streamlit as st

from .models import ChatMessage
from .service import ChatService


class ChatUI:
    """Streamlit UI layer for the fake chat demo."""

    def __init__(self, service: ChatService) -> None:
        self._service = service

    def _ensure_session(self) -> None:
        if "user_id" not in st.session_state:
            st.session_state.user_id = "guest"
        if "messages" not in st.session_state:
            st.session_state.messages: List[ChatMessage] = []
        if "history_loaded" not in st.session_state:
            st.session_state.history_loaded = False

    def _load_history_once(self) -> None:
        if not st.session_state.history_loaded and st.session_state.user_id:
            history = self._service.get_history(st.session_state.user_id)
            st.session_state.messages = history
            st.session_state.history_loaded = True

    def _render_sidebar(self) -> None:
        with st.sidebar:
            st.header("Settings")
            new_user_id = st.text_input("User ID", value=st.session_state.user_id)
            if new_user_id != st.session_state.user_id:
                st.session_state.user_id = new_user_id.strip() or "guest"
                st.session_state.history_loaded = False
                self._load_history_once()

            if st.button("Clear history", type="secondary"):
                if st.session_state.user_id:
                    self._service.clear_user_history(st.session_state.user_id)
                    st.session_state.messages = []
                    st.session_state.history_loaded = True
                    st.toast("History cleared", icon="✅")

    def _render_messages(self) -> None:
        for message in st.session_state.messages:
            with st.chat_message(message.role):
                st.markdown(message.content)

    def render(self) -> None:
        self._ensure_session()
        st.set_page_config(page_title="Tree Evaluator Chat Demo", page_icon="🌳", layout="centered")
        st.title("🌳 Tree Evaluator — Fake Chat")
        st.caption("Demo chat con storico per utente su SQLite (chat_index.db)")

        self._render_sidebar()
        self._load_history_once()
        self._render_messages()

        if prompt := st.chat_input("Scrivi un messaggio…"):
            user_id = st.session_state.user_id
            user_msg, assistant_msg = self._service.send_and_reply(user_id=user_id, user_content=prompt)
            st.session_state.messages.extend([user_msg, assistant_msg])
            with st.chat_message("user"):
                st.markdown(user_msg.content)
            with st.chat_message("assistant"):
                st.markdown(assistant_msg.content)


