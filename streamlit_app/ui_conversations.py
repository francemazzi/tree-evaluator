from __future__ import annotations

import streamlit as st


def render_conversations(ui) -> None:
    """Render conversation management controls."""
    st.divider()
    st.header("💬 Conversazioni")

    if st.button("➕ Nuova Chat", use_container_width=True, type="primary"):
        ui._create_new_conversation()
        st.rerun()

    if not st.session_state.conversations:
        ui._load_conversations()

    if st.session_state.conversations:
        for conversation in st.session_state.conversations:
            _render_conversation_row(ui, conversation)
    else:
        st.info("Nessuna conversazione. Crea la tua prima chat!")


def _render_conversation_row(ui, conversation) -> None:
    is_editing = st.session_state.editing_conversation_id == conversation.id
    if is_editing:
        _render_editing_row(ui, conversation)
    else:
        _render_regular_row(ui, conversation)


def _render_editing_row(ui, conversation) -> None:
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        new_title = st.text_input(
            "Rinomina",
            value=conversation.title,
            key=f"rename_input_{conversation.id}",
            label_visibility="collapsed",
        )
    with col2:
        if st.button("✓", key=f"save_{conversation.id}", help="Salva"):
            if new_title.strip():
                ui._service.rename_conversation(conversation.id, new_title.strip())
                conversation.title = new_title.strip()
            st.session_state.editing_conversation_id = None
            st.rerun()
    with col3:
        if st.button("✗", key=f"cancel_{conversation.id}", help="Annulla"):
            st.session_state.editing_conversation_id = None
            st.rerun()


def _render_regular_row(ui, conversation) -> None:
    is_current = conversation.id == st.session_state.current_conversation_id
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        button_kwargs = {
            "label": f"{'▶️ ' if is_current else ''}{conversation.title}",
            "key": f"conv_{conversation.id}",
            "use_container_width": True,
        }
        if is_current:
            button_kwargs["type"] = "secondary"

        if st.button(**button_kwargs):
            ui._load_conversation_messages(conversation.id)
            st.rerun()

    with col2:
        if st.button("✏️", key=f"edit_{conversation.id}", help="Rinomina conversazione"):
            st.session_state.editing_conversation_id = conversation.id
            st.rerun()

    with col3:
        if st.button("🗑️", key=f"del_{conversation.id}", help="Elimina conversazione"):
            ui._service.delete_conversation(conversation.id)
            st.session_state.conversations.remove(conversation)
            if conversation.id == st.session_state.current_conversation_id:
                st.session_state.current_conversation_id = None
                st.session_state.messages = []
            st.rerun()
