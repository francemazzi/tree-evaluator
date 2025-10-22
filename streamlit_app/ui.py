from __future__ import annotations

from typing import List, Optional

import streamlit as st

from streamlit_app.models import ChatMessage, Conversation
from streamlit_app.service import ChatService


class ChatUI:
    """Streamlit UI layer for the chat demo with conversation management."""

    def __init__(self, service: ChatService) -> None:
        self._service = service

    def _ensure_session(self) -> None:
        """Initialize session state variables."""
        if "user_id" not in st.session_state:
            st.session_state.user_id = "guest"
        if "current_conversation_id" not in st.session_state:
            st.session_state.current_conversation_id: Optional[int] = None
        if "messages" not in st.session_state:
            st.session_state.messages: List[ChatMessage] = []
        if "conversations" not in st.session_state:
            st.session_state.conversations: List[Conversation] = []

    def _load_conversations(self) -> None:
        """Load all conversations for the current user."""
        st.session_state.conversations = self._service.list_user_conversations(st.session_state.user_id)

    def _load_conversation_messages(self, conversation_id: int) -> None:
        """Load messages for a specific conversation."""
        st.session_state.messages = self._service.get_conversation_messages(conversation_id)
        st.session_state.current_conversation_id = conversation_id

    def _create_new_conversation(self) -> None:
        """Create a new conversation for the current user."""
        conversation = self._service.create_new_conversation(st.session_state.user_id)
        st.session_state.conversations.insert(0, conversation)
        st.session_state.current_conversation_id = conversation.id
        st.session_state.messages = []

    def _render_sidebar(self) -> None:
        """Render the sidebar with user settings and conversation list."""
        with st.sidebar:
            st.header("⚙️ Settings")
            
            # User ID input
            new_user_id = st.text_input("User ID", value=st.session_state.user_id, key="user_id_input")
            if new_user_id != st.session_state.user_id:
                st.session_state.user_id = new_user_id.strip() or "guest"
                st.session_state.current_conversation_id = None
                st.session_state.messages = []
                st.rerun()

            st.divider()
            st.header("💬 Conversazioni")
            
            # New conversation button
            if st.button("➕ Nuova Chat", use_container_width=True, type="primary"):
                self._create_new_conversation()
                st.rerun()

            # Load conversations if not loaded
            if not st.session_state.conversations:
                self._load_conversations()

            # Display conversation list
            if st.session_state.conversations:
                for conv in st.session_state.conversations:
                    # Highlight current conversation
                    is_current = conv.id == st.session_state.current_conversation_id
                    
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        button_kwargs = {
                            "label": f"{'▶️ ' if is_current else ''}{conv.title}",
                            "key": f"conv_{conv.id}",
                            "use_container_width": True,
                        }
                        if is_current:
                            button_kwargs["type"] = "secondary"
                        
                        if st.button(**button_kwargs):
                            self._load_conversation_messages(conv.id)
                            st.rerun()
                    
                    with col2:
                        if st.button("🗑️", key=f"del_{conv.id}", help="Elimina conversazione"):
                            self._service.delete_conversation(conv.id)
                            st.session_state.conversations.remove(conv)
                            if conv.id == st.session_state.current_conversation_id:
                                st.session_state.current_conversation_id = None
                                st.session_state.messages = []
                            st.rerun()
            else:
                st.info("Nessuna conversazione. Crea la tua prima chat!")

    def _render_messages(self) -> None:
        """Render all messages in the current conversation."""
        for message in st.session_state.messages:
            with st.chat_message(message.role):
                st.markdown(message.content)

    def render(self) -> None:
        """Main render method for the chat UI."""
        self._ensure_session()
        st.set_page_config(page_title="Tree Evaluator Chat Demo", page_icon="🌳", layout="centered")
        st.title("🌳 Tree Evaluator — Chat Demo")
        st.caption("Demo chat con conversazioni multiple e storico su SQLite")

        self._render_sidebar()

        # Main chat area
        if st.session_state.current_conversation_id is None:
            st.info("👈 Seleziona una conversazione dalla sidebar o creane una nuova per iniziare!")
        else:
            self._render_messages()

            # Chat input
            if prompt := st.chat_input("Scrivi un messaggio…"):
                user_id = st.session_state.user_id
                conversation_id = st.session_state.current_conversation_id
                
                # Send message and get reply
                user_msg, assistant_msg = self._service.send_and_reply(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    user_content=prompt
                )
                
                # Update local state
                st.session_state.messages.extend([user_msg, assistant_msg])
                
                # Display new messages
                with st.chat_message("user"):
                    st.markdown(user_msg.content)
                with st.chat_message("assistant"):
                    st.markdown(assistant_msg.content)
