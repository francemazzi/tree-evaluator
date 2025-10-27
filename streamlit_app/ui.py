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
        if "openai_api_key" not in st.session_state:
            # Load API key from database if exists
            saved_key = self._service.get_user_api_key(st.session_state.get("user_id", "guest"))
            st.session_state.openai_api_key = saved_key or ""
        if "current_conversation_id" not in st.session_state:
            st.session_state.current_conversation_id: Optional[int] = None
        if "messages" not in st.session_state:
            st.session_state.messages: List[ChatMessage] = []
        if "conversations" not in st.session_state:
            st.session_state.conversations: List[Conversation] = []
        if "editing_conversation_id" not in st.session_state:
            st.session_state.editing_conversation_id: Optional[int] = None

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
            
            # OpenAI API Key input
            new_api_key = st.text_input(
                "OpenAI API Key",
                value=st.session_state.openai_api_key,
                type="password",
                key="api_key_input",
                help="Inserisci la tua chiave API OpenAI (sk-...). Verrà salvata in modo persistente.",
                placeholder="sk-..."
            )
            if new_api_key != st.session_state.openai_api_key:
                st.session_state.openai_api_key = new_api_key.strip()
                # Save API key to database
                if new_api_key.strip():
                    self._service.save_user_api_key(st.session_state.user_id, new_api_key.strip())
                    st.success("✅ Chiave API salvata!")
                # Reset agent to force re-initialization with new key
                self._service._agent = None
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
                    # Check if this conversation is being edited
                    is_editing = st.session_state.editing_conversation_id == conv.id
                    
                    if is_editing:
                        # Show rename input
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            new_title = st.text_input(
                                "Rinomina",
                                value=conv.title,
                                key=f"rename_input_{conv.id}",
                                label_visibility="collapsed"
                            )
                        with col2:
                            if st.button("✓", key=f"save_{conv.id}", help="Salva"):
                                if new_title.strip():
                                    self._service.rename_conversation(conv.id, new_title.strip())
                                    conv.title = new_title.strip()
                                st.session_state.editing_conversation_id = None
                                st.rerun()
                        with col3:
                            if st.button("✗", key=f"cancel_{conv.id}", help="Annulla"):
                                st.session_state.editing_conversation_id = None
                                st.rerun()
                    else:
                        # Normal view
                        is_current = conv.id == st.session_state.current_conversation_id
                        
                        col1, col2, col3 = st.columns([3, 1, 1])
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
                            if st.button("✏️", key=f"edit_{conv.id}", help="Rinomina conversazione"):
                                st.session_state.editing_conversation_id = conv.id
                                st.rerun()
                        
                        with col3:
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
        st.set_page_config(page_title="Tree Evaluator Chat", page_icon="🌳", layout="centered")
        st.title("🌳 Tree Evaluator — AI Chat")
        st.caption("Chatbot intelligente con LangChain/LangGraph per analisi alberi e dataset Vienna")

        self._render_sidebar()

        # Main chat area
        if st.session_state.current_conversation_id is None:
            st.info("👈 Seleziona una conversazione dalla sidebar o creane una nuova per iniziare!")
            
            # Show welcome message with instructions
            if not st.session_state.openai_api_key:
                st.warning("""
                ### 🔑 Configurazione richiesta
                
                Per usare il chatbot intelligente, inserisci la tua **OpenAI API Key** nelle impostazioni (sidebar in alto).
                
                **Come ottenere una chiave:**
                1. Vai su [platform.openai.com](https://platform.openai.com/api-keys)
                2. Crea un account o effettua il login
                3. Genera una nuova API key (inizia con `sk-...`)
                4. Copia e incolla la chiave nelle impostazioni
                
                **Senza chiave API**, il chatbot userà risposte demo di fallback.
                """)
        else:
            self._render_messages()

            # Chat input
            if prompt := st.chat_input("Scrivi un messaggio…"):
                # Check if API key is provided (warn but continue)
                if not st.session_state.openai_api_key:
                    st.info("ℹ️ Nessuna API key configurata. Userò risposte demo. Inserisci la chiave OpenAI nelle impostazioni per usare l'agent intelligente.")
                
                user_id = st.session_state.user_id
                conversation_id = st.session_state.current_conversation_id
                
                # Add user message immediately
                user_msg = self._service.add_user_message(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    content=prompt
                )
                st.session_state.messages.append(user_msg)
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(user_msg.content)
                
                # Stream assistant response
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    # Stream from agent
                    for chunk in self._service.stream_reply(
                        user_id=user_id,
                        conversation_id=conversation_id,
                        last_user_message=prompt,
                        openai_api_key=st.session_state.openai_api_key or None
                    ):
                        full_response = chunk
                        # Update placeholder with current response
                        message_placeholder.markdown(full_response + "▌")
                    
                    # Final update without cursor
                    message_placeholder.markdown(full_response)
                
                # Add assistant message to session state
                # (already persisted by stream_reply, just update UI state)
                from streamlit_app.models import ChatMessage
                assistant_msg = ChatMessage.new(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    role="assistant",
                    content=full_response
                )
                st.session_state.messages.append(assistant_msg)
