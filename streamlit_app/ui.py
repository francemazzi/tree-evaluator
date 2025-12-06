from __future__ import annotations

import json
from typing import List, Optional

import plotly.graph_objects as go
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
            st.header("📊 Gestione Dataset")
            
            # Dataset source selection
            data_source = st.radio(
                "Fonte Dati",
                [
                    "🇦🇹 Dataset Vienna (229K alberi)", 
                    "🇮🇹 Dataset Milano (251K alberi)",
                    "📁 Carica CSV Personalizzato"
                ],
                key="data_source_radio"
            )
            
            # Handle Vienna dataset
            if data_source == "🇦🇹 Dataset Vienna (229K alberi)":
                # Clear custom dataset state
                for key in ["custom_db_path", "custom_table_name", "data_description", 
                           "uploaded_file_name", "dataset_metadata", "selected_preset"]:
                    if key in st.session_state:
                        del st.session_state[key]
                
                # Set Vienna as selected preset
                if st.session_state.get("selected_preset") != "vienna":
                    st.session_state.selected_preset = "vienna"
                    self._service._agent = None
                
                st.info("""🌳 **Dataset Vienna Trees (BAUMKATOGD)**
                
- **Alberi totali:** 229.298
- **Distretti:** 23
- **Colonne principali:** specie, anno piantumazione, circonferenza, altezza, via
- **Periodo:** Dati storici fino ad oggi
                """)
            
            # Handle Milano dataset
            elif data_source == "🇮🇹 Dataset Milano (251K alberi)":
                # Clear custom dataset state
                for key in ["custom_db_path", "custom_table_name", "data_description", 
                           "uploaded_file_name", "dataset_metadata"]:
                    if key in st.session_state:
                        del st.session_state[key]
                
                # Set Milano as selected preset
                if st.session_state.get("selected_preset") != "milano":
                    st.session_state.selected_preset = "milano"
                    self._service._agent = None
                
                st.info("""🌳 **Dataset Milano Trees**
                
- **Alberi totali:** 251.165
- **Municipi:** 9
- **Specie uniche:** 338
- **Colonne principali:** genere, specie, varietà, diametro tronco, altezza, via, coordinate GPS
- **Periodo:** Dal 1984 ad oggi
                """)
            
            elif data_source == "📁 Carica CSV Personalizzato":
                st.info("📁 Carica un file CSV per analizzarlo con l'AI")
                
                uploaded_file = st.file_uploader(
                    "Seleziona file CSV",
                    type=["csv"],
                    key="csv_uploader",
                    help="Il file verrà automaticamente convertito in database SQL"
                )
                
                # Optional: description of the data
                description = st.text_area(
                    "Descrizione dati (opzionale)",
                    placeholder="Es: Questo dataset contiene vendite mensili per regione dal 2020 al 2024...",
                    key="data_description",
                    help="Fornisci un contesto che aiuti l'AI a comprendere meglio i tuoi dati",
                    height=100
                )
                
                if uploaded_file:
                    # Only process if file has changed or not yet processed
                    current_file_name = st.session_state.get("uploaded_file_name", None)
                    
                    if current_file_name != uploaded_file.name:
                        with st.spinner("📥 Caricamento e conversione CSV in corso..."):
                            try:
                                from pathlib import Path
                                from streamlit_app.services.data_manager import DynamicDataManager
                                
                                # Initialize manager
                                manager = DynamicDataManager(Path("temp_data"))
                                
                                # Process uploaded file
                                db_path, table_name, metadata = manager.process_uploaded_file(uploaded_file)
                                
                                # Update session state
                                st.session_state.custom_db_path = str(db_path)
                                st.session_state.custom_table_name = table_name
                                st.session_state.data_description = description
                                st.session_state.uploaded_file_name = uploaded_file.name
                                st.session_state.dataset_metadata = metadata
                                
                                # Force agent re-initialization
                                self._service._agent = None
                                
                                st.success(f"✅ Dataset caricato con successo!")
                                
                                # Show metadata
                                with st.expander("📋 Info Dataset"):
                                    st.write(f"**File:** {metadata['original_filename']}")
                                    st.write(f"**Righe:** {metadata['row_count']:,}")
                                    st.write(f"**Colonne:** {metadata['column_count']}")
                                    st.write(f"**Tabella SQL:** `{table_name}`")
                                    st.write("\n**Colonne:**")
                                    for orig, sql in metadata['column_mapping'].items():
                                        st.write(f"- {orig} → `{sql}`")
                                
                            except Exception as e:
                                st.error(f"❌ Errore nel caricamento: {str(e)}")
                                # Clear any partial state
                                if "custom_db_path" in st.session_state:
                                    del st.session_state.custom_db_path
                    else:
                        # File already loaded, just show info
                        st.success(f"✅ Dataset attivo: {uploaded_file.name}")
                        if "dataset_metadata" in st.session_state:
                            metadata = st.session_state.dataset_metadata
                            with st.expander("📋 Info Dataset"):
                                st.write(f"**File:** {metadata['original_filename']}")
                                st.write(f"**Righe:** {metadata['row_count']:,}")
                                st.write(f"**Colonne:** {metadata['column_count']}")
                
                # Button to reset to default dataset
                if st.button("🔄 Torna al Dataset Vienna", use_container_width=True):
                    # Clear custom dataset state
                    for key in ["custom_db_path", "custom_table_name", "data_description", 
                               "uploaded_file_name", "dataset_metadata", "selected_preset"]:
                        if key in st.session_state:
                            del st.session_state[key]
                    # Force agent re-initialization
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

    def _extract_chart_from_response(self, content: str) -> tuple[str, Optional[dict]]:
        """Extract chart JSON from assistant response if present.
        
        Returns:
            Tuple of (text_content, chart_data_dict or None)
        """
        # Look for chart data between markers
        print(f"[DEBUG UI] Checking for chart markers in response (length: {len(content)})")
        print(f"[DEBUG UI] Has CHART_DATA_START: {'CHART_DATA_START' in content}")
        print(f"[DEBUG UI] Has CHART_DATA_END: {'CHART_DATA_END' in content}")
        
        if "CHART_DATA_START" in content and "CHART_DATA_END" in content:
            try:
                start_marker = "CHART_DATA_START"
                end_marker = "CHART_DATA_END"
                
                start_idx = content.find(start_marker) + len(start_marker)
                end_idx = content.find(end_marker)
                
                if start_idx > len(start_marker) and end_idx > start_idx:
                    json_str = content[start_idx:end_idx].strip()
                    print(f"[DEBUG UI] Extracted JSON string length: {len(json_str)}")
                    chart_data = json.loads(json_str)
                    print(f"[DEBUG UI] Successfully parsed chart JSON! Has chart_json: {'chart_json' in chart_data}")
                    
                    if chart_data.get("success") and "chart_json" in chart_data:
                        # Remove chart data section from text
                        text_before = content[:content.find(start_marker)].strip()
                        text_after = content[content.find(end_marker) + len(end_marker):].strip()
                        text_content = (text_before + " " + text_after).strip()
                        print(f"[DEBUG UI] Chart extraction successful!")
                        return text_content, chart_data
            except (json.JSONDecodeError, ValueError) as e:
                print(f"[ERROR UI] Error parsing chart data: {e}")
        
        # Fallback: try old method (for backward compatibility)
        if "chart_json" in content.lower() or '"success": true' in content:
            try:
                # Try to extract JSON object from text
                start_idx = content.find('{')
                end_idx = content.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    json_str = content[start_idx:end_idx+1]
                    chart_data = json.loads(json_str)
                    if chart_data.get("success") and "chart_json" in chart_data:
                        # Remove JSON from text
                        text_before = content[:start_idx].strip()
                        text_after = content[end_idx+1:].strip()
                        text_content = (text_before + " " + text_after).strip()
                        return text_content, chart_data
            except (json.JSONDecodeError, ValueError):
                pass
        
        return content, None
    
    def _render_messages(self) -> None:
        """Render all messages in the current conversation."""
        for message in st.session_state.messages:
            with st.chat_message(message.role):
                # Check if message contains chart data
                if message.role == "assistant":
                    text_content, chart_data = self._extract_chart_from_response(message.content)
                    
                    if chart_data and chart_data.get("success"):
                        # Display text content
                        if text_content:
                            st.markdown(text_content)
                        
                        # Display chart
                        try:
                            chart_json = chart_data["chart_json"]
                            fig = go.Figure(json.loads(chart_json))
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show chart info
                            with st.expander("ℹ️ Dettagli grafico"):
                                st.write(f"**Tipo:** {chart_data.get('chart_type', 'N/A')}")
                                st.write(f"**Punti dati:** {chart_data.get('data_points', 'N/A')}")
                                if "sql_executed" in chart_data:
                                    st.code(chart_data["sql_executed"], language="sql")
                        except Exception as e:
                            st.error(f"Errore nella visualizzazione del grafico: {e}")
                    else:
                        st.markdown(message.content)
                else:
                    st.markdown(message.content)

    def _inject_custom_css(self) -> None:
        """Inject custom CSS for reasoning section styling."""
        st.markdown("""
        <style>
        /* Reasoning section styling - visually distinct from chat */
        .reasoning-container {
            background: linear-gradient(135deg, rgba(45, 55, 72, 0.95) 0%, rgba(26, 32, 44, 0.95) 100%);
            border: 1px solid rgba(99, 179, 237, 0.3);
            border-left: 4px solid #63b3ed;
            border-radius: 8px;
            padding: 16px 20px;
            margin: 12px 0;
            font-size: 0.9em;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        
        .reasoning-header {
            display: flex;
            align-items: center;
            gap: 8px;
            color: #90cdf4;
            font-weight: 600;
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid rgba(99, 179, 237, 0.2);
        }
        
        .reasoning-content {
            color: #e2e8f0;
            line-height: 1.6;
        }
        
        .reasoning-step {
            padding: 8px 12px;
            margin: 6px 0;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 6px;
            border-left: 2px solid rgba(99, 179, 237, 0.5);
        }
        
        .reasoning-step-emoji {
            margin-right: 8px;
        }
        
        /* Loading animation for reasoning */
        .reasoning-loading {
            display: inline-block;
            animation: pulse 1.5s ease-in-out infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 0.6; }
            50% { opacity: 1; }
        }
        
        /* Expander override for reasoning */
        .stExpander[data-testid="stExpander"] > div:first-child {
            background: linear-gradient(135deg, rgba(45, 55, 72, 0.8) 0%, rgba(26, 32, 44, 0.8) 100%);
            border: 1px solid rgba(99, 179, 237, 0.3);
            border-radius: 8px;
        }
        
        /* Better contrast for reasoning expander header */
        .stExpander[data-testid="stExpander"] summary {
            color: #90cdf4 !important;
        }
        
        /* Tool result styling */
        .tool-result {
            background: rgba(72, 187, 120, 0.1);
            border-left: 3px solid #48bb78;
            padding: 8px 12px;
            margin: 8px 0;
            border-radius: 0 6px 6px 0;
            font-size: 0.88em;
        }
        
        /* Validation styling */
        .validation-result {
            background: rgba(159, 122, 234, 0.1);
            border-left: 3px solid #9f7aea;
            padding: 8px 12px;
            margin: 8px 0;
            border-radius: 0 6px 6px 0;
            font-size: 0.88em;
        }
        </style>
        """, unsafe_allow_html=True)

    def _format_reasoning_step(self, step: str) -> str:
        """Format a reasoning step with appropriate emoji and styling."""
        # Determine step type and assign emoji
        step_lower = step.lower()
        
        if "✅" in step or "risultati" in step_lower:
            emoji = "✅"
            css_class = "tool-result"
        elif "✓" in step or "validazione" in step_lower:
            emoji = "✓"
            css_class = "validation-result"
        elif "query sql" in step_lower or "select" in step_lower:
            emoji = "🔍"
            css_class = "reasoning-step"
        else:
            emoji = "💭"
            css_class = "reasoning-step"
        
        return f'<div class="{css_class}"><span class="reasoning-step-emoji">{emoji}</span>{step}</div>'

    def _render_reasoning_box(self, steps: list, is_loading: bool = False) -> str:
        """Render reasoning steps in a styled container."""
        header_text = "🧠 Processo di ragionamento"
        if is_loading:
            header_text += ' <span class="reasoning-loading">●●●</span>'
        
        formatted_steps = "".join([self._format_reasoning_step(s) for s in steps])
        
        return f"""
        <div class="reasoning-container">
            <div class="reasoning-header">{header_text}</div>
            <div class="reasoning-content">{formatted_steps}</div>
        </div>
        """

    def render(self) -> None:
        """Main render method for the chat UI."""
        self._ensure_session()
        st.set_page_config(page_title="Tree Evaluator Chat", page_icon="🌳", layout="centered")
        self._inject_custom_css()
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
                    # Create container for reasoning steps
                    reasoning_placeholder = st.empty()
                    response_placeholder = st.empty()
                    chart_placeholder = st.empty()
                    
                    reasoning_steps = []
                    full_response = ""
                    chart_data = None
                    
                    # Stream from agent
                    for chunk in self._service.stream_reply(
                        user_id=user_id,
                        conversation_id=conversation_id,
                        last_user_message=prompt,
                        openai_api_key=st.session_state.openai_api_key or None
                    ):
                        chunk_type = chunk.get("type", "response")
                        chunk_content = chunk.get("content", "")
                        
                        if chunk_type == "reasoning":
                            # Add reasoning step
                            reasoning_steps.append(chunk_content)
                            # Update reasoning display with styled box
                            reasoning_html = self._render_reasoning_box(reasoning_steps, is_loading=True)
                            reasoning_placeholder.markdown(reasoning_html, unsafe_allow_html=True)
                        
                        elif chunk_type == "response":
                            # Update final response
                            full_response = chunk_content
                            # Show reasoning in collapsed expander with better styling
                            if reasoning_steps:
                                with reasoning_placeholder:
                                    with st.expander("🧠 Processo di ragionamento", expanded=False):
                                        # Use styled HTML for each step
                                        steps_html = "".join([self._format_reasoning_step(s) for s in reasoning_steps])
                                        st.markdown(f'<div class="reasoning-content">{steps_html}</div>', unsafe_allow_html=True)
                            
                            # Check if response contains chart data
                            text_content, extracted_chart = self._extract_chart_from_response(full_response)
                            
                            if extracted_chart and extracted_chart.get("success"):
                                chart_data = extracted_chart
                                response_placeholder.markdown(text_content + "▌")
                            else:
                                response_placeholder.markdown(full_response + "▌")
                    
                    # Final update without cursor
                    if full_response:
                        text_content, extracted_chart = self._extract_chart_from_response(full_response)
                        
                        if extracted_chart and extracted_chart.get("success"):
                            # Display text without chart JSON
                            if text_content:
                                response_placeholder.markdown(text_content)
                            
                            # Display chart
                            with chart_placeholder:
                                try:
                                    chart_json = extracted_chart["chart_json"]
                                    fig = go.Figure(json.loads(chart_json))
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Show chart info
                                    with st.expander("ℹ️ Dettagli grafico"):
                                        st.write(f"**Tipo:** {extracted_chart.get('chart_type', 'N/A')}")
                                        st.write(f"**Punti dati:** {extracted_chart.get('data_points', 'N/A')}")
                                        if "sql_executed" in extracted_chart:
                                            st.code(extracted_chart["sql_executed"], language="sql")
                                except Exception as e:
                                    st.error(f"Errore nella visualizzazione del grafico: {e}")
                        else:
                            response_placeholder.markdown(full_response)
                
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
