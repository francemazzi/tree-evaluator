from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Tuple

from streamlit_app.models import ChatMessage, Conversation, UserLlmSettings
from streamlit_app.repository import ChatRepository


class ChatService:
    """Application service orchestrating chat interactions and persistence."""

    def __init__(self, repository: ChatRepository, agent=None) -> None:
        self._repository = repository
        self._agent = agent  # LangGraph agent (optional, lazy-loaded)

    # Conversation management
    
    def create_new_conversation(self, user_id: str, title: Optional[str] = None) -> Conversation:
        """Create a new conversation for the user with auto-generated title."""
        if title is None:
            # Generate automatic title: "Chat HH:MM-DD-MM-YYYY"
            now = datetime.now()
            title = f"Chat {now.strftime('%H:%M-%d-%m-%Y')}"
        
        conversation = Conversation.new(user_id=user_id, title=title)
        conversation_id = self._repository.create_conversation(conversation)
        conversation.id = conversation_id
        return conversation

    def list_user_conversations(self, user_id: str) -> List[Conversation]:
        """List all conversations for a user."""
        return self._repository.list_conversations_by_user(user_id)

    def get_conversation(self, conversation_id: int) -> Optional[Conversation]:
        """Get a specific conversation by ID."""
        return self._repository.get_conversation(conversation_id)

    def rename_conversation(self, conversation_id: int, new_title: str) -> None:
        """Rename a conversation."""
        self._repository.update_conversation_title(conversation_id, new_title)

    def delete_conversation(self, conversation_id: int) -> int:
        """Delete a conversation and all its messages."""
        return self._repository.delete_conversation(conversation_id)

    # User settings management
    
    def save_user_api_key(self, user_id: str, api_key: str) -> None:
        """Save user's OpenAI API key."""
        # Keep backward compatibility: only updating key, preserving existing provider/models.
        current = self.get_user_llm_settings(user_id)
        updated = current
        updated.openai_api_key = api_key
        self.save_user_llm_settings(updated)
    
    def get_user_api_key(self, user_id: str) -> Optional[str]:
        """Get user's saved OpenAI API key."""
        settings = self._repository.get_user_settings(user_id)
        return settings.get("openai_api_key") if settings else None

    def get_user_llm_settings(self, user_id: str) -> UserLlmSettings:
        raw = self._repository.get_user_settings(user_id) or {}
        defaults = UserLlmSettings.default(user_id)
        return UserLlmSettings(
            user_id=user_id,
            provider=str(raw.get("llm_provider") or defaults.provider),
            openai_api_key=str(raw.get("openai_api_key") or defaults.openai_api_key),
            openai_chat_model=str(raw.get("openai_chat_model") or defaults.openai_chat_model),
            openai_embedding_model=str(raw.get("openai_embedding_model") or defaults.openai_embedding_model),
            ollama_base_url=str(raw.get("ollama_base_url") or defaults.ollama_base_url),
            ollama_chat_model=str(raw.get("ollama_chat_model") or defaults.ollama_chat_model),
            ollama_embedding_model=str(raw.get("ollama_embedding_model") or defaults.ollama_embedding_model),
        )

    def save_user_llm_settings(self, settings: UserLlmSettings) -> None:
        self._repository.save_user_settings(
            user_id=settings.user_id,
            openai_api_key=settings.openai_api_key,
            llm_provider=settings.provider,
            openai_chat_model=settings.openai_chat_model,
            openai_embedding_model=settings.openai_embedding_model,
            ollama_base_url=settings.ollama_base_url,
            ollama_chat_model=settings.ollama_chat_model,
            ollama_embedding_model=settings.ollama_embedding_model,
        )

    # Message management
    
    def get_conversation_messages(self, conversation_id: int) -> List[ChatMessage]:
        """Get all messages in a conversation."""
        return self._repository.list_messages_by_conversation(conversation_id)

    def add_user_message(self, user_id: str, conversation_id: int, content: str) -> ChatMessage:
        """Add a user message to a conversation."""
        message = ChatMessage.new(user_id=user_id, conversation_id=conversation_id, role="user", content=content)
        self._repository.add_message(message)
        return message

    def _get_or_create_agent(self, user_id: str, openai_api_key: Optional[str] = None):
        """Lazy-load the LangGraph agent using saved per-user preferences (OpenAI/Ollama)."""
        preferences = self.get_user_llm_settings(user_id)
        # If UI passed a key explicitly, it overrides persisted key.
        if openai_api_key is not None:
            preferences.openai_api_key = openai_api_key

        if preferences.provider == "openai" and not preferences.openai_api_key:
            return None
            
        # Se agent già esiste, ritorna quello esistente
        if self._agent is not None:
            return self._agent
            
        # Crea nuovo agent
        try:
            from streamlit_app.agent import TreeEvaluatorAgent
            import streamlit as st
            from pathlib import Path
            
            # Check if custom dataset is configured
            custom_db_path = st.session_state.get("custom_db_path", None)
            custom_table_name = st.session_state.get("custom_table_name", None)
            data_description = st.session_state.get("data_description", "")
            selected_preset = st.session_state.get("selected_preset", "vienna")
            
            # Inizializza agent con configurazione dataset
            if custom_db_path and custom_table_name:
                # Custom uploaded CSV
                self._agent = TreeEvaluatorAgent(
                    openai_api_key=preferences.openai_api_key or None,
                    provider=preferences.provider,
                    openai_chat_model=preferences.openai_chat_model,
                    openai_embedding_model=preferences.openai_embedding_model,
                    ollama_base_url=preferences.ollama_base_url,
                    ollama_chat_model=preferences.ollama_chat_model,
                    ollama_embedding_model=preferences.ollama_embedding_model,
                    custom_db_path=Path(custom_db_path),
                    custom_table_name=custom_table_name,
                    data_description=data_description
                )
            elif selected_preset == "milano":
                # Milano preset dataset
                self._agent = TreeEvaluatorAgent(
                    openai_api_key=preferences.openai_api_key or None,
                    provider=preferences.provider,
                    openai_chat_model=preferences.openai_chat_model,
                    openai_embedding_model=preferences.openai_embedding_model,
                    ollama_base_url=preferences.ollama_base_url,
                    ollama_chat_model=preferences.ollama_chat_model,
                    ollama_embedding_model=preferences.ollama_embedding_model,
                    dataset_preset="milano"
                )
            else:
                # Default: Vienna dataset
                self._agent = TreeEvaluatorAgent(
                    openai_api_key=preferences.openai_api_key or None,
                    provider=preferences.provider,
                    openai_chat_model=preferences.openai_chat_model,
                    openai_embedding_model=preferences.openai_embedding_model,
                    ollama_base_url=preferences.ollama_base_url,
                    ollama_chat_model=preferences.ollama_chat_model,
                    ollama_embedding_model=preferences.ollama_embedding_model,
                )
            
            return self._agent
            
        except ImportError as e:
            import streamlit as st
            st.error(f"❌ Errore import dipendenze: {e}\nInstalla: pip install -r requirements.txt")
            print(f"Import error: {e}")
            return None
        except ValueError as e:
            import streamlit as st
            st.error(f"❌ Chiave API non valida: {e}")
            print(f"ValueError: {e}")
            return None
        except Exception as e:
            import streamlit as st
            st.error(f"❌ Errore inizializzazione agent: {e}")
            print(f"Agent init error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _generate_fake_reply(self, user_id: str, conversation_id: int, last_user_message: str, openai_api_key: Optional[str] = None) -> ChatMessage:
        """Generate reply using LangGraph agent or fallback to demo response."""
        # Try to use agent with provided API key
        agent = self._get_or_create_agent(user_id=user_id, openai_api_key=openai_api_key)
        
        if agent is not None:
            try:
                # Get conversation history for context
                history = self.get_conversation_messages(conversation_id)
                # Convert to format expected by agent (exclude current message)
                history_dicts = [
                    {"role": msg.role, "content": msg.content}
                    for msg in history
                ]
                
                # Get agent response
                reply_text = agent.chat(last_user_message, history=history_dicts)
            except Exception as e:
                # Fallback to demo response if agent fails
                print(f"Warning: Agent failed, using fallback: {e}")
                timestamp = datetime.utcnow().strftime("%H:%M:%S")
                reply_text = (
                    f"Echo ({timestamp}): I received your message — '{last_user_message}'. "
                    "This is a demo assistant response."
                )
        else:
            # No agent available, use demo response
            timestamp = datetime.utcnow().strftime("%H:%M:%S")
            reply_text = (
                f"Echo ({timestamp}): I received your message — '{last_user_message}'. "
                "This is a demo assistant response."
            )
        
        reply = ChatMessage.new(user_id=user_id, conversation_id=conversation_id, role="assistant", content=reply_text)
        self._repository.add_message(reply)
        return reply

    def stream_reply(self, user_id: str, conversation_id: int, last_user_message: str, openai_api_key: Optional[str] = None):
        """Stream reply from agent with real-time updates including reasoning steps.
        
        Yields dict with 'type' and 'content':
        - type: 'reasoning' for internal steps, 'response' for final answer
        - content: the text to display
        
        Returns the complete message at the end for persistence.
        """
        agent = self._get_or_create_agent(user_id=user_id, openai_api_key=openai_api_key)
        preferences = self.get_user_llm_settings(user_id)
        
        if agent is not None:
            try:
                # Get conversation history
                history = self.get_conversation_messages(conversation_id)
                history_dicts = [
                    {"role": msg.role, "content": msg.content}
                    for msg in history
                ]
                
                # Stream response from agent
                full_response = ""
                for chunk in agent.stream_chat(last_user_message, history=history_dicts):
                    # chunk is a dict with 'type' and 'content'
                    if chunk.get("type") == "response":
                        full_response = chunk.get("content", "")
                    yield chunk
                
                # After streaming, save complete message
                if full_response:
                    reply = ChatMessage.new(
                        user_id=user_id,
                        conversation_id=conversation_id,
                        role="assistant",
                        content=full_response
                    )
                    self._repository.add_message(reply)
                    return reply
                    
            except Exception as e:
                print(f"Warning: Agent streaming failed: {e}")
                import traceback
                traceback.print_exc()

                fallback_text = self._format_llm_error_for_user(preferences, e)
                yield {"type": "response", "content": fallback_text}
                
                reply = ChatMessage.new(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    role="assistant",
                    content=fallback_text
                )
                self._repository.add_message(reply)
                return reply
        else:
            # No agent, use demo
            timestamp = datetime.utcnow().strftime("%H:%M:%S")
            fallback_text = f"Echo ({timestamp}): {last_user_message} [demo]"
            yield {"type": "response", "content": fallback_text}
            
            reply = ChatMessage.new(
                user_id=user_id,
                conversation_id=conversation_id,
                role="assistant",
                content=fallback_text
            )
            self._repository.add_message(reply)
            return reply

    def _format_llm_error_for_user(self, preferences: UserLlmSettings, error: Exception) -> str:
        """User-facing error message with actionable hints (only when needed)."""
        err = str(error)

        if preferences.provider == "ollama":
            import re

            # Common Ollama error pattern: model 'xxx' not found (status code: 404)
            m = re.search(r"model\s+'([^']+)'\s+not\s+found", err, flags=re.IGNORECASE)
            model_name = m.group(1) if m else None
            if model_name:
                return (
                    f"❌ Modello Ollama **{model_name}** non trovato (404).\n\n"
                    f"Esegui sul tuo host:\n"
                    f"```bash\nollama pull {model_name}\n```\n"
                    f"Poi in Settings → **Aggiorna modelli** e selezionalo.\n\n"
                    f"Se non hai ancora modelli:\n"
                    f"```bash\nollama pull nomic-embed-text\nollama pull gpt-oss:20b\n```"
                )

            # Generic Ollama hint (no templates unless error actually happens)
            return (
                "❌ Errore durante la chiamata a Ollama.\n\n"
                "Controlla che Ollama sia avviato e che il base URL sia corretto.\n\n"
                "Se ti mancano modelli:\n"
                "```bash\nollama pull nomic-embed-text\nollama pull gpt-oss:20b\n```"
            )

        timestamp = datetime.utcnow().strftime("%H:%M:%S")
        return f"Echo ({timestamp}): {last_user_message} [fallback - {err}]"

    def send_and_reply(self, user_id: str, conversation_id: int, user_content: str, openai_api_key: Optional[str] = None) -> Tuple[ChatMessage, ChatMessage]:
        """Send a message and get a reply (with optional OpenAI API key)."""
        user_message = self.add_user_message(user_id=user_id, conversation_id=conversation_id, content=user_content)
        assistant_message = self._generate_fake_reply(
            user_id=user_id,
            conversation_id=conversation_id,
            last_user_message=user_content,
            openai_api_key=openai_api_key
        )
        return user_message, assistant_message


