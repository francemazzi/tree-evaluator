from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Tuple

from streamlit_app.models import ChatMessage, Conversation
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

    # Message management
    
    def get_conversation_messages(self, conversation_id: int) -> List[ChatMessage]:
        """Get all messages in a conversation."""
        return self._repository.list_messages_by_conversation(conversation_id)

    def add_user_message(self, user_id: str, conversation_id: int, content: str) -> ChatMessage:
        """Add a user message to a conversation."""
        message = ChatMessage.new(user_id=user_id, conversation_id=conversation_id, role="user", content=content)
        self._repository.add_message(message)
        return message

    def _get_or_create_agent(self, openai_api_key: Optional[str] = None):
        """Lazy-load the LangGraph agent with optional API key."""
        # Se non c'è API key, ritorna None subito
        if not openai_api_key:
            return None
            
        # Se agent già esiste, ritorna quello esistente
        if self._agent is not None:
            return self._agent
            
        # Crea nuovo agent
        try:
            from streamlit_app.agent import TreeEvaluatorAgent
            import streamlit as st
            
            # Inizializza agent senza mostrare messaggi ripetitivi
            self._agent = TreeEvaluatorAgent(openai_api_key=openai_api_key)
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
        agent = self._get_or_create_agent(openai_api_key=openai_api_key)
        
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
        """Stream reply from agent with real-time updates.
        
        Yields chunks of the response as they are generated.
        Returns the complete message at the end for persistence.
        """
        agent = self._get_or_create_agent(openai_api_key=openai_api_key)
        
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
                    full_response = chunk  # Agent yields full content each time
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
                # Fallback
                timestamp = datetime.utcnow().strftime("%H:%M:%S")
                fallback_text = f"Echo ({timestamp}): {last_user_message} [fallback]"
                yield fallback_text
                
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
            yield fallback_text
            
            reply = ChatMessage.new(
                user_id=user_id,
                conversation_id=conversation_id,
                role="assistant",
                content=fallback_text
            )
            self._repository.add_message(reply)
            return reply

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


