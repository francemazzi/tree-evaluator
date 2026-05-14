from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, List, Optional, Tuple

from streamlit_app.models import ChatMessage, Conversation, UserLlmSettings
from streamlit_app.repository import ChatRepository
from streamlit_app.service_agent_factory import get_or_create_agent

logger = logging.getLogger(__name__)


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
        openai_auth_method = str(raw.get("openai_auth_method") or defaults.openai_auth_method)
        if openai_auth_method not in {"api_key", "codex_oauth"}:
            openai_auth_method = defaults.openai_auth_method
        anthropic_auth_method = str(raw.get("anthropic_auth_method") or defaults.anthropic_auth_method)
        if anthropic_auth_method not in {"oauth", "api_key"}:
            anthropic_auth_method = defaults.anthropic_auth_method
        return UserLlmSettings(
            user_id=user_id,
            provider=str(raw.get("llm_provider") or defaults.provider),
            openai_auth_method=openai_auth_method,
            openai_api_key=str(raw.get("openai_api_key") or defaults.openai_api_key),
            openai_codex_oauth_token=str(raw.get("openai_codex_oauth_token") or defaults.openai_codex_oauth_token),
            anthropic_setup_token=str(raw.get("anthropic_setup_token") or defaults.anthropic_setup_token),
            anthropic_auth_method=anthropic_auth_method,
            anthropic_oauth_refresh_token=str(raw.get("anthropic_oauth_refresh_token") or defaults.anthropic_oauth_refresh_token),
            anthropic_api_key=str(raw.get("anthropic_api_key") or defaults.anthropic_api_key),
            anthropic_chat_model=str(raw.get("anthropic_chat_model") or defaults.anthropic_chat_model),
            openai_chat_model=str(raw.get("openai_chat_model") or defaults.openai_chat_model),
            openai_embedding_model=str(raw.get("openai_embedding_model") or defaults.openai_embedding_model),
            ollama_base_url=str(raw.get("ollama_base_url") or defaults.ollama_base_url),
            ollama_chat_model=str(raw.get("ollama_chat_model") or defaults.ollama_chat_model),
            ollama_embedding_model=str(raw.get("ollama_embedding_model") or defaults.ollama_embedding_model),
            interface_language=str(raw.get("interface_language") or defaults.interface_language),
        )

    def save_user_llm_settings(self, settings: UserLlmSettings) -> None:
        self._repository.save_user_settings(
            user_id=settings.user_id,
            openai_api_key=settings.openai_api_key,
            openai_auth_method=settings.openai_auth_method,
            openai_codex_oauth_token=settings.openai_codex_oauth_token,
            anthropic_setup_token=settings.anthropic_setup_token,
            anthropic_auth_method=settings.anthropic_auth_method,
            anthropic_oauth_refresh_token=settings.anthropic_oauth_refresh_token,
            anthropic_api_key=settings.anthropic_api_key,
            anthropic_chat_model=settings.anthropic_chat_model,
            llm_provider=settings.provider,
            openai_chat_model=settings.openai_chat_model,
            openai_embedding_model=settings.openai_embedding_model,
            ollama_base_url=settings.ollama_base_url,
            ollama_chat_model=settings.ollama_chat_model,
            ollama_embedding_model=settings.ollama_embedding_model,
            interface_language=settings.interface_language,
        )

    def _resolve_openai_oauth_tokens(self, preferences: UserLlmSettings) -> dict[str, Any]:
        refresh_token = (preferences.openai_codex_oauth_token or "").strip()
        if not refresh_token:
            return {}

        try:
            from streamlit_app.llm.openai_oauth import refresh_access_token

            tokens = refresh_access_token(refresh_token)
        except Exception as e:
            logger.warning("OpenAI OAuth refresh failed: %s", e)
            return {}

        access_token = str(tokens.get("access_token") or "").strip()
        next_refresh_token = str(tokens.get("refresh_token") or "").strip()
        if next_refresh_token and next_refresh_token != refresh_token:
            preferences.openai_codex_oauth_token = next_refresh_token
            self.save_user_llm_settings(preferences)
        if not access_token:
            return {}
        return {
            "access_token": access_token,
            "account_id": tokens.get("account_id"),
            "is_fedramp_account": bool(tokens.get("is_fedramp_account") or False),
        }

    def _resolve_anthropic_oauth_tokens(self, preferences: UserLlmSettings) -> dict[str, Any]:
        refresh_token = (preferences.anthropic_oauth_refresh_token or "").strip()
        if not refresh_token:
            return {}

        try:
            from streamlit_app.llm.anthropic_oauth import refresh_access_token

            tokens = refresh_access_token(refresh_token)
        except Exception as e:
            logger.warning("Anthropic OAuth refresh failed: %s", e)
            return {}

        access_token = str(tokens.get("access_token") or "").strip()
        next_refresh_token = str(tokens.get("refresh_token") or "").strip()
        if next_refresh_token and next_refresh_token != refresh_token:
            preferences.anthropic_oauth_refresh_token = next_refresh_token
            self.save_user_llm_settings(preferences)
        if not access_token:
            return {}
        return {
            "access_token": access_token,
            "expires_at": tokens.get("expires_at"),
        }

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
        """Lazy-load the LangGraph agent using saved per-user preferences."""
        return get_or_create_agent(self, user_id=user_id, openai_api_key=openai_api_key)

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
                logger.warning("Agent failed, using fallback: %s", e)
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
                reasoning_steps = []  # Collect reasoning steps
                for chunk in agent.stream_chat(last_user_message, history=history_dicts):
                    # chunk is a dict with 'type' and 'content'
                    if chunk.get("type") == "response":
                        full_response = chunk.get("content", "")
                    elif chunk.get("type") == "reasoning":
                        reasoning_steps.append(chunk.get("content", ""))
                    yield chunk
                
                # After streaming, save complete message with reasoning
                if full_response:
                    # Combine reasoning steps into a single string
                    reasoning_text = "\n".join(reasoning_steps) if reasoning_steps else None
                    reply = ChatMessage.new(
                        user_id=user_id,
                        conversation_id=conversation_id,
                        role="assistant",
                        content=full_response,
                        reasoning=reasoning_text
                    )
                    self._repository.add_message(reply)
                    return reply
                    
            except Exception as e:
                logger.exception("Agent streaming failed: %s", e)

                fallback_text = self._format_llm_error_for_user(preferences, e, last_user_message)
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

    def _format_llm_error_for_user(self, preferences: UserLlmSettings, error: Exception, last_user_message: Optional[str] = None) -> str:
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

        if preferences.provider == "openai" and preferences.openai_auth_method == "codex_oauth":
            return (
                "Errore durante la chiamata al backend ChatGPT/Codex.\n\n"
                f"Dettaglio tecnico: `{err}`\n\n"
                "Il pairing e il refresh token sono stati letti, ma la richiesta modello non e' stata completata. "
                "Se il dettaglio indica 401/403, rigenera il codice dispositivo dalla sidebar e completa di nuovo il pairing."
            )

        if preferences.provider == "anthropic":
            return (
                "Errore durante la chiamata al backend Claude/Anthropic.\n\n"
                f"Dettaglio tecnico: `{err}`\n\n"
                "Il pairing o la chiave Anthropic sono stati letti, ma la richiesta modello non e' stata completata. "
                "Se il dettaglio indica 401/403, rigenera il login Claude OAuth dalla sidebar o verifica la API key."
            )

        timestamp = datetime.utcnow().strftime("%H:%M:%S")
        user_msg = last_user_message if last_user_message else "messaggio utente"
        return f"Echo ({timestamp}): {user_msg} [fallback - {err}]"

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
