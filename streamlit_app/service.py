from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Tuple

from streamlit_app.models import ChatMessage, Conversation
from streamlit_app.repository import ChatRepository


class ChatService:
    """Application service orchestrating chat interactions and persistence."""

    def __init__(self, repository: ChatRepository) -> None:
        self._repository = repository

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

    def _generate_fake_reply(self, user_id: str, conversation_id: int, last_user_message: str) -> ChatMessage:
        """Produce a simple deterministic reply for demo purposes."""
        timestamp = datetime.utcnow().strftime("%H:%M:%S")
        reply_text = (
            f"Echo ({timestamp}): I received your message — '{last_user_message}'. "
            "This is a demo assistant response."
        )
        reply = ChatMessage.new(user_id=user_id, conversation_id=conversation_id, role="assistant", content=reply_text)
        self._repository.add_message(reply)
        return reply

    def send_and_reply(self, user_id: str, conversation_id: int, user_content: str) -> Tuple[ChatMessage, ChatMessage]:
        """Send a message and get a fake reply."""
        user_message = self.add_user_message(user_id=user_id, conversation_id=conversation_id, content=user_content)
        assistant_message = self._generate_fake_reply(user_id=user_id, conversation_id=conversation_id, last_user_message=user_content)
        return user_message, assistant_message


