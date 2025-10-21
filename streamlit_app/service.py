from __future__ import annotations

from datetime import datetime
from typing import List, Tuple

from .models import ChatMessage
from .repository import ChatRepository


class ChatService:
    """Application service orchestrating chat interactions and persistence."""

    def __init__(self, repository: ChatRepository) -> None:
        self._repository = repository

    def get_history(self, user_id: str) -> List[ChatMessage]:
        return self._repository.list_messages_by_user(user_id)

    def add_user_message(self, user_id: str, content: str) -> ChatMessage:
        message = ChatMessage.new(user_id=user_id, role="user", content=content)
        self._repository.add_message(message)
        return message

    def _generate_fake_reply(self, user_id: str, last_user_message: str) -> ChatMessage:
        """Produce a simple deterministic reply for demo purposes.

        The content references the user's last message to make the
        interaction feel coherent without external dependencies.
        """
        timestamp = datetime.utcnow().strftime("%H:%M:%S")
        reply_text = (
            f"Echo ({timestamp}): I received your message — '{last_user_message}'. "
            "This is a demo assistant response."
        )
        reply = ChatMessage.new(user_id=user_id, role="assistant", content=reply_text)
        self._repository.add_message(reply)
        return reply

    def send_and_reply(self, user_id: str, user_content: str) -> Tuple[ChatMessage, ChatMessage]:
        user_message = self.add_user_message(user_id=user_id, content=user_content)
        assistant_message = self._generate_fake_reply(user_id=user_id, last_user_message=user_content)
        return user_message, assistant_message

    def clear_user_history(self, user_id: str) -> int:
        return self._repository.clear_user_history(user_id)


