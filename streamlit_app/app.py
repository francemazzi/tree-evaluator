from __future__ import annotations

from pathlib import Path

from .repository import ChatRepository
from .service import ChatService
from .ui import ChatUI


def main() -> None:
    db_path = Path("chat_index.db")
    repository = ChatRepository(db_path=db_path)
    service = ChatService(repository=repository)
    ui = ChatUI(service=service)
    ui.render()


if __name__ == "__main__":
    main()


