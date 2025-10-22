from __future__ import annotations

import os
from pathlib import Path

from streamlit_app.repository import ChatRepository
from streamlit_app.service import ChatService
from streamlit_app.ui import ChatUI


def main() -> None:
    db_path_str = os.getenv("CHAT_DB_PATH", "data/chat_index.db")
    db_path = Path(db_path_str)
    repository = ChatRepository(db_path=db_path)
    service = ChatService(repository=repository)
    ui = ChatUI(service=service)
    ui.render()


if __name__ == "__main__":
    main()


