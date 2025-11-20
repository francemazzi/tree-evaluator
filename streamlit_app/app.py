from __future__ import annotations

import os
import sys
from pathlib import Path

# Add root directory to Python path to allow absolute imports
# This is necessary for Streamlit Cloud where PYTHONPATH might not include the root
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from streamlit_app.repository import ChatRepository
from streamlit_app.service import ChatService
from streamlit_app.ui import ChatUI


def main() -> None:
    db_path_str = os.getenv("CHAT_DB_PATH", "data/chat_index.db")
    db_path = Path(db_path_str)
    
    # Ensure directory exists for Streamlit Cloud
    if not db_path.parent.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
    repository = ChatRepository(db_path=db_path)
    service = ChatService(repository=repository)
    ui = ChatUI(service=service)
    ui.render()


if __name__ == "__main__":
    main()


