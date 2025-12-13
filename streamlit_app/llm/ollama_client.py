from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import httpx


@dataclass(frozen=True)
class OllamaModelInfo:
    name: str


class OllamaClient:
    """Minimal Ollama HTTP client (models listing) for Streamlit UI."""

    def __init__(self, base_url: str, timeout_s: float = 2.5) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

    def list_model_names(self) -> List[str]:
        """
        Return model names from `/api/tags`.
        If Ollama is unreachable or returns unexpected payload, returns empty list.
        """
        try:
            with httpx.Client(timeout=self._timeout_s) as client:
                resp = client.get(f"{self._base_url}/api/tags")
                resp.raise_for_status()
                payload = resp.json()
        except Exception:
            return []

        models = payload.get("models", []) if isinstance(payload, dict) else []
        names: List[str] = []
        for m in models:
            if isinstance(m, dict) and isinstance(m.get("name"), str):
                names.append(m["name"])
        return sorted(set(names))

    def pick_default_embedding_model(self, model_names: List[str]) -> Optional[str]:
        """Try to pick a sensible embedding model from the list."""
        preferred = [
            "nomic-embed-text",
            "mxbai-embed-large",
            "bge-m3",
        ]
        for p in preferred:
            for n in model_names:
                if n.split(":")[0] == p:
                    return n
        return model_names[0] if model_names else None


