from __future__ import annotations

import os
import socket
from typing import Optional


class OllamaBaseUrlResolver:
    """Resolves a sensible default Ollama base URL depending on runtime environment.

    Goal:
    - When running on the host (no Docker): use localhost.
    - When running inside a Docker container: prefer host Ollama via host.docker.internal,
      falling back to common Linux bridge gateway IP.
    """

    _DEFAULT_PORT: int = 11434

    def resolve(self) -> str:
        if not self._is_running_in_docker():
            return f"http://localhost:{self._DEFAULT_PORT}"

        # Docker Desktop (macOS/Windows) usually provides host.docker.internal
        if self._can_resolve_hostname("host.docker.internal"):
            return f"http://host.docker.internal:{self._DEFAULT_PORT}"

        # Linux: common default bridge gateway for host access
        return f"http://172.17.0.1:{self._DEFAULT_PORT}"

    def _is_running_in_docker(self) -> bool:
        # Standard marker used by Docker.
        if os.path.exists("/.dockerenv"):
            return True
        # Some environments only expose cgroup hints.
        cgroup_path = "/proc/1/cgroup"
        if os.path.exists(cgroup_path):
            try:
                with open(cgroup_path, "r", encoding="utf-8") as f:
                    content = f.read()
                return "docker" in content or "containerd" in content
            except Exception:
                return False
        return False

    def _can_resolve_hostname(self, hostname: str) -> bool:
        try:
            socket.gethostbyname(hostname)
            return True
        except Exception:
            return False


