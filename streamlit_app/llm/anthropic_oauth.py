from __future__ import annotations

import base64
import hashlib
import http.server
import os
import queue
import secrets
import socketserver
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional
from urllib.parse import parse_qs, urlencode, urlparse

import httpx


CLIENT_ID = os.getenv("ANTHROPIC_OAUTH_CLIENT_ID", "9d1c250a-e61b-44d9-88ed-5944d1962f5e")
AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"
MANUAL_REDIRECT_URI = "https://console.anthropic.com/oauth/code/callback"
LOOPBACK_HOST = os.getenv("ANTHROPIC_OAUTH_LOOPBACK_HOST", "127.0.0.1")
LOOPBACK_PORT_RANGE = tuple(range(53682, 53691))
DEFAULT_SCOPES = ("user:inference",)


class AnthropicAuthorizationPending(ValueError):
    """Raised when the OAuth authorization callback has not arrived yet."""


@dataclass(frozen=True)
class PkcePair:
    verifier: str
    challenge: str


@dataclass
class LoopbackServer:
    redirect_uri: str
    server: socketserver.ThreadingTCPServer
    thread: threading.Thread
    result_queue: "queue.Queue[dict[str, str]]"

    def poll(self) -> Optional[dict[str, str]]:
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class _ReusableThreadingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


def generate_pkce() -> PkcePair:
    verifier = _base64url(secrets.token_bytes(32))
    challenge = _base64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return PkcePair(verifier=verifier, challenge=challenge)


def generate_state() -> str:
    return _base64url(secrets.token_bytes(24))


def build_authorize_url(
    *,
    redirect_uri: str,
    code_challenge: str,
    state: str,
    scopes: Iterable[str] = DEFAULT_SCOPES,
    client_id: str = CLIENT_ID,
) -> str:
    query = {
        "code": "true",
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": " ".join(scope for scope in scopes if scope),
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
    }
    return f"{AUTHORIZE_URL}?{urlencode(query)}"


def exchange_code_for_tokens(
    *,
    code: str,
    verifier: str,
    redirect_uri: str,
    state: Optional[str] = None,
    client_id: str = CLIENT_ID,
) -> dict[str, Any]:
    return _normalize_token_payload(
        _post_token_json(
            {
                "grant_type": "authorization_code",
                "code": code.strip(),
                "state": (state or verifier).strip(),
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "code_verifier": verifier.strip(),
            }
        )
    )


def refresh_access_token(refresh_token: str, client_id: str = CLIENT_ID) -> dict[str, Any]:
    return _normalize_token_payload(
        _post_token_json(
            {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token.strip(),
                "client_id": client_id,
            }
        )
    )


def start_loopback_server(expected_state: str) -> LoopbackServer:
    result_queue: "queue.Queue[dict[str, str]]" = queue.Queue(maxsize=1)
    handler = _build_callback_handler(expected_state, result_queue)

    last_error: Optional[Exception] = None
    for port in LOOPBACK_PORT_RANGE:
        try:
            server = _ReusableThreadingTCPServer((LOOPBACK_HOST, port), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            return LoopbackServer(
                redirect_uri=f"http://{LOOPBACK_HOST}:{port}/callback",
                server=server,
                thread=thread,
                result_queue=result_queue,
            )
        except OSError as exc:
            last_error = exc
            continue

    raise RuntimeError(f"No available Anthropic OAuth loopback port: {last_error}")


def parse_pasted_code(value: str, default_state: Optional[str] = None) -> dict[str, str]:
    raw = (value or "").strip()
    if not raw:
        raise ValueError("Codice OAuth Anthropic vuoto.")
    if "#" in raw:
        code, state = raw.split("#", 1)
    else:
        code, state = raw, default_state or ""
    code = code.strip()
    state = state.strip()
    if not code:
        raise ValueError("Codice OAuth Anthropic non valido.")
    return {"code": code, "state": state}


def _post_token_json(data: dict[str, str]) -> dict[str, Any]:
    with httpx.Client(timeout=20.0) as client:
        response = client.post(
            TOKEN_URL,
            json=data,
            headers={"Content-Type": "application/json"},
        )
    if response.status_code < 200 or response.status_code >= 300:
        detail = response.text.strip() or response.reason_phrase
        raise ValueError(f"Token OAuth Anthropic non valido: {detail}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Risposta OAuth Anthropic non valida.")
    return payload


def _normalize_token_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    access_token = str(
        normalized.get("access_token") or normalized.get("accessToken") or ""
    ).strip()
    refresh_token = str(
        normalized.get("refresh_token") or normalized.get("refreshToken") or ""
    ).strip()
    normalized["access_token"] = access_token
    normalized["refresh_token"] = refresh_token
    expires_in = normalized.get("expires_in")
    if not normalized.get("expires_at"):
        try:
            normalized["expires_at"] = int(time.time() * 1000) + int(expires_in or 3600) * 1000
        except (TypeError, ValueError):
            normalized["expires_at"] = int(time.time() * 1000) + 3600 * 1000
    return normalized


def _build_callback_handler(
    expected_state: str,
    result_queue: "queue.Queue[dict[str, str]]",
) -> type[http.server.BaseHTTPRequestHandler]:
    class OAuthCallbackHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
            parsed = urlparse(self.path)
            if parsed.path != "/callback":
                self._respond(404, "Callback OAuth non trovata.")
                return

            params = parse_qs(parsed.query)
            code = (params.get("code") or [""])[0].strip()
            state = (params.get("state") or [""])[0].strip()
            error = (params.get("error") or [""])[0].strip()

            if error:
                self._respond(400, f"Login Claude non riuscito: {error}")
                return
            if not code:
                self._respond(400, "Callback OAuth Anthropic senza code.")
                return
            if state != expected_state:
                self._respond(400, "State OAuth Anthropic non valido.")
                return

            try:
                result_queue.put_nowait({"code": code, "state": state})
            except queue.Full:
                pass
            self._respond(200, "Login Claude completato. Puoi chiudere questa finestra e tornare all'app.")

        def log_message(self, format: str, *args: object) -> None:
            return

        def _respond(self, status: int, message: str) -> None:
            body = (
                "<!doctype html><html><body>"
                f"<h1>{message}</h1>"
                "</body></html>"
            ).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return OAuthCallbackHandler


def _base64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")
