from __future__ import annotations

import json
from urllib.parse import parse_qs, urlparse

import httpx

from streamlit_app.llm import anthropic_oauth


def _mock_client(monkeypatch, handler):
    real_client = httpx.Client
    transport = httpx.MockTransport(handler)

    def client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(anthropic_oauth.httpx, "Client", client_factory)


def test_build_authorize_url_contains_pkce_and_copy_paste_flags():
    url = anthropic_oauth.build_authorize_url(
        redirect_uri="http://127.0.0.1:53682/callback",
        code_challenge="challenge",
        state="state-123",
    )

    parsed = urlparse(url)
    params = parse_qs(parsed.query)

    assert parsed.scheme == "https"
    assert parsed.netloc == "claude.ai"
    assert params["code"] == ["true"]
    assert params["client_id"] == [anthropic_oauth.CLIENT_ID]
    assert params["response_type"] == ["code"]
    assert params["redirect_uri"] == ["http://127.0.0.1:53682/callback"]
    assert params["scope"] == ["user:inference"]
    assert params["code_challenge"] == ["challenge"]
    assert params["code_challenge_method"] == ["S256"]
    assert params["state"] == ["state-123"]


def test_exchange_code_for_tokens_posts_expected_payload(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["payload"] = request.read().decode()
        return httpx.Response(
            200,
            json={
                "access_token": "access-token",
                "refresh_token": "refresh-token",
                "expires_in": 120,
            },
        )

    _mock_client(monkeypatch, handler)

    tokens = anthropic_oauth.exchange_code_for_tokens(
        code="code-123",
        state="state-123",
        verifier="verifier-123",
        redirect_uri="http://127.0.0.1:53682/callback",
    )

    payload = json.loads(captured["payload"])
    assert captured["url"] == anthropic_oauth.TOKEN_URL
    assert payload["grant_type"] == "authorization_code"
    assert payload["code"] == "code-123"
    assert tokens["access_token"] == "access-token"
    assert tokens["refresh_token"] == "refresh-token"
    assert isinstance(tokens["expires_at"], int)


def test_refresh_access_token_posts_refresh_grant(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = request.read().decode()
        return httpx.Response(
            200,
            json={
                "accessToken": "access-token",
                "refreshToken": "next-refresh-token",
                "expires_in": 60,
            },
        )

    _mock_client(monkeypatch, handler)

    tokens = anthropic_oauth.refresh_access_token("refresh-token")

    payload = json.loads(captured["payload"])
    assert payload["grant_type"] == "refresh_token"
    assert payload["refresh_token"] == "refresh-token"
    assert tokens["access_token"] == "access-token"
    assert tokens["refresh_token"] == "next-refresh-token"


def test_loopback_server_accepts_matching_state_and_rejects_mismatch():
    server = anthropic_oauth.start_loopback_server("expected-state")
    try:
        ok = httpx.get(
            f"{server.redirect_uri}?code=code-123&state=expected-state",
            timeout=5.0,
        )
        assert ok.status_code == 200
        assert server.poll() == {"code": "code-123", "state": "expected-state"}

        mismatch = httpx.get(
            f"{server.redirect_uri}?code=code-456&state=wrong-state",
            timeout=5.0,
        )
        assert mismatch.status_code == 400
    finally:
        server.close()
