from __future__ import annotations

import httpx

from streamlit_app.llm import openai_oauth


def _mock_client(monkeypatch, handler):
    real_client = httpx.Client
    transport = httpx.MockTransport(handler)

    def client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(openai_oauth.httpx, "Client", client_factory)


def test_request_device_code_returns_pairing_details(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/accounts/deviceauth/usercode"
        return httpx.Response(
            200,
            json={
                "device_auth_id": "deviceauth_123",
                "user_code": "ABCD-1234",
                "interval": "5",
                "expires_at": "2026-05-06T21:15:41+00:00",
            },
        )

    _mock_client(monkeypatch, handler)

    payload = openai_oauth.request_device_code()

    assert payload["verification_url"] == "https://auth.openai.com/codex/device"
    assert payload["device_auth_id"] == "deviceauth_123"
    assert payload["user_code"] == "ABCD-1234"


def test_complete_device_code_login_exchanges_authorization_code(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/accounts/deviceauth/token":
            return httpx.Response(
                200,
                json={
                    "authorization_code": "auth-code",
                    "code_challenge": "challenge",
                    "code_verifier": "verifier",
                },
            )
        if request.url.path == "/oauth/token":
            body = request.content.decode()
            assert "grant_type=authorization_code" in body
            assert "code=auth-code" in body
            assert "code_verifier=verifier" in body
            return httpx.Response(
                200,
                json={
                    "access_token": "access-token",
                    "id_token": "id-token",
                    "refresh_token": "refresh-token",
                },
            )
        raise AssertionError(f"Unexpected request path: {request.url.path}")

    _mock_client(monkeypatch, handler)

    payload = openai_oauth.complete_device_code_login("deviceauth_123", "ABCD-1234")

    assert payload["refresh_token"] == "refresh-token"


def test_complete_device_code_login_reports_pending_authorization(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": {"code": "deviceauth_authorization_unknown"}})

    _mock_client(monkeypatch, handler)

    try:
        openai_oauth.complete_device_code_login("deviceauth_123", "ABCD-1234")
    except openai_oauth.DeviceAuthorizationPending:
        pass
    else:
        raise AssertionError("Expected pending device authorization")
