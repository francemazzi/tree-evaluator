from __future__ import annotations

import base64
import json
import os
from typing import Any

import httpx


CLIENT_ID = os.getenv("OPENAI_OAUTH_CLIENT_ID", "app_EMoamEEZ73f0CkXaXp7hrann")
TOKEN_URL = "https://auth.openai.com/oauth/token"
DEVICE_USER_CODE_URL = "https://auth.openai.com/api/accounts/deviceauth/usercode"
DEVICE_TOKEN_URL = "https://auth.openai.com/api/accounts/deviceauth/token"
DEVICE_PAIRING_URL = "https://auth.openai.com/codex/device"
DEVICE_CALLBACK_URL = "https://auth.openai.com/deviceauth/callback"


class DeviceAuthorizationPending(ValueError):
    """Raised when the user has not completed device pairing yet."""


def refresh_access_token(refresh_token: str) -> dict[str, Any]:
    return normalize_chatgpt_token_payload(
        _post_token_form(
            {
                "grant_type": "refresh_token",
                "client_id": CLIENT_ID,
                "refresh_token": refresh_token.strip(),
            }
        )
    )


def normalize_chatgpt_token_payload(payload: dict[str, Any]) -> dict[str, Any]:
    id_token = str(payload.get("id_token") or "").strip()
    claims = parse_chatgpt_id_token(id_token) if id_token else {}
    account_id = (
        str(payload.get("account_id") or "").strip()
        or str(claims.get("chatgpt_account_id") or "").strip()
        or None
    )
    normalized = dict(payload)
    normalized["account_id"] = account_id
    normalized["is_fedramp_account"] = bool(claims.get("chatgpt_account_is_fedramp") or False)
    return normalized


def parse_chatgpt_id_token(id_token: str) -> dict[str, Any]:
    payload = _decode_jwt_payload(id_token)
    auth_claims = payload.get("https://api.openai.com/auth")
    if not isinstance(auth_claims, dict):
        return {}
    return {
        "chatgpt_plan_type": auth_claims.get("chatgpt_plan_type"),
        "chatgpt_user_id": auth_claims.get("chatgpt_user_id") or auth_claims.get("user_id"),
        "chatgpt_account_id": auth_claims.get("chatgpt_account_id"),
        "chatgpt_account_is_fedramp": bool(auth_claims.get("chatgpt_account_is_fedramp")),
    }


def _decode_jwt_payload(jwt_token: str) -> dict[str, Any]:
    parts = jwt_token.split(".")
    if len(parts) != 3 or not all(parts):
        return {}
    payload_b64 = parts[1]
    payload_b64 += "=" * (-len(payload_b64) % 4)
    try:
        decoded = base64.urlsafe_b64decode(payload_b64.encode("ascii"))
        payload = json.loads(decoded)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def request_device_code() -> dict[str, Any]:
    with httpx.Client(timeout=20.0) as client:
        response = client.post(
            DEVICE_USER_CODE_URL,
            json={"client_id": CLIENT_ID},
            headers={"Content-Type": "application/json"},
        )
    payload = _parse_json_response(response, "Richiesta codice dispositivo OpenAI non riuscita")
    user_code = str(payload.get("user_code") or payload.get("usercode") or "").strip()
    device_auth_id = str(payload.get("device_auth_id") or "").strip()
    if not user_code or not device_auth_id:
        raise ValueError("Risposta device pairing OpenAI incompleta.")
    return {
        "verification_url": DEVICE_PAIRING_URL,
        "user_code": user_code,
        "device_auth_id": device_auth_id,
        "interval": str(payload.get("interval") or "5"),
        "expires_at": str(payload.get("expires_at") or ""),
    }


def complete_device_code_login(device_auth_id: str, user_code: str) -> dict[str, Any]:
    authorization = _request_device_authorization(device_auth_id, user_code)
    authorization_code = str(authorization.get("authorization_code") or "").strip()
    code_verifier = str(authorization.get("code_verifier") or "").strip()
    if not authorization_code or not code_verifier:
        raise ValueError("Autorizzazione device pairing OpenAI incompleta.")
    return normalize_chatgpt_token_payload(
        _post_token_form(
            {
                "grant_type": "authorization_code",
                "code": authorization_code,
                "redirect_uri": DEVICE_CALLBACK_URL,
                "client_id": CLIENT_ID,
                "code_verifier": code_verifier,
            }
        )
    )


def _request_device_authorization(device_auth_id: str, user_code: str) -> dict[str, Any]:
    with httpx.Client(timeout=20.0) as client:
        response = client.post(
            DEVICE_TOKEN_URL,
            json={
                "device_auth_id": device_auth_id.strip(),
                "user_code": user_code.strip(),
            },
            headers={"Content-Type": "application/json"},
        )
    if response.status_code in {403, 404}:
        raise DeviceAuthorizationPending("Device pairing non ancora completato nel browser.")
    return _parse_json_response(response, "Device pairing OpenAI non riuscito")


def _post_token_form(data: dict[str, str]) -> dict[str, Any]:
    with httpx.Client(timeout=20.0) as client:
        response = client.post(
            TOKEN_URL,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    return _parse_json_response(response, "Token OAuth OpenAI non valido")


def _parse_json_response(response: httpx.Response, error_prefix: str) -> dict[str, Any]:
    if response.status_code < 200 or response.status_code >= 300:
        detail = response.text.strip() or response.reason_phrase
        raise ValueError(f"{error_prefix}: {detail}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Risposta OAuth OpenAI non valida.")
    return payload
