from __future__ import annotations

import json

import httpx
from langchain_core.messages import HumanMessage

from streamlit_app.llm import anthropic_backend
from streamlit_app.llm.anthropic_backend import (
    CLAUDE_OAUTH_DEFAULT_MODEL,
    CLAUDE_OAUTH_SUPPORTED_MODELS,
    ClaudeOAuthChatModel,
    resolve_claude_oauth_model,
)


def _mock_client(monkeypatch, handler):
    real_client = httpx.Client
    transport = httpx.MockTransport(handler)

    def client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(anthropic_backend.httpx, "Client", client_factory)


def test_anthropic_backend_posts_messages_request_and_reads_sse(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = request.headers
        captured["payload"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            content=(
                b'data: {"type":"content_block_delta","index":0,'
                b'"delta":{"type":"text_delta","text":"ciao"}}\n\n'
                b'data: {"type":"message_stop"}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    _mock_client(monkeypatch, handler)

    model = ClaudeOAuthChatModel(
        model_name="claude-sonnet-4-5",
        access_token="access-token",
    )
    message = model.invoke([HumanMessage(content="ciao")])

    assert message.content == "ciao"
    assert captured["headers"]["authorization"] == "Bearer access-token"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert captured["payload"]["model"] == "claude-sonnet-4-5"
    assert captured["payload"]["messages"][0] == {"role": "user", "content": "ciao"}


def test_anthropic_backend_maps_tool_use_events(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=(
                b"data: "
                + json.dumps(
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "query_tree_dataset",
                            "input": {"natural_query": "top specie"},
                        },
                    }
                ).encode()
                + b"\n\n"
                b'data: {"type":"content_block_stop","index":0}\n\n'
                b'data: {"type":"message_stop"}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    _mock_client(monkeypatch, handler)

    model = ClaudeOAuthChatModel(
        model_name="claude-sonnet-4-5",
        access_token="access-token",
    )
    message = model.invoke([HumanMessage(content="top specie")])

    assert message.tool_calls == [
        {
            "name": "query_tree_dataset",
            "args": {"natural_query": "top specie"},
            "id": "toolu_1",
            "type": "tool_call",
        }
    ]


def test_anthropic_backend_reports_non_success_stream_body(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"message": "token expired"}},
            headers={"request-id": "req_123"},
        )

    _mock_client(monkeypatch, handler)

    model = ClaudeOAuthChatModel(
        model_name="claude-sonnet-4-5",
        access_token="access-token",
    )

    try:
        model.invoke([HumanMessage(content="ciao")])
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected backend error")

    assert "Anthropic backend error 401" in message
    assert "token expired" in message
    assert "req_123" in message


def test_resolve_claude_oauth_model_keeps_supported_models():
    for model in CLAUDE_OAUTH_SUPPORTED_MODELS:
        assert resolve_claude_oauth_model(model) == model


def test_resolve_claude_oauth_model_falls_back_for_unsupported_models(caplog):
    caplog.set_level("WARNING", logger="streamlit_app.llm.anthropic_backend")

    for unsupported in ("claude-2", "gpt-5.5", "", None):
        assert resolve_claude_oauth_model(unsupported) == CLAUDE_OAUTH_DEFAULT_MODEL

    assert any(
        "is not supported by the Claude OAuth backend" in record.message
        for record in caplog.records
    )
