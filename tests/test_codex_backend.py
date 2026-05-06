from __future__ import annotations

import json

import httpx
from langchain_core.messages import HumanMessage

from streamlit_app.llm import codex_backend
from streamlit_app.llm.codex_backend import (
    CODEX_OAUTH_DEFAULT_MODEL,
    CODEX_OAUTH_SUPPORTED_MODELS,
    ChatGPTCodexBackendChatModel,
    DeterministicHashEmbeddings,
    resolve_codex_oauth_model,
)


def _mock_client(monkeypatch, handler):
    real_client = httpx.Client
    transport = httpx.MockTransport(handler)

    def client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(codex_backend.httpx, "Client", client_factory)


def test_codex_backend_posts_responses_request_and_reads_sse(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["headers"] = request.headers
        captured["payload"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            content=(
                b'data: {"type":"response.output_text.delta","delta":"ciao"}\n\n'
                b"data: "
                + json.dumps(
                    {
                        "type": "response.output_item.done",
                        "item": {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "ciao"}],
                        },
                    }
                ).encode()
                + b"\n\n"
                b'data: {"type":"response.completed"}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    _mock_client(monkeypatch, handler)

    model = ChatGPTCodexBackendChatModel(
        model_name="gpt-5.3-codex",
        access_token="access-token",
        account_id="account-123",
        base_url="https://chatgpt.com/backend-api/codex",
    )
    message = model.invoke([HumanMessage(content="ciao")])

    assert message.content == "ciao"
    assert captured["path"] == "/backend-api/codex/responses"
    assert captured["headers"]["authorization"] == "Bearer access-token"
    assert captured["headers"]["chatgpt-account-id"] == "account-123"
    assert captured["headers"]["originator"] == "codex_cli_rs"
    assert captured["payload"]["model"] == "gpt-5.3-codex"
    assert captured["payload"]["input"][0]["role"] == "user"


def test_codex_backend_maps_function_call_events(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=(
                b"data: "
                + json.dumps(
                    {
                        "type": "response.output_item.done",
                        "item": {
                            "type": "function_call",
                            "name": "query_tree_dataset",
                            "call_id": "call_1",
                            "arguments": json.dumps({"natural_query": "top specie"}),
                        },
                    }
                ).encode()
                + b"\n\n"
                b'data: {"type":"response.completed"}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    _mock_client(monkeypatch, handler)

    model = ChatGPTCodexBackendChatModel(
        model_name="gpt-5.3-codex",
        access_token="access-token",
    )
    message = model.invoke([HumanMessage(content="top specie")])

    assert message.tool_calls == [
        {
            "name": "query_tree_dataset",
            "args": {"natural_query": "top specie"},
            "id": "call_1",
            "type": "tool_call",
        }
    ]


def test_codex_backend_reports_non_success_stream_body(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"message": "token expired"}},
            headers={"x-request-id": "req_123"},
        )

    _mock_client(monkeypatch, handler)

    model = ChatGPTCodexBackendChatModel(
        model_name="gpt-5.3-codex",
        access_token="access-token",
    )

    try:
        model.invoke([HumanMessage(content="ciao")])
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected backend error")

    assert "Codex backend error 401" in message
    assert "token expired" in message
    assert "req_123" in message


def test_resolve_codex_oauth_model_keeps_supported_models():
    for model in CODEX_OAUTH_SUPPORTED_MODELS:
        assert resolve_codex_oauth_model(model) == model


def test_resolve_codex_oauth_model_falls_back_for_platform_only_models(caplog):
    caplog.set_level("WARNING", logger="streamlit_app.llm.codex_backend")

    for unsupported in ("gpt-5", "gpt-5-mini", "gpt-4o", "o1", "", None):
        assert resolve_codex_oauth_model(unsupported) == CODEX_OAUTH_DEFAULT_MODEL

    # Empty/None inputs short-circuit to the default without warning,
    # but at least one warning must be emitted for the platform-only models.
    assert any(
        "is not supported by the ChatGPT/Codex OAuth backend" in record.message
        for record in caplog.records
    )


def test_hash_embeddings_are_stable_and_local():
    embeddings = DeterministicHashEmbeddings(dimension=16)

    first = embeddings.embed_query("Tilia cordata")
    second = embeddings.embed_query("Tilia cordata")
    other = embeddings.embed_query("Acer campestre")

    assert first == second
    assert first != other
    assert len(first) == 16
