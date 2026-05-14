from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from typing import Any, Iterable, Iterator, List, Optional

import httpx
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field, SecretStr

logger = logging.getLogger(__name__)


CODEX_BACKEND_BASE_URL = os.getenv(
    "OPENAI_CODEX_BACKEND_BASE_URL",
    "https://chatgpt.com/backend-api/codex",
)
CODEX_CLIENT_VERSION = os.getenv("OPENAI_CODEX_CLIENT_VERSION", "0.128.0")
CODEX_ORIGINATOR = os.getenv("OPENAI_CODEX_ORIGINATOR", "codex_cli_rs")

# Models accepted by the ChatGPT backend when authenticating via Codex OAuth.
# Keep `CODEX_OAUTH_DEFAULT_MODEL` in sync with the upstream Codex CLI default.
# Reference: https://developers.openai.com/codex/models
CODEX_OAUTH_DEFAULT_MODEL = "gpt-5.5"
CODEX_OAUTH_SUPPORTED_MODELS: tuple[str, ...] = (
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.3-codex-spark",
    "gpt-5.3-codex",
    "gpt-5.2-codex",
)


def resolve_codex_oauth_model(model: Optional[str]) -> str:
    """Return a Codex/OAuth-compatible model name, falling back to the default."""
    candidate = (model or "").strip()
    if not candidate:
        return CODEX_OAUTH_DEFAULT_MODEL
    if candidate in CODEX_OAUTH_SUPPORTED_MODELS:
        return candidate
    logger.warning(
        "Model %r is not supported by the ChatGPT/Codex OAuth backend; "
        "falling back to %r.",
        candidate,
        CODEX_OAUTH_DEFAULT_MODEL,
    )
    return CODEX_OAUTH_DEFAULT_MODEL


class ChatGPTCodexBackendChatModel(BaseChatModel):
    """LangChain chat model backed by the ChatGPT Codex Responses endpoint."""

    model_name: str
    access_token: SecretStr
    account_id: Optional[str] = None
    is_fedramp_account: bool = False
    base_url: str = CODEX_BACKEND_BASE_URL
    timeout_s: float = 120.0
    bound_tools: List[dict[str, Any]] = Field(default_factory=list)
    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    thread_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    installation_id: str = Field(default_factory=lambda: uuid.uuid4().hex)

    @property
    def _llm_type(self) -> str:
        return "chatgpt-codex-backend"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "base_url": self.base_url}

    def bind_tools(self, tools: Iterable[Any], **kwargs: Any) -> "ChatGPTCodexBackendChatModel":
        tool_choice = kwargs.get("tool_choice")
        converted_tools = [_to_responses_tool(tool) for tool in tools]
        if tool_choice not in (None, "auto"):
            logger.debug("Ignoring unsupported Codex backend tool_choice=%r", tool_choice)
        return self.model_copy(update={"bound_tools": converted_tools})

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = self._build_payload(messages)
        content, tool_calls = self._stream_response(payload)
        if stop:
            content = _apply_stop_tokens(content, stop)
        message = AIMessage(content=content, tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _build_payload(self, messages: List[BaseMessage]) -> dict[str, Any]:
        instructions: list[str] = []
        input_items: list[dict[str, Any]] = []

        for message in messages:
            if isinstance(message, SystemMessage):
                text = _message_text(message.content)
                if text:
                    instructions.append(text)
                continue

            input_items.extend(_message_to_response_items(message))

        return {
            "model": self.model_name,
            "instructions": "\n\n".join(instructions),
            "input": input_items,
            "tools": self.bound_tools,
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "reasoning": None,
            "store": False,
            "stream": True,
            "include": [],
            "prompt_cache_key": self.thread_id,
            "client_metadata": {
                "x-codex-installation-id": self.installation_id,
            },
        }

    def _stream_response(self, payload: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
        url = f"{self.base_url.rstrip('/')}/responses"
        text_chunks: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        with httpx.Client(timeout=self.timeout_s) as client:
            with client.stream(
                "POST",
                url,
                json=payload,
                headers=self._headers(),
            ) as response:
                if response.status_code < 200 or response.status_code >= 300:
                    response.read()
                    raise ValueError(_format_backend_error(response))

                for event in _iter_sse_events(response.iter_lines()):
                    event_type = str(event.get("type") or "")
                    if event_type == "response.output_text.delta":
                        delta = str(event.get("delta") or "")
                        if delta:
                            text_chunks.append(delta)
                    elif event_type == "response.output_item.done":
                        item = event.get("item")
                        if isinstance(item, dict):
                            _collect_output_item(item, text_chunks, tool_calls)
                    elif event_type == "response.completed":
                        break
                    elif event_type == "response.failed":
                        raise ValueError(_format_failed_event(event))

        return "".join(text_chunks), tool_calls

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.access_token.get_secret_value()}",
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "User-Agent": f"{CODEX_ORIGINATOR}/{CODEX_CLIENT_VERSION}",
            "originator": CODEX_ORIGINATOR,
            "version": CODEX_CLIENT_VERSION,
            "session_id": self.session_id,
            "thread_id": self.thread_id,
            "x-client-request-id": self.thread_id,
        }
        if self.account_id:
            headers["ChatGPT-Account-ID"] = self.account_id
        if self.is_fedramp_account:
            headers["X-OpenAI-Fedramp"] = "true"
        return headers


class DeterministicHashEmbeddings(Embeddings):
    """Small local embedding fallback used when ChatGPT OAuth is selected."""

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        vector = [0.0] * self._dimension
        tokens = (text or "").lower().split()
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self._dimension
            sign = 1.0 if digest[4] % 2 else -1.0
            vector[index] += sign

        norm = sum(value * value for value in vector) ** 0.5
        if norm:
            vector = [value / norm for value in vector]
        return vector


def _to_responses_tool(tool: Any) -> dict[str, Any]:
    converted = convert_to_openai_tool(tool)
    if converted.get("type") != "function":
        return converted

    function = converted.get("function")
    if isinstance(function, dict):
        return {
            "type": "function",
            "name": str(function.get("name") or ""),
            "description": str(function.get("description") or ""),
            "parameters": function.get("parameters") or {"type": "object", "properties": {}},
            "strict": False,
        }

    return {
        "type": "function",
        "name": str(converted.get("name") or ""),
        "description": str(converted.get("description") or ""),
        "parameters": converted.get("parameters") or {"type": "object", "properties": {}},
        "strict": False,
    }


def _message_to_response_items(message: BaseMessage) -> list[dict[str, Any]]:
    if isinstance(message, HumanMessage):
        return [_message_item("user", _content_items(message.content, "input_text"))]

    if isinstance(message, ToolMessage):
        return [
            {
                "type": "function_call_output",
                "call_id": str(getattr(message, "tool_call_id", "") or ""),
                "output": _message_text(message.content),
            }
        ]

    if isinstance(message, AIMessage):
        items: list[dict[str, Any]] = []
        text = _message_text(message.content)
        if text:
            items.append(_message_item("assistant", _content_items(text, "output_text")))

        for tool_call in getattr(message, "tool_calls", None) or []:
            items.append(
                {
                    "type": "function_call",
                    "call_id": str(tool_call.get("id") or uuid.uuid4().hex),
                    "name": str(tool_call.get("name") or ""),
                    "arguments": json.dumps(tool_call.get("args") or {}, ensure_ascii=False),
                }
            )
        return items

    text = _message_text(message.content)
    return [_message_item("user", _content_items(text, "input_text"))] if text else []


def _message_item(role: str, content: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "type": "message",
        "role": role,
        "content": content,
    }


def _content_items(content: Any, item_type: str) -> list[dict[str, str]]:
    text = _message_text(content)
    if not text:
        return []
    return [{"type": item_type, "text": text}]


def _message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                value = item.get("text") or item.get("content")
                if value:
                    parts.append(str(value))
        return "\n".join(parts)
    return str(content)


def _iter_sse_events(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    data_lines: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if data_lines:
                payload = "\n".join(data_lines)
                data_lines = []
                if payload != "[DONE]":
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        logger.debug("Ignoring non-JSON SSE payload: %s", payload)
                    else:
                        if isinstance(event, dict):
                            yield event
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].strip())

    if data_lines:
        payload = "\n".join(data_lines)
        if payload != "[DONE]":
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                return
            if isinstance(event, dict):
                yield event


def _collect_output_item(
    item: dict[str, Any],
    text_chunks: list[str],
    tool_calls: list[dict[str, Any]],
) -> None:
    item_type = str(item.get("type") or "")
    if item_type == "function_call":
        args_text = str(item.get("arguments") or "{}")
        try:
            args = json.loads(args_text)
        except json.JSONDecodeError:
            args = {"__raw_arguments": args_text}
        tool_calls.append(
            {
                "name": str(item.get("name") or ""),
                "args": args if isinstance(args, dict) else {"value": args},
                "id": str(item.get("call_id") or item.get("id") or uuid.uuid4().hex),
            }
        )
        return

    if item_type != "message":
        return

    if text_chunks:
        return

    for content_item in item.get("content") or []:
        if not isinstance(content_item, dict):
            continue
        if content_item.get("type") in {"output_text", "text"}:
            text = str(content_item.get("text") or "")
            if text:
                text_chunks.append(text)


def _format_backend_error(response: httpx.Response) -> str:
    request_id = response.headers.get("x-request-id") or response.headers.get("cf-ray")
    try:
        detail = response.text.strip() or response.reason_phrase
    except httpx.ResponseNotRead:
        detail = response.reason_phrase
    suffix = f" request_id={request_id}" if request_id else ""
    return f"Codex backend error {response.status_code}: {detail}{suffix}"


def _format_failed_event(event: dict[str, Any]) -> str:
    response = event.get("response")
    if isinstance(response, dict):
        error = response.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("code") or error
            return f"Codex backend failed: {message}"
    return f"Codex backend failed: {event}"


def _apply_stop_tokens(content: str, stop: list[str]) -> str:
    cut = len(content)
    for token in stop:
        index = content.find(token)
        if index >= 0:
            cut = min(cut, index)
    return content[:cut]
