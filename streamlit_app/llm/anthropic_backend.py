from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, Iterable, Iterator, List, Optional

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field, SecretStr

logger = logging.getLogger(__name__)


ANTHROPIC_MESSAGES_URL = os.getenv("ANTHROPIC_MESSAGES_URL", "https://api.anthropic.com/v1/messages")
ANTHROPIC_VERSION = os.getenv("ANTHROPIC_VERSION", "2023-06-01")
ANTHROPIC_OAUTH_BETA = os.getenv("ANTHROPIC_OAUTH_BETA", "oauth-2025-04-20")
CLAUDE_OAUTH_DEFAULT_MODEL = "claude-sonnet-4-5"
CLAUDE_OAUTH_SUPPORTED_MODELS: tuple[str, ...] = (
    "claude-sonnet-4-5",
    "claude-opus-4-5",
    "claude-haiku-4-5",
    "claude-3-7-sonnet-latest",
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-latest",
)


def resolve_claude_oauth_model(model: Optional[str]) -> str:
    candidate = (model or "").strip()
    if not candidate:
        return CLAUDE_OAUTH_DEFAULT_MODEL
    if candidate in CLAUDE_OAUTH_SUPPORTED_MODELS:
        return candidate
    logger.warning(
        "Model %r is not supported by the Claude OAuth backend; falling back to %r.",
        candidate,
        CLAUDE_OAUTH_DEFAULT_MODEL,
    )
    return CLAUDE_OAUTH_DEFAULT_MODEL


class ClaudeOAuthChatModel(BaseChatModel):
    """LangChain chat model backed by Anthropic Messages with Claude OAuth."""

    model_name: str
    access_token: SecretStr
    base_url: str = ANTHROPIC_MESSAGES_URL
    timeout_s: float = 120.0
    temperature: float = 1.0
    max_tokens: int = 4096
    bound_tools: List[dict[str, Any]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "claude-oauth-backend"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "base_url": self.base_url}

    def bind_tools(self, tools: Iterable[Any], **kwargs: Any) -> "ClaudeOAuthChatModel":
        tool_choice = kwargs.get("tool_choice")
        converted_tools = [_to_anthropic_tool(tool) for tool in tools]
        if tool_choice not in (None, "auto"):
            logger.debug("Ignoring unsupported Claude OAuth tool_choice=%r", tool_choice)
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
        system_parts: list[str] = []
        anthropic_messages: list[dict[str, Any]] = []

        for message in messages:
            if isinstance(message, SystemMessage):
                text = _message_text(message.content)
                if text:
                    system_parts.append(text)
                continue

            item = _message_to_anthropic(message)
            if item:
                anthropic_messages.append(item)

        payload: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": anthropic_messages,
            "stream": True,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if self.bound_tools:
            payload["tools"] = self.bound_tools
            payload["tool_choice"] = {"type": "auto"}
        return payload

    def _stream_response(self, payload: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
        text_chunks: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        pending_tool_blocks: dict[int, dict[str, Any]] = {}

        with httpx.Client(timeout=self.timeout_s) as client:
            with client.stream(
                "POST",
                self.base_url,
                json=payload,
                headers=self._headers(),
            ) as response:
                if response.status_code < 200 or response.status_code >= 300:
                    response.read()
                    raise ValueError(_format_backend_error(response))

                for event in _iter_sse_events(response.iter_lines()):
                    event_type = str(event.get("type") or "")
                    if event_type == "content_block_start":
                        _collect_content_block_start(event, pending_tool_blocks)
                    elif event_type == "content_block_delta":
                        _collect_content_block_delta(event, text_chunks, pending_tool_blocks)
                    elif event_type == "content_block_stop":
                        _flush_tool_block(event, pending_tool_blocks, tool_calls)
                    elif event_type == "message_stop":
                        break
                    elif event_type == "error":
                        raise ValueError(_format_failed_event(event))

        for index in sorted(pending_tool_blocks):
            _append_tool_call(pending_tool_blocks[index], tool_calls)
        return "".join(text_chunks), tool_calls

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.access_token.get_secret_value()}",
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "anthropic-version": ANTHROPIC_VERSION,
        }
        if ANTHROPIC_OAUTH_BETA:
            headers["anthropic-beta"] = ANTHROPIC_OAUTH_BETA
        return headers


def _to_anthropic_tool(tool: Any) -> dict[str, Any]:
    converted = convert_to_openai_tool(tool)
    function = converted.get("function") if converted.get("type") == "function" else converted
    if not isinstance(function, dict):
        function = {}
    return {
        "name": str(function.get("name") or ""),
        "description": str(function.get("description") or ""),
        "input_schema": function.get("parameters") or {"type": "object", "properties": {}},
    }


def _message_to_anthropic(message: BaseMessage) -> dict[str, Any]:
    if isinstance(message, HumanMessage):
        text = _message_text(message.content)
        return {"role": "user", "content": text} if text else {}

    if isinstance(message, ToolMessage):
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": str(getattr(message, "tool_call_id", "") or ""),
                    "content": _message_text(message.content),
                }
            ],
        }

    if isinstance(message, AIMessage):
        content: list[dict[str, Any]] = []
        text = _message_text(message.content)
        if text:
            content.append({"type": "text", "text": text})
        for tool_call in getattr(message, "tool_calls", None) or []:
            content.append(
                {
                    "type": "tool_use",
                    "id": str(tool_call.get("id") or uuid.uuid4().hex),
                    "name": str(tool_call.get("name") or ""),
                    "input": tool_call.get("args") or {},
                }
            )
        return {"role": "assistant", "content": content} if content else {}

    text = _message_text(message.content)
    return {"role": "user", "content": text} if text else {}


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
                yield from _decode_sse_payload(data_lines)
                data_lines = []
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].strip())

    if data_lines:
        yield from _decode_sse_payload(data_lines)


def _decode_sse_payload(data_lines: list[str]) -> Iterator[dict[str, Any]]:
    payload = "\n".join(data_lines)
    if payload == "[DONE]":
        return
    try:
        event = json.loads(payload)
    except json.JSONDecodeError:
        logger.debug("Ignoring non-JSON Anthropic SSE payload: %s", payload)
        return
    if isinstance(event, dict):
        yield event


def _collect_content_block_start(event: dict[str, Any], pending_tool_blocks: dict[int, dict[str, Any]]) -> None:
    index = int(event.get("index") or 0)
    block = event.get("content_block")
    if not isinstance(block, dict) or block.get("type") != "tool_use":
        return
    pending_tool_blocks[index] = {
        "id": str(block.get("id") or uuid.uuid4().hex),
        "name": str(block.get("name") or ""),
        "input": block.get("input") if isinstance(block.get("input"), dict) else {},
        "partial_json": "",
    }


def _collect_content_block_delta(
    event: dict[str, Any],
    text_chunks: list[str],
    pending_tool_blocks: dict[int, dict[str, Any]],
) -> None:
    delta = event.get("delta")
    if not isinstance(delta, dict):
        return
    delta_type = str(delta.get("type") or "")
    if delta_type == "text_delta":
        text = str(delta.get("text") or "")
        if text:
            text_chunks.append(text)
        return
    if delta_type == "input_json_delta":
        index = int(event.get("index") or 0)
        tool_block = pending_tool_blocks.setdefault(
            index,
            {"id": uuid.uuid4().hex, "name": "", "input": {}, "partial_json": ""},
        )
        tool_block["partial_json"] = str(tool_block.get("partial_json") or "") + str(
            delta.get("partial_json") or ""
        )


def _flush_tool_block(
    event: dict[str, Any],
    pending_tool_blocks: dict[int, dict[str, Any]],
    tool_calls: list[dict[str, Any]],
) -> None:
    index = int(event.get("index") or 0)
    tool_block = pending_tool_blocks.pop(index, None)
    if not tool_block:
        return
    _append_tool_call(tool_block, tool_calls)


def _append_tool_call(tool_block: dict[str, Any], tool_calls: list[dict[str, Any]]) -> None:
    partial_json = str(tool_block.get("partial_json") or "")
    args = tool_block.get("input") if isinstance(tool_block.get("input"), dict) else {}
    if partial_json:
        try:
            parsed = json.loads(partial_json)
            args = parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            args = {"__raw_arguments": partial_json}
    tool_calls.append(
        {
            "name": str(tool_block.get("name") or ""),
            "args": args,
            "id": str(tool_block.get("id") or uuid.uuid4().hex),
        }
    )


def _format_backend_error(response: httpx.Response) -> str:
    request_id = response.headers.get("request-id") or response.headers.get("x-request-id")
    try:
        detail = response.text.strip() or response.reason_phrase
    except httpx.ResponseNotRead:
        detail = response.reason_phrase
    suffix = f" request_id={request_id}" if request_id else ""
    return f"Anthropic backend error {response.status_code}: {detail}{suffix}"


def _format_failed_event(event: dict[str, Any]) -> str:
    error = event.get("error")
    if isinstance(error, dict):
        message = error.get("message") or error.get("type") or error
        return f"Anthropic backend failed: {message}"
    return f"Anthropic backend failed: {event}"


def _apply_stop_tokens(content: str, stop: list[str]) -> str:
    cut = len(content)
    for token in stop:
        index = content.find(token)
        if index >= 0:
            cut = min(cut, index)
    return content[:cut]
