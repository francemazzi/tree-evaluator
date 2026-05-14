from __future__ import annotations

from typing import List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from streamlit_app.agent.prompts import SystemPrompts


def prepare_messages_for_llm(
    messages: List[BaseMessage],
    language: str,
    memory_section: str = "",
) -> List[BaseMessage]:
    """Prepare messages for an LLM call with a fresh system prompt and truncation."""
    messages = [message for message in messages if not isinstance(message, SystemMessage)]
    system_prompt = SystemPrompts.get_system_prompt(language)
    if memory_section:
        system_prompt = system_prompt + "\n\n" + memory_section

    truncate_label = "... [truncated]" if language == "en" else "... [troncato]"
    return build_minimal_messages([SystemMessage(content=system_prompt)] + list(messages), truncate_label)


def build_minimal_messages(
    messages: List[BaseMessage],
    truncate_label: str,
) -> List[BaseMessage]:
    """Build a compact message list for tool-calling LLMs."""
    system_messages = [message for message in messages if isinstance(message, SystemMessage)]
    last_user_msg = _last_message_of_type(messages, HumanMessage)
    last_ai_with_tools, tool_messages_after_ai = _last_tool_exchange(messages)

    minimal_messages: List[BaseMessage] = []
    minimal_messages.extend(system_messages[:1])
    if last_user_msg:
        minimal_messages.append(last_user_msg)
    if last_ai_with_tools:
        minimal_messages.append(last_ai_with_tools)
        minimal_messages.extend(tool_messages_after_ai)

    return [
        truncate_message(message, 1500 if isinstance(message, ToolMessage) else 400, truncate_label)
        for message in minimal_messages
    ]


def truncate_message(msg: BaseMessage, max_len: int, truncate_label: str) -> BaseMessage:
    """Return a copy of a LangChain message with bounded textual content."""
    content = (msg.content or "") if hasattr(msg, "content") else ""
    if len(content) > max_len:
        content = content[:max_len] + truncate_label
    if isinstance(msg, HumanMessage):
        return HumanMessage(content=content)
    if isinstance(msg, AIMessage):
        return AIMessage(content=content, tool_calls=getattr(msg, "tool_calls", None) or [])
    if isinstance(msg, SystemMessage):
        return SystemMessage(content=content)
    if isinstance(msg, ToolMessage):
        return ToolMessage(
            content=content,
            tool_call_id=getattr(msg, "tool_call_id", ""),
            name=getattr(msg, "name", ""),
        )
    return msg


def _last_message_of_type(messages: List[BaseMessage], cls):
    for message in reversed(messages):
        if isinstance(message, cls):
            return message
    return None


def _last_tool_exchange(messages: List[BaseMessage]) -> tuple[AIMessage | None, list[ToolMessage]]:
    last_ai_with_tools = None
    tool_messages_after_ai: list[ToolMessage] = []

    for index, message in enumerate(messages):
        if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
            last_ai_with_tools = message
            tool_messages_after_ai = []
            for next_message in messages[index + 1:]:
                if isinstance(next_message, ToolMessage):
                    tool_messages_after_ai.append(next_message)
                elif isinstance(next_message, AIMessage):
                    break

    return last_ai_with_tools, tool_messages_after_ai
