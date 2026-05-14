from __future__ import annotations

from typing import List, Optional

from langchain_core.messages import AIMessage, HumanMessage


def chat(agent, message: str, history: Optional[List[dict]] = None) -> str:
    """Run a synchronous chat turn for a TreeEvaluatorAgent instance."""
    messages = agent._convert_history_to_messages(history)
    messages.append(HumanMessage(content=message))
    agent._transcript.log_user_message(message)

    initial_state = agent._create_initial_state(messages)
    result = agent._graph.invoke(initial_state, config={"recursion_limit": 30})

    final_message = result["messages"][-1]
    response_text = (
        final_message.content
        if isinstance(final_message, AIMessage)
        else str(final_message.content)
    )
    agent._transcript.log_agent_response(response_text[:300] if response_text else "")
    agent._save_response_facts(response_text)
    return response_text


def stream_chat(agent, message: str, history: Optional[List[dict]] = None):
    """Stream a chat turn for a TreeEvaluatorAgent instance."""
    messages = agent._convert_history_to_messages(history)
    messages.append(HumanMessage(content=message))
    agent._transcript.log_user_message(message)

    final_response = None
    chart_data_json = None
    map_data_json = None
    initial_state = agent._create_initial_state(messages)

    for event in agent._graph.stream(
        initial_state,
        config={"recursion_limit": 30},
        stream_mode="updates",
    ):
        current_language = agent._interface_language
        if "detected_language" in event.get("query_optimizer", {}):
            current_language = event["query_optimizer"].get(
                "detected_language", agent._interface_language
            )

        for node_name, node_output in event.items():
            result = agent._process_stream_event(
                node_name, node_output, current_language, len(messages)
            )
            if not result:
                continue

            if result.get("type") == "final_response":
                final_response = result.get("content")
            elif "final_response" in result:
                final_response = result.pop("final_response")
                if result.get("type"):
                    yield result
            elif result.get("type") == "chart_data":
                chart_data_json = result.get("data")
            elif result.get("type") == "map_data":
                map_data_json = result.get("data")
            elif result.get("type") == "reasoning":
                yield result

    if final_response:
        agent._transcript.log_agent_response(final_response[:300])

        if chart_data_json and "CHART_DATA_START" not in final_response:
            final_response = f"{final_response}\n\nCHART_DATA_START\n{chart_data_json}\nCHART_DATA_END"

        if map_data_json and "MAP_DATA_START" not in final_response:
            final_response = f"{final_response}\n\nMAP_DATA_START\n{map_data_json}\nMAP_DATA_END"

        agent._save_response_facts(final_response)
        yield {"type": "response", "content": final_response}
