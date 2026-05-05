from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from streamlit_app.agent.core import TreeEvaluatorAgent


def test_pending_tool_calls_extracts_latest_ai_tool_calls() -> None:
    older = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "generate_chart",
                "args": {"chart_type": "bar"},
                "id": "older-call",
                "type": "tool_call",
            }
        ],
    )
    latest = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "query_tree_dataset",
                "args": {"natural_query": "quanti alberi ci sono?"},
                "id": "latest-call",
                "type": "tool_call",
            }
        ],
    )

    pending = TreeEvaluatorAgent._pending_tool_calls(
        [HumanMessage(content="crea un grafico"), older, latest]
    )

    assert pending == latest.tool_calls
