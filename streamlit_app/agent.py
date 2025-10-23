from __future__ import annotations

import os
from typing import Annotated, List, Literal, Optional, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from streamlit_app.tools.co2_tool import CO2CalculationTool
from streamlit_app.tools.dataset_tool import DatasetQueryTool
from streamlit_app.tools.environment_tool import EnvironmentEstimationTool

# Load environment variables
load_dotenv()


class AgentState(TypedDict):
    """State for the LangGraph agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]


class TreeEvaluatorAgent:
    """LangGraph-based agent that orchestrates tree evaluation tools."""

    def __init__(self, openai_api_key: Optional[str] = None) -> None:
        """Initialize the agent with tools and LLM.

        Args:
            openai_api_key: OpenAI API key. If not provided, tries OPENAI_API_KEY env var.
        """
        # Get API key - prioritize parameter, then env var
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Provide it via UI Settings or set OPENAI_API_KEY environment variable."
            )

        # Initialize tools
        self._tools = [
            CO2CalculationTool(),
            EnvironmentEstimationTool(),
            DatasetQueryTool(),
        ]

        # Initialize LLM with tools
        self._llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            api_key=api_key,
        ).bind_tools(self._tools)

        # Build graph
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)

        # Define nodes
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", ToolNode(self._tools))

        # Set entry point
        workflow.set_entry_point("agent")

        # Define edges
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END,
            },
        )

        # After tool execution, always return to agent
        workflow.add_edge("tools", "agent")

        return workflow.compile()

    def _call_model(self, state: AgentState) -> dict:
        """Call the LLM model."""
        messages = state["messages"]

        # Add system message if not present
        if not any(isinstance(m, SystemMessage) for m in messages):
            system_msg = SystemMessage(
                content="""You are a helpful tree evaluation assistant with access to:

1. **CO2 Calculation Tool**: Calculate CO2 sequestration and biomass for individual trees given their measurements.
2. **Environmental Estimation Tool**: Compute volume, biomass, and carbon stock using alternative formulas.
3. **Dataset Query Tool**: Query a real Vienna trees dataset (BAUMKATOGD) with filtering, aggregation, and statistics.

Guidelines:
- When users ask about CO2 or carbon sequestration for specific measurements, use the CO2 calculation tool.
- When users ask about the dataset (counts, species, districts, statistics), use the dataset query tool.
- Always provide clear, helpful responses in Italian.
- If you need more information, ask the user.
- When using tools, explain the results in a user-friendly way.
- For wood density, use species-specific values if known, otherwise default to 0.6 g/cm³.

Common wood densities (g/cm³):
- Acer (Acero): 0.56
- Tilia (Tiglio): 0.49
- Carpinus (Carpino): 0.75
- Gleditsia: 0.62
- Aesculus (Ippocastano): 0.53
- Quercus (Quercia): 0.75
- Fraxinus (Frassino): 0.69
- Betula (Betulla): 0.65
"""
            )
            messages = [system_msg] + list(messages)

        response = self._llm.invoke(messages)
        return {"messages": [response]}

    def _should_continue(self, state: AgentState) -> Literal["continue", "end"]:
        """Determine if we should continue to tools or end."""
        messages = state["messages"]
        last_message = messages[-1]

        # If the LLM makes a tool call, continue to tools
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "continue"

        # Otherwise, end
        return "end"

    def chat(self, message: str, history: Optional[List[dict]] = None) -> str:
        """Chat with the agent.

        Args:
            message: User message
            history: Optional chat history as list of dicts with 'role' and 'content' keys

        Returns:
            Agent response as string
        """
        # Convert history to messages
        messages: List[BaseMessage] = []
        if history:
            for msg in history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))

        # Add current message
        messages.append(HumanMessage(content=message))

        # Run graph
        result = self._graph.invoke({"messages": messages})

        # Extract final response
        final_message = result["messages"][-1]
        if isinstance(final_message, AIMessage):
            return final_message.content

        return str(final_message.content)

    def stream_chat(self, message: str, history: Optional[List[dict]] = None):
        """Stream chat response with LangGraph streaming.

        Args:
            message: User message
            history: Optional chat history

        Yields:
            Chunks of the response as they are generated
        """
        # Convert history to messages
        messages: List[BaseMessage] = []
        if history:
            for msg in history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))

        # Add current message
        messages.append(HumanMessage(content=message))

        # Stream from graph
        for event in self._graph.stream({"messages": messages}, stream_mode="values"):
            # Get the last message in the state
            last_message = event["messages"][-1]
            
            # If it's an AIMessage with content, yield it
            if isinstance(last_message, AIMessage) and last_message.content:
                # Check if it's the final response (not a tool call)
                if not last_message.tool_calls:
                    yield last_message.content

