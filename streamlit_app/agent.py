from __future__ import annotations

import json
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
    optimized_query: Optional[str]
    tasks: Optional[List[str]]
    validation_result: Optional[dict]


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

        # Initialize LLM (used by tools for text-to-SQL translation)
        self._base_llm = ChatOpenAI(
            model="gpt-5",
            temperature=1,  # Low temperature for SQL generation
            api_key=api_key,
        )

        # Initialize tools with LLM
        self._tools = [
            CO2CalculationTool(),
            EnvironmentEstimationTool(),
            DatasetQueryTool(llm=self._base_llm),
        ]

        # Initialize LLM with tools bound
        self._llm = self._base_llm.bind_tools(self._tools)

        # Build graph
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow with query optimization and validation."""
        workflow = StateGraph(AgentState)

        # Define nodes
        workflow.add_node("query_optimizer", self._optimize_query)
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", ToolNode(self._tools))
        workflow.add_node("validator", self._validate_response)

        # Set entry point - start with query optimization
        workflow.set_entry_point("query_optimizer")

        # Query optimizer -> agent
        workflow.add_edge("query_optimizer", "agent")

        # Agent decides: continue to tools or validate
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "validate": "validator",
            },
        )

        # After tool execution, return to agent
        workflow.add_edge("tools", "agent")

        # Validator decides: complete or retry
        workflow.add_conditional_edges(
            "validator",
            self._should_retry,
            {
                "complete": END,
                "retry": "agent",
            },
        )

        return workflow.compile()

    def _optimize_query(self, state: AgentState) -> dict:
        """Optimize user query and break it into tasks."""
        messages = state["messages"]
        
        # Get the last user message
        last_user_msg = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_user_msg = msg.content
                break
        
        if not last_user_msg:
            return {"optimized_query": None, "tasks": []}
        
        # Use LLM to optimize query and create tasks
        optimizer_prompt = f"""Analizza la seguente domanda dell'utente e:
1. Riformulala in modo più chiaro e specifico
2. Scomponila in task specifici e ordinati

Domanda originale: {last_user_msg}

Rispondi in formato JSON con:
- "optimized_query": la domanda ottimizzata
- "tasks": lista di task specifici da completare

Esempio:
{{
    "optimized_query": "Calcola il sequestro di CO2 per un albero di Acer di 30cm DBH e 15m altezza",
    "tasks": [
        "Identificare la specie (Acer) e la densità del legno appropriata (0.56 g/cm³)",
        "Calcolare il volume utilizzando DBH (30cm) e altezza (15m)",
        "Calcolare la biomassa e il sequestro di CO2",
        "Presentare i risultati in modo chiaro"
    ]
}}"""
        
        try:
            # Create a temporary LLM without tools for optimization
            optimizer_llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.3,
                api_key=self._llm.client.api_key,
            )
            
            response = optimizer_llm.invoke([HumanMessage(content=optimizer_prompt)])
            
            # Parse JSON response
            response_text = response.content.strip()
            
            # Extract JSON from markdown code blocks if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            optimization_result = json.loads(response_text)
            
            optimized_query = optimization_result.get("optimized_query", last_user_msg)
            tasks = optimization_result.get("tasks", [])
            
            # Add optimization info as a system message
            optimization_msg = SystemMessage(
                content=f"""Query ottimizzata: {optimized_query}

Task da completare:
{chr(10).join(f'{i+1}. {task}' for i, task in enumerate(tasks))}"""
            )
            
            return {
                "messages": [optimization_msg],
                "optimized_query": optimized_query,
                "tasks": tasks,
            }
            
        except Exception as e:
            # If optimization fails, continue with original query
            return {
                "optimized_query": last_user_msg,
                "tasks": [last_user_msg],
            }

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

    def _should_continue(self, state: AgentState) -> Literal["continue", "validate"]:
        """Determine if we should continue to tools or validate response."""
        messages = state["messages"]
        last_message = messages[-1]

        # If the LLM makes a tool call, continue to tools
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "continue"

        # Otherwise, validate the response
        return "validate"
    
    def _validate_response(self, state: AgentState) -> dict:
        """Validate if the response adequately answers the user's question."""
        messages = state["messages"]
        tasks = state.get("tasks", [])
        optimized_query = state.get("optimized_query", "")
        
        # Get original user question
        user_question = None
        for msg in messages:
            if isinstance(msg, HumanMessage) and not msg.content.startswith("Query ottimizzata"):
                user_question = msg.content
                break
        
        # Get agent's response
        agent_response = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                agent_response = msg.content
                break
        
        if not user_question or not agent_response:
            return {"validation_result": {"is_complete": True, "feedback": ""}}
        
        # Create validation prompt
        validation_prompt = f"""Valuta se la seguente risposta risponde adeguatamente alla domanda dell'utente.

Domanda originale: {user_question}

Query ottimizzata: {optimized_query}

Task da completare:
{chr(10).join(f'{i+1}. {task}' for i, task in enumerate(tasks))}

Risposta fornita: {agent_response}

Analizza se:
1. Tutti i task sono stati completati
2. La risposta è accurata e completa
3. La risposta risponde effettivamente alla domanda

Rispondi in formato JSON:
{{
    "is_complete": true/false,
    "completed_tasks": ["lista", "dei", "task", "completati"],
    "missing_tasks": ["lista", "dei", "task", "mancanti"],
    "feedback": "breve feedback su cosa manca o cosa migliorare (se incompleto)"
}}"""
        
        try:
            # Create validator LLM
            validator_llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.2,
                api_key=self._llm.client.api_key,
            )
            
            response = validator_llm.invoke([HumanMessage(content=validation_prompt)])
            
            # Parse JSON response
            response_text = response.content.strip()
            
            # Extract JSON from markdown code blocks if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            validation_result = json.loads(response_text)
            
            # If incomplete, add feedback as system message for retry
            if not validation_result.get("is_complete", True):
                feedback_msg = SystemMessage(
                    content=f"""⚠️ Validazione risposta:
Task mancanti: {', '.join(validation_result.get('missing_tasks', []))}

Feedback: {validation_result.get('feedback', '')}

Per favore, completa la risposta affrontando i task mancanti."""
                )
                return {
                    "messages": [feedback_msg],
                    "validation_result": validation_result,
                }
            
            return {"validation_result": validation_result}
            
        except Exception as e:
            # If validation fails, assume complete
            return {"validation_result": {"is_complete": True, "feedback": ""}}
    
    def _should_retry(self, state: AgentState) -> Literal["complete", "retry"]:
        """Determine if we should retry or complete based on validation."""
        validation_result = state.get("validation_result", {})
        
        # Check if response is complete
        is_complete = validation_result.get("is_complete", True)
        
        if is_complete:
            return "complete"
        else:
            return "retry"

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
        """Stream chat response with LangGraph streaming, including reasoning steps.

        Args:
            message: User message
            history: Optional chat history

        Yields:
            Dict with 'type' and 'content' keys:
            - type: 'reasoning' (internal step) or 'response' (final answer)
            - content: the actual content to display
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

        # Track execution
        final_response = None
        retry_count = 0
        max_retries = 2

        # Stream from graph with updates mode to see each node
        for event in self._graph.stream({"messages": messages}, stream_mode="updates"):
            # event is a dict with node_name: node_output
            for node_name, node_output in event.items():
                
                # Emit reasoning for each node
                if node_name == "query_optimizer":
                    optimized = node_output.get("optimized_query", "")
                    tasks = node_output.get("tasks", [])
                    if optimized:
                        reasoning = f"🔍 **Ottimizzazione Query**\n\n"
                        reasoning += f"Query ottimizzata: *{optimized}*\n\n"
                        if tasks:
                            reasoning += "**Task identificati:**\n"
                            for i, task in enumerate(tasks, 1):
                                reasoning += f"{i}. {task}\n"
                        yield {"type": "reasoning", "content": reasoning}
                
                elif node_name == "agent":
                    # Check if agent is calling tools or responding
                    node_messages = node_output.get("messages", [])
                    if node_messages:
                        last_msg = node_messages[-1]
                        if isinstance(last_msg, AIMessage):
                            if last_msg.tool_calls:
                                # Agent is calling tools
                                reasoning = f"🛠️ **Chiamata Tool**\n\n"
                                for tool_call in last_msg.tool_calls:
                                    tool_name = tool_call.get("name", "unknown")
                                    reasoning += f"- Utilizzo: `{tool_name}`\n"
                                yield {"type": "reasoning", "content": reasoning}
                            elif last_msg.content and not last_msg.tool_calls:
                                # Agent has a response (might be intermediate or final)
                                final_response = last_msg.content
                
                elif node_name == "tools":
                    # Tool execution completed
                    reasoning = f"✅ **Tool Eseguito**\n\nElaborazione risultati...\n"
                    yield {"type": "reasoning", "content": reasoning}
                
                elif node_name == "validator":
                    validation = node_output.get("validation_result", {})
                    is_complete = validation.get("is_complete", True)
                    
                    if is_complete:
                        reasoning = f"✓ **Validazione Completata**\n\nLa risposta è completa e accurata.\n"
                        yield {"type": "reasoning", "content": reasoning}
                    else:
                        retry_count += 1
                        if retry_count > max_retries:
                            reasoning = f"⚠️ **Validazione**\n\nRaggiunto limite retry. Proseguo con la risposta attuale.\n"
                            yield {"type": "reasoning", "content": reasoning}
                            break
                        
                        missing = validation.get("missing_tasks", [])
                        feedback = validation.get("feedback", "")
                        reasoning = f"⚠️ **Validazione (Tentativo {retry_count})**\n\n"
                        if missing:
                            reasoning += f"Task mancanti: {', '.join(missing)}\n"
                        if feedback:
                            reasoning += f"\n{feedback}\n"
                        reasoning += "\nRielaborazione risposta...\n"
                        yield {"type": "reasoning", "content": reasoning}
        
        # Yield final response
        if final_response:
            yield {"type": "response", "content": final_response}

