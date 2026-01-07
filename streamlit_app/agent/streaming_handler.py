"""Streaming handler for LangGraph agent responses."""

from __future__ import annotations

import json
from typing import Any, Dict

from langchain_core.messages import AIMessage


class StreamingHandler:
    """Handler for formatting streaming responses from the agent."""
    
    @staticmethod
    def handle_context_manager_event(node_output: Dict[str, Any], original_count: int) -> Dict[str, str]:
        """Handle context manager node events.
        
        Args:
            node_output: Output from context_manager node
            original_count: Original message count
            
        Returns:
            Dict with type and content for streaming
        """
        message_count = node_output.get("message_count", 0)
        
        if message_count < original_count:
            reasoning = "🧹 **Gestione Contesto**\n\n"
            reasoning += f"Messaggi originali: {original_count}\n"
            reasoning += f"Messaggi ottimizzati: {message_count}\n"
            reasoning += "Contesto lungo compresso per evitare limiti di token.\n"
            return {"type": "reasoning", "content": reasoning}
        return None
    
    @staticmethod
    def handle_query_optimizer_event(node_output: Dict[str, Any]) -> Dict[str, str]:
        """Handle query optimizer node events."""
        optimized = node_output.get("optimized_query", "")
        tasks = node_output.get("tasks", [])
        
        if optimized:
            reasoning = "🔍 **Ottimizzazione Query**\n\n"
            reasoning += f"Query ottimizzata: *{optimized}*\n\n"
            if tasks:
                reasoning += "**Task identificati:**\n\n<ol>\n"
                for task in tasks:
                    reasoning += f"<li>{task}</li>\n"
                reasoning += "</ol>\n\n"
            return {"type": "reasoning", "content": reasoning}
        return None
    
    @staticmethod
    def handle_agent_event(node_output: Dict[str, Any]) -> Dict[str, Any]:
        """Handle agent node events (tool calls or responses)."""
        node_messages = node_output.get("messages", [])
        if not node_messages:
            return None
        
        last_msg = node_messages[-1]
        if not isinstance(last_msg, AIMessage):
            return None
        
        if last_msg.tool_calls:
            # Agent is calling tools
            reasoning = "🛠️ **Chiamata Tool**\n\n"
            for tool_call in last_msg.tool_calls:
                tool_name = tool_call.get("name", "unknown")
                tool_args = tool_call.get("args", {})
                
                reasoning += f"- **Tool**: `{tool_name}`\n"
                reasoning += StreamingHandler._format_tool_args(tool_name, tool_args)
                reasoning += "\n"
            
            return {"type": "reasoning", "content": reasoning}
        elif last_msg.content and not last_msg.tool_calls:
            # Final or intermediate response
            return {"type": "final_response", "content": last_msg.content}
        
        return None
    
    @staticmethod
    def _format_tool_args(tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Format tool arguments for display."""
        if tool_name == "query_tree_dataset":
            natural_q = tool_args.get("natural_query", "N/A")
            return f"  - **Query**: _{natural_q}_\n"
        elif tool_name in ("calculate_co2", "estimate_environment"):
            dbh = tool_args.get("dbh_cm", "N/A")
            height = tool_args.get("height_m", "N/A")
            result = f"  - **DBH**: {dbh} cm\n  - **Altezza**: {height} m\n"
            if "wood_density" in tool_args:
                result += f"  - **Densità legno**: {tool_args['wood_density']} g/cm³\n"
            return result
        elif tool_name == "calculate_co2_aggregate":
            query = tool_args.get("natural_query", "N/A")
            return f"  - **Query**: _{query}_\n"
        elif tool_name == "generate_chart":
            chart_type = tool_args.get("chart_type", "N/A")
            return f"  - **Tipo grafico**: {chart_type}\n"
        return ""
    
    @staticmethod
    def handle_tools_event(node_output: Dict[str, Any]) -> tuple:
        """Handle tools node events (tool results).
        
        Returns:
            Tuple of (reasoning_dict, chart_json, map_json)
        """
        node_messages = node_output.get("messages", [])
        chart_data_json = None
        map_data_json = None
        
        for msg in node_messages:
            if not hasattr(msg, 'content') or not msg.content:
                continue
            
            try:
                # Parse result data
                if isinstance(msg.content, str):
                    result_data = json.loads(msg.content)
                else:
                    result_data = msg.content
                
                # Capture chart/map data
                if "chart_json" in result_data and result_data.get("success"):
                    chart_data_json = json.dumps(result_data, ensure_ascii=False, indent=2)
                
                if "map_html" in result_data and result_data.get("success"):
                    map_data_json = json.dumps(result_data, ensure_ascii=False, indent=2)
                
                # Build reasoning message
                reasoning = "✅ **Risultati Tool**\n\n"
                reasoning += StreamingHandler._format_tool_results(result_data)
                
                return {"type": "reasoning", "content": reasoning}, chart_data_json, map_data_json
                
            except (json.JSONDecodeError, AttributeError):
                reasoning = "✅ **Tool Eseguito**\n\nElaborazione risultati...\n"
                return {"type": "reasoning", "content": reasoning}, chart_data_json, map_data_json
        
        return None, chart_data_json, map_data_json
    
    @staticmethod
    def _format_tool_results(result_data: Dict[str, Any]) -> str:
        """Format tool results for display."""
        output = ""
        
        # SQL query
        if "sql_executed" in result_data:
            sql = result_data.get("sql_executed", "")
            output += f"**Query SQL generata:**\n```sql\n{sql}\n```\n\n"
        
        # Row count with vector search info
        if "row_count" in result_data:
            row_count = result_data.get("row_count", 0)
            
            if result_data.get("vector_search_applied", False):
                total_found = result_data.get("total_rows_found", row_count)
                output += "🔍 **Vector Search Applicata**\n"
                output += f"📊 **Righe totali trovate**: {total_found}\n"
                output += f"✨ **Top risultati più rilevanti**: {row_count}\n"
                if "info" in result_data:
                    output += f"ℹ️  {result_data['info']}\n"
            else:
                output += f"📊 **Righe trovate**: {row_count}\n"
                if "warning" in result_data:
                    output += f"⚠️  **Attenzione**: {result_data['warning']}\n"
            output += "\n"
        
        # Results preview
        if "results" in result_data:
            results = result_data.get("results", [])
            if results:
                output += "**Primi risultati:**\n\n<ol>\n"
                for row in results[:3]:
                    output += "<li>"
                    if "genus_species" in row:
                        output += f"Specie: {row['genus_species']} "
                    if "count" in row:
                        output += f"Count: {row['count']} "
                    if "district" in row:
                        output += f"Distretto: {row['district']} "
                    if "trunk_circumference" in row:
                        output += f"Circonferenza: {row['trunk_circumference']}cm "
                    output += "</li>\n"
                output += "</ol>\n"
                
                if len(results) > 3:
                    output += f"... e altri {len(results) - 3} risultati\n"
        
        # Single value results
        elif "result" in result_data and "column" in result_data:
            result_val = result_data.get("result")
            column_name = result_data.get("column")
            output += f"**{column_name}**: {result_val}\n"
        
        # CO2 results
        if "co2_sequestration_kg" in result_data:
            co2 = result_data.get("co2_sequestration_kg", 0)
            output += f"🌱 **CO2 sequestrato**: {co2} kg\n"
        
        return output
    
    @staticmethod
    def handle_budget_check_event(node_output: Dict[str, Any]) -> Dict[str, Any]:
        """Handle budget check node events."""
        if node_output.get("budget_exceeded"):
            node_messages = node_output.get("messages", [])
            final_response = None
            if node_messages:
                last_msg = node_messages[-1]
                if isinstance(last_msg, AIMessage) and last_msg.content:
                    final_response = last_msg.content
            
            return {
                "type": "reasoning",
                "content": "⚠️ **Budget Limit**\n\nLimite di esecuzione raggiunto. Interruzione per prevenire loop infiniti.\n",
                "final_response": final_response
            }
        else:
            status = node_output.get("budget_status", {})
            if status:
                reasoning = "✓ **Budget Check**\n\n"
                reasoning += f"Tool calls: {status.get('total_tool_calls', 'N/A')}\n"
                reasoning += f"Tempo: {status.get('elapsed_time', 'N/A')}\n"
                return {"type": "reasoning", "content": reasoning}
        return None
    
    @staticmethod
    def handle_tool_loop_guard_event(node_output: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool loop guard node events."""
        node_messages = node_output.get("messages", [])
        if node_messages:
            last_msg = node_messages[-1]
            if isinstance(last_msg, AIMessage) and last_msg.content:
                # Extract chart/map data from the response if present
                chart_json = None
                map_json = None
                content = last_msg.content
                
                # Extract chart data
                if "CHART_DATA_START" in content and "CHART_DATA_END" in content:
                    try:
                        start_idx = content.find("CHART_DATA_START") + len("CHART_DATA_START")
                        end_idx = content.find("CHART_DATA_END")
                        json_str = content[start_idx:end_idx].strip()
                        chart_json = json_str  # Keep as string for streaming handler
                    except Exception:
                        pass
                
                # Extract map data
                if "MAP_DATA_START" in content and "MAP_DATA_END" in content:
                    try:
                        start_idx = content.find("MAP_DATA_START") + len("MAP_DATA_START")
                        end_idx = content.find("MAP_DATA_END")
                        json_str = content[start_idx:end_idx].strip()
                        map_json = json_str  # Keep as string for streaming handler
                    except Exception:
                        pass
                
                result = {
                    "type": "reasoning",
                    "content": "🛑 **Stop Anti-Loop**\n\nRilevata ripetizione della stessa chiamata tool. Interrompo ed entro in modalità chiarimento.\n",
                    "final_response": last_msg.content
                }
                
                # Add chart/map data if found
                if chart_json:
                    result["chart_json"] = chart_json
                if map_json:
                    result["map_json"] = map_json
                
                return result
        else:
            if node_output.get("tool_loop_action") == "replan":
                return {
                    "type": "reasoning",
                    "content": "🔁 **Recovery Anti-Loop**\n\nRilevata ripetizione della stessa chiamata tool. Provo a cambiare strategia (replanning).\n"
                }
        return None
    
    @staticmethod
    def handle_validator_event(node_output: Dict[str, Any], retry_count: int, max_retries: int) -> tuple:
        """Handle validator node events.
        
        Returns:
            Tuple of (reasoning_dict, new_retry_count)
        """
        validation = node_output.get("validation_result", {})
        is_complete = validation.get("is_complete", True)
        
        if is_complete:
            reasoning = "✓ **Validazione Completata**\n\nLa risposta è completa e accurata.\n"
            return {"type": "reasoning", "content": reasoning}, retry_count
        else:
            new_retry_count = retry_count + 1
            if new_retry_count > max_retries:
                reasoning = "⚠️ **Validazione**\n\nRaggiunto limite retry. Proseguo con la risposta attuale.\n"
                return {"type": "reasoning", "content": reasoning}, new_retry_count
            else:
                missing = validation.get("missing_tasks", [])
                feedback = validation.get("feedback", "")
                reasoning = f"⚠️ **Validazione (Tentativo {new_retry_count})**\n\n"
                if missing:
                    reasoning += f"Task mancanti: {', '.join(missing)}\n"
                if feedback:
                    reasoning += f"\n{feedback}\n"
                reasoning += "\nRielaborazione risposta...\n"
                return {"type": "reasoning", "content": reasoning}, new_retry_count

