"""Streaming handler for LangGraph agent responses."""

from __future__ import annotations

import json
from typing import Any, Dict, Literal

from langchain_core.messages import AIMessage

from streamlit_app.agent.translations import Language, format_translation, get_translation


class StreamingHandler:
    """Handler for formatting streaming responses from the agent."""
    
    @staticmethod
    def handle_language_detector_event(node_output: Dict[str, Any], language: Language = "it") -> Dict[str, str]:
        """Handle language detector node events.
        
        Args:
            node_output: Output from language_detector node
            language: Language for messages ("it" or "en")
            
        Returns:
            Dict with type and content for streaming, or None if no display needed
        """
        detected_language = node_output.get("detected_language", "it")
        
        language_name = get_translation("language_detected", detected_language)
        
        reasoning = "🌐 **Lingua / Language**\n\n"
        reasoning += f"{language_name}\n"
        
        return {"type": "reasoning", "content": reasoning}
    
    @staticmethod
    def handle_context_manager_event(node_output: Dict[str, Any], original_count: int, language: Language = "it") -> Dict[str, str]:
        """Handle context manager node events.
        
        Args:
            node_output: Output from context_manager node
            original_count: Original message count
            language: Language for messages ("it" or "en")
            
        Returns:
            Dict with type and content for streaming
        """
        message_count = node_output.get("message_count", 0)
        
        if message_count < original_count:
            reasoning = f"{get_translation('context_management', language)}\n\n"
            reasoning += f"{get_translation('original_messages', language)}: {original_count}\n"
            reasoning += f"{get_translation('optimized_messages', language)}: {message_count}\n"
            reasoning += f"{get_translation('context_compressed', language)}\n"
            return {"type": "reasoning", "content": reasoning}
        return None
    
    @staticmethod
    def handle_query_optimizer_event(node_output: Dict[str, Any], language: Language = "it") -> Dict[str, str]:
        """Handle query optimizer node events.
        
        Args:
            node_output: Output from query_optimizer node
            language: Language for messages ("it" or "en")
        """
        optimized = node_output.get("optimized_query", "")
        tasks = node_output.get("tasks", [])
        
        if optimized:
            reasoning = f"{get_translation('query_optimization', language)}\n\n"
            reasoning += f"{get_translation('optimized_query_label', language)}: *{optimized}*\n\n"
            if tasks:
                reasoning += f"{get_translation('tasks_identified', language)}\n\n<ol>\n"
                for task in tasks:
                    reasoning += f"<li>{task}</li>\n"
                reasoning += "</ol>\n\n"
            return {"type": "reasoning", "content": reasoning}
        return None
    
    @staticmethod
    def handle_agent_event(node_output: Dict[str, Any], language: Language = "it") -> Dict[str, Any]:
        """Handle agent node events (tool calls or responses).
        
        Args:
            node_output: Output from agent node
            language: Language for messages ("it" or "en")
        """
        node_messages = node_output.get("messages", [])
        if not node_messages:
            return None
        
        last_msg = node_messages[-1]
        if not isinstance(last_msg, AIMessage):
            return None
        
        if last_msg.tool_calls:
            # Agent is calling tools
            reasoning = f"{get_translation('tool_call', language)}\n\n"
            for tool_call in last_msg.tool_calls:
                tool_name = tool_call.get("name", "unknown")
                tool_args = tool_call.get("args", {})
                
                reasoning += f"- **{get_translation('tool', language)}**: `{tool_name}`\n"
                reasoning += StreamingHandler._format_tool_args(tool_name, tool_args, language)
                reasoning += "\n"
            
            return {"type": "reasoning", "content": reasoning}
        elif last_msg.content and not last_msg.tool_calls:
            # Final or intermediate response
            return {"type": "final_response", "content": last_msg.content}
        
        return None
    
    @staticmethod
    def _format_tool_args(tool_name: str, tool_args: Dict[str, Any], language: Language = "it") -> str:
        """Format tool arguments for display.
        
        Args:
            tool_name: Name of the tool
            tool_args: Tool arguments
            language: Language for messages ("it" or "en")
        """
        if tool_name == "query_tree_dataset":
            natural_q = tool_args.get("natural_query", "N/A")
            return f"  - **{get_translation('query', language)}**: _{natural_q}_\n"
        elif tool_name in ("calculate_co2", "estimate_environment"):
            dbh = tool_args.get("dbh_cm", "N/A")
            height = tool_args.get("height_m", "N/A")
            result = f"  - **{get_translation('dbh', language)}**: {dbh} cm\n  - **{get_translation('height', language)}**: {height} m\n"
            if "wood_density" in tool_args:
                result += f"  - **{get_translation('wood_density', language)}**: {tool_args['wood_density']} g/cm³\n"
            return result
        elif tool_name == "calculate_co2_aggregate":
            query = tool_args.get("natural_query", "N/A")
            return f"  - **{get_translation('query', language)}**: _{query}_\n"
        elif tool_name == "generate_chart":
            chart_type = tool_args.get("chart_type", "N/A")
            return f"  - **{get_translation('chart_type', language)}**: {chart_type}\n"
        return ""
    
    @staticmethod
    def handle_tools_event(node_output: Dict[str, Any], language: Language = "it") -> tuple:
        """Handle tools node events (tool results).
        
        Args:
            node_output: Output from tools node
            language: Language for messages ("it" or "en")
        
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
                reasoning = f"{get_translation('tool_results', language)}\n\n"
                reasoning += StreamingHandler._format_tool_results(result_data, language)
                
                return {"type": "reasoning", "content": reasoning}, chart_data_json, map_data_json
                
            except (json.JSONDecodeError, AttributeError):
                reasoning = f"{get_translation('tool_executed', language)}\n\n{get_translation('processing_results', language)}\n"
                return {"type": "reasoning", "content": reasoning}, chart_data_json, map_data_json
        
        return None, chart_data_json, map_data_json
    
    @staticmethod
    def _format_tool_results(result_data: Dict[str, Any], language: Language = "it") -> str:
        """Format tool results for display.
        
        Args:
            result_data: Tool result data
            language: Language for messages ("it" or "en")
        """
        output = ""
        
        # SQL query
        if "sql_executed" in result_data:
            sql = result_data.get("sql_executed", "")
            output += f"{get_translation('sql_query_generated', language)}\n```sql\n{sql}\n```\n\n"
        
        # Row count with vector search info
        if "row_count" in result_data:
            row_count = result_data.get("row_count", 0)
            
            if result_data.get("vector_search_applied", False):
                total_found = result_data.get("total_rows_found", row_count)
                output += f"{get_translation('vector_search_applied', language)}\n"
                output += f"📊 {get_translation('total_rows_found', language)}: {total_found}\n"
                output += f"✨ {get_translation('top_relevant_results', language)}: {row_count}\n"
                if "info" in result_data:
                    output += f"ℹ️  {result_data['info']}\n"
            else:
                output += f"📊 {get_translation('rows_found', language)}: {row_count}\n"
                if "warning" in result_data:
                    output += f"⚠️  {get_translation('warning', language)}: {result_data['warning']}\n"
            output += "\n"
        
        # Results preview
        if "results" in result_data:
            results = result_data.get("results", [])
            if results:
                output += f"{get_translation('first_results', language)}\n\n<ol>\n"
                for row in results[:3]:
                    output += "<li>"
                    if "genus_species" in row:
                        output += f"{get_translation('species', language)}: {row['genus_species']} "
                    if "count" in row:
                        output += f"{get_translation('count', language)}: {row['count']} "
                    if "district" in row:
                        output += f"{get_translation('district', language)}: {row['district']} "
                    if "trunk_circumference" in row:
                        output += f"{get_translation('circumference', language)}: {row['trunk_circumference']}cm "
                    output += "</li>\n"
                output += "</ol>\n"
                
                if len(results) > 3:
                    output += format_translation('and_others', language, count=len(results) - 3) + "\n"
        
        # Single value results
        elif "result" in result_data and "column" in result_data:
            result_val = result_data.get("result")
            column_name = result_data.get("column")
            output += f"**{column_name}**: {result_val}\n"
        
        # CO2 results
        if "co2_sequestration_kg" in result_data:
            co2 = result_data.get("co2_sequestration_kg", 0)
            output += f"🌱 {get_translation('co2_sequestered', language)}: {co2} kg\n"
        
        return output
    
    @staticmethod
    def handle_budget_check_event(node_output: Dict[str, Any], language: Language = "it") -> Dict[str, Any]:
        """Handle budget check node events.
        
        Args:
            node_output: Output from budget_check node
            language: Language for messages ("it" or "en")
        """
        if node_output.get("budget_exceeded"):
            node_messages = node_output.get("messages", [])
            final_response = None
            if node_messages:
                last_msg = node_messages[-1]
                if isinstance(last_msg, AIMessage) and last_msg.content:
                    final_response = last_msg.content
            
            return {
                "type": "reasoning",
                "content": f"{get_translation('budget_limit', language)}\n\n{get_translation('execution_limit_reached', language)}\n",
                "final_response": final_response
            }
        else:
            status = node_output.get("budget_status", {})
            if status:
                reasoning = f"{get_translation('budget_check', language)}\n\n"
                reasoning += f"{get_translation('tool_calls', language)}: {status.get('total_tool_calls', 'N/A')}\n"
                reasoning += f"{get_translation('time', language)}: {status.get('elapsed_time', 'N/A')}\n"
                return {"type": "reasoning", "content": reasoning}
        return None
    
    @staticmethod
    def handle_tool_loop_guard_event(node_output: Dict[str, Any], language: Language = "it") -> Dict[str, Any]:
        """Handle tool loop guard node events.
        
        Args:
            node_output: Output from tool_loop_guard node
            language: Language for messages ("it" or "en")
        """
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
                    "content": f"{get_translation('stop_anti_loop', language)}\n\n{get_translation('repetition_detected', language)}\n",
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
                    "content": f"{get_translation('recovery_anti_loop', language)}\n\n{get_translation('repetition_detected_replan', language)}\n"
                }
        return None
    
    @staticmethod
    def handle_validator_event(node_output: Dict[str, Any], retry_count: int, max_retries: int, language: Language = "it") -> tuple:
        """Handle validator node events.
        
        Args:
            node_output: Output from validator node
            retry_count: Current retry count
            max_retries: Maximum retries allowed
            language: Language for messages ("it" or "en")
        
        Returns:
            Tuple of (reasoning_dict, new_retry_count)
        """
        validation = node_output.get("validation_result", {})
        is_complete = validation.get("is_complete", True)
        
        if is_complete:
            reasoning = f"{get_translation('validation_completed', language)}\n\n{get_translation('response_complete_accurate', language)}\n"
            return {"type": "reasoning", "content": reasoning}, retry_count
        else:
            new_retry_count = retry_count + 1
            if new_retry_count > max_retries:
                reasoning = f"{get_translation('validation', language)}\n\n{get_translation('retry_limit_reached', language)}\n"
                return {"type": "reasoning", "content": reasoning}, new_retry_count
            else:
                missing = validation.get("missing_tasks", [])
                feedback = validation.get("feedback", "")
                reasoning = format_translation('validation_attempt', language, count=new_retry_count) + "\n\n"
                if missing:
                    reasoning += f"{get_translation('missing_tasks', language)}: {', '.join(missing)}\n"
                if feedback:
                    reasoning += f"\n{feedback}\n"
                reasoning += f"\n{get_translation('reprocessing_response', language)}\n"
                return {"type": "reasoning", "content": reasoning}, new_retry_count

