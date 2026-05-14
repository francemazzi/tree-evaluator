"""Tool loop detection and recovery."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage

from streamlit_app.agent.extraction import DataExtractor
from streamlit_app.agent.tool_loop_formatters import (
    format_chart_results,
    format_co2_aggregate_results,
    format_dataset_results,
    format_map_results,
)
from streamlit_app.agent.tool_loop_replan import create_replan_prompt
from streamlit_app.llm.tool_loop_guard import ToolLoopGuard


class SemanticToolLoopDetector:
    """Detects tool loops using semantic argument similarity."""

    SIMILARITY_THRESHOLD = 0.85  # Above this = considered same call

    def __init__(self):
        self.call_history: List[Dict[str, Any]] = []

    def record_call(self, tool_name: str, args: Dict[str, Any]) -> None:
        """Record a tool call for similarity tracking."""
        self.call_history.append({
            "tool_name": tool_name,
            "args": args,
        })

    def is_semantic_loop(self, tool_name: str, args: Dict[str, Any]) -> bool:
        """Check if this call is semantically similar to recent calls.

        Args:
            tool_name: Name of the tool being called.
            args: Arguments for the tool call.

        Returns:
            True if this appears to be a repetitive loop call.
        """
        # Get recent calls to the same tool
        recent_same_tool = [
            c for c in self.call_history[-5:]
            if c["tool_name"] == tool_name
        ]

        if len(recent_same_tool) < 2:
            return False

        # Check similarity with last 3 calls
        for prev_call in recent_same_tool[-3:]:
            similarity = self._compute_arg_similarity(prev_call["args"], args)
            if similarity >= self.SIMILARITY_THRESHOLD:
                return True

        return False

    def _compute_arg_similarity(self, args1: Dict[str, Any], args2: Dict[str, Any]) -> float:
        """Compute similarity between two argument dictionaries.

        Args:
            args1: First argument dict.
            args2: Second argument dict.

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        if args1 == args2:
            return 1.0

        if not args1 or not args2:
            return 0.0

        # Compare common keys
        all_keys = set(args1.keys()) | set(args2.keys())
        if not all_keys:
            return 0.0

        matches = 0
        for key in all_keys:
            val1 = args1.get(key)
            val2 = args2.get(key)

            if val1 == val2:
                matches += 1
            elif isinstance(val1, str) and isinstance(val2, str):
                # Partial string match for query-like arguments
                if self._string_similarity(val1, val2) > 0.8:
                    matches += 0.5

        return matches / len(all_keys)

    def _string_similarity(self, s1: str, s2: str) -> float:
        """Compute simple string similarity using word overlap."""
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0

    def get_loop_info(self, tool_name: str) -> Dict[str, Any]:
        """Get information about potential loops for a tool."""
        tool_calls = [c for c in self.call_history if c["tool_name"] == tool_name]
        return {
            "tool_name": tool_name,
            "total_calls": len(tool_calls),
            "recent_calls": len([c for c in self.call_history[-10:] if c["tool_name"] == tool_name]),
        }


class ToolLoopManager:
    """Manager for detecting and recovering from tool call loops."""

    MAX_CALLS_PER_TOOL = 5  # If same tool called 5+ times, force replan

    def __init__(self):
        """Initialize tool loop manager."""
        self._extractor = DataExtractor()
        self._loop_guard = ToolLoopGuard(max_consecutive_repeats=2)
        self._semantic_detector = SemanticToolLoopDetector()
    
    def check_for_loops(self, messages: Sequence[BaseMessage], state: dict) -> dict:
        """Check for tool call loops and decide on action.
        
        Args:
            messages: Current conversation messages
            state: Current agent state
            
        Returns:
            Dict with loop detection results and suggested action
        """
        # Count ALL ToolMessages in the conversation by tool name
        tool_call_counts: Dict[str, int] = {}
        for msg in messages:
            if isinstance(msg, ToolMessage):
                tool_name = getattr(msg, "name", None)
                if tool_name:
                    tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1

        # Extract pending tool calls from last AI message for semantic analysis
        pending_tool_calls = []
        for msg in reversed(list(messages)):
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                pending_tool_calls = msg.tool_calls
                break

        # Check for semantic loops using argument similarity
        for tc in pending_tool_calls:
            tool_name = tc.get("name", "unknown")
            args = tc.get("args", {})

            # Record call for future similarity checks
            self._semantic_detector.record_call(tool_name, args)

            # Check if this is a semantic loop (same tool with similar args)
            if self._semantic_detector.is_semantic_loop(tool_name, args):
                loop_info = self._semantic_detector.get_loop_info(tool_name)
                detected_language = state.get("detected_language", "it")
                if detected_language not in ["it", "en"]:
                    detected_language = "it"

                # Try to get any existing results before forcing stop
                dataset_results = self._extractor.extract_dataset_results(messages)
                if dataset_results:
                    response = self._format_dataset_results(dataset_results, messages, detected_language)
                else:
                    if detected_language == "en":
                        response = (
                            f"I've already called `{tool_name}` multiple times with similar parameters. "
                            "Let me summarize what I found so far and provide you with a response."
                        )
                    else:
                        response = (
                            f"Ho già chiamato `{tool_name}` più volte con parametri simili. "
                            "Lasciatemi riassumere quello che ho trovato finora e fornirvi una risposta."
                        )

                return {
                    "messages": [AIMessage(content=response)],
                    "tool_loop_detected": True,
                    "tool_loop_action": "stop",
                    "tool_loop_details": {
                        "semantic_loop": True,
                        "tool_name": tool_name,
                        **loop_info,
                    },
                    "tool_call_counts": tool_call_counts,
                }

        # CRITICAL: If calculate_co2_aggregate has been called 4+ times with valid results,
        # FORCE a response using those results instead of allowing more calls
        # NOTE: Threshold increased from 2 to 4 to allow legitimate comparative queries
        # (e.g., "CO2 for tree A" then "CO2 for tree B")
        co2_call_count = tool_call_counts.get("calculate_co2_aggregate", 0)
        if co2_call_count >= 4:
            # Check if we have valid results to use
            co2_results = self._extractor.extract_co2_aggregate_results(messages)
            if co2_results:
                # Get detected language from state
                detected_language = state.get("detected_language", "it")
                if detected_language not in ["it", "en"]:
                    detected_language = "it"
                # We have results! Force a response NOW using answer_hint if available
                response = self._format_co2_aggregate_results(co2_results[0], detected_language)
                return {
                    "messages": [AIMessage(content=response)],
                    "tool_loop_detected": True,
                    "tool_loop_action": "stop",
                    "tool_loop_details": {"forced_response": True, "co2_results": True},
                    "tool_call_counts": tool_call_counts,
                }
        
        # CRITICAL: If query_tree_dataset has been called 4+ times with valid results,
        # FORCE a response using those results instead of allowing more calls
        # NOTE: Threshold increased from 2 to 4 to allow legitimate follow-up queries
        dataset_call_count = tool_call_counts.get("query_tree_dataset", 0)
        if dataset_call_count >= 4:
            # Check if we have valid results to use
            dataset_results = self._extractor.extract_dataset_results(messages)
            if dataset_results:
                # Get detected language from state
                detected_language = state.get("detected_language", "it")
                if detected_language not in ["it", "en"]:
                    detected_language = "it"
                # We have results! Force a response NOW
                response = self._format_dataset_results(dataset_results, messages, detected_language)
                return {
                    "messages": [AIMessage(content=response)],
                    "tool_loop_detected": True,
                    "tool_loop_action": "stop",
                    "tool_loop_details": {"forced_response": True, "results_count": len(dataset_results)},
                    "tool_call_counts": tool_call_counts,
                }
        
        # CRITICAL: If generate_chart has been called 3+ times with successful result,
        # FORCE a response with the chart instead of allowing more calls
        # NOTE: Threshold increased from 2 to 3 to allow chart refinements
        chart_call_count = tool_call_counts.get("generate_chart", 0)
        if chart_call_count >= 3:
            # Check if we have a successful chart
            chart_results = self._extractor.extract_chart_results(messages)
            if chart_results:
                # We have a chart! Force a response NOW
                response = self._format_chart_results(chart_results, messages)
                return {
                    "messages": [AIMessage(content=response)],
                    "tool_loop_detected": True,
                    "tool_loop_action": "stop",
                    "tool_loop_details": {"forced_response": True, "chart_count": len(chart_results)},
                    "tool_call_counts": tool_call_counts,
                }
        
        # CRITICAL: If generate_map has been called 3+ times with successful result,
        # FORCE a response with the map instead of allowing more calls
        # NOTE: Threshold increased from 2 to 3 to allow map refinements
        map_call_count = tool_call_counts.get("generate_map", 0)
        if map_call_count >= 3:
            # Check if we have a successful map
            map_results = self._extractor.extract_map_results(messages)
            if map_results:
                # We have a map! Force a response NOW
                response = self._format_map_results(map_results, messages)
                return {
                    "messages": [AIMessage(content=response)],
                    "tool_loop_detected": True,
                    "tool_loop_action": "stop",
                    "tool_loop_details": {"forced_response": True, "map_count": len(map_results)},
                    "tool_call_counts": tool_call_counts,
                }
        
        # Check if any single tool has been called too many times (even with different args)
        abused_tool = None
        for tool_name, count in tool_call_counts.items():
            if count >= self.MAX_CALLS_PER_TOOL:
                abused_tool = tool_name
                break
        
        if abused_tool:
            call_count = tool_call_counts[abused_tool]
            replan_count = int(state.get("tool_loop_replan_count") or 0)
            
            details = {
                "tool_calls": [{"name": abused_tool}],
                "fingerprint": f"{abused_tool}:repeated_{call_count}_times",
                "abuse_detected": True,
                "call_count": call_count,
            }
            
            # If tool called 10+ times OR we've already tried 3+ replans, FORCE STOP
            if call_count >= 10 or replan_count >= 3:
                from streamlit_app.agent.response_builder import ResponseBuilder
                fallback_response = ResponseBuilder.build_dynamic_fallback_response(
                    abused_tool=abused_tool,
                    call_count=call_count,
                    messages=messages,
                    extractor=self._extractor,
                )
                return {
                    "messages": [AIMessage(content=fallback_response)],
                    "tool_loop_detected": True,
                    "tool_loop_action": "stop",
                    "tool_loop_details": details,
                    "tool_call_counts": tool_call_counts,
                }
            
            # Otherwise, force replan
            return {
                "tool_loop_detected": True,
                "tool_loop_action": "replan",
                "tool_loop_details": details,
                "tool_call_counts": tool_call_counts,
            }
        
        # Also check for exact fingerprint repeats (same tool + same args)
        last_fp = state.get("tool_last_fingerprint")
        repeat = int(state.get("tool_repeat_count") or 0)

        decision, new_fp, new_repeat = self._loop_guard.evaluate(
            messages=messages, 
            last_fingerprint=last_fp, 
            repeat_count=repeat
        )
        
        if decision.action == "stop" and decision.user_message:
            return {
                "messages": [AIMessage(content=decision.user_message)],
                "tool_last_fingerprint": new_fp,
                "tool_repeat_count": new_repeat,
                "tool_loop_detected": True,
                "tool_loop_action": "stop",
                "tool_loop_details": decision.details,
                "tool_call_counts": tool_call_counts,
            }
        if decision.action == "replan":
            # Before allowing replan, check if we already have valid results
            dataset_results = self._extractor.extract_dataset_results(messages)
            if dataset_results and len(dataset_results) > 0:
                # Get detected language from state
                detected_language = state.get("detected_language", "it")
                if detected_language not in ["it", "en"]:
                    detected_language = "it"
                response = self._format_dataset_results(dataset_results, messages, detected_language)
                return {
                    "messages": [AIMessage(content=response)],
                    "tool_last_fingerprint": new_fp,
                    "tool_repeat_count": new_repeat,
                    "tool_loop_detected": True,
                    "tool_loop_action": "stop",
                    "tool_loop_details": {"forced_response": True, "results_count": len(dataset_results)},
                    "tool_call_counts": tool_call_counts,
                }
            
            return {
                "tool_last_fingerprint": new_fp,
                "tool_repeat_count": new_repeat,
                "tool_loop_detected": True,
                "tool_loop_action": "replan",
                "tool_loop_details": decision.details,
                "tool_call_counts": tool_call_counts,
            }
        
        return {
            "tool_last_fingerprint": new_fp,
            "tool_repeat_count": new_repeat,
            "tool_loop_detected": False,
            "tool_loop_action": "continue",
            "tool_loop_details": None,
            "tool_call_counts": tool_call_counts,
        }
    
    def create_replan_prompt(self, state: dict, messages: Sequence[BaseMessage]) -> SystemMessage:
        """Create a self-reflection prompt to recover from tool loops."""
        return create_replan_prompt(state, messages)

    def _format_dataset_results(self, results: List[dict], messages: Sequence[BaseMessage], language: str = "it") -> str:
        """Format dataset results as user-friendly response."""
        return format_dataset_results(results, messages, language)

    def _format_chart_results(self, chart_results: List[dict], messages: Sequence[BaseMessage]) -> str:
        """Format chart results as user-friendly response."""
        return format_chart_results(chart_results, messages)

    def _format_map_results(self, map_results: List[dict], messages: Sequence[BaseMessage]) -> str:
        """Format map results as user-friendly response."""
        return format_map_results(map_results, messages)

    def _format_co2_aggregate_results(self, result: dict, language: str = "it") -> str:
        """Format CO2 aggregate results as user-friendly response."""
        return format_co2_aggregate_results(result, language)
