"""Tool loop detection and recovery mechanisms."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage

from streamlit_app.agent.extraction import DataExtractor
from streamlit_app.llm.tool_loop_guard import ToolLoopGuard


class ToolLoopManager:
    """Manager for detecting and recovering from tool call loops."""
    
    MAX_CALLS_PER_TOOL = 5  # If same tool called 5+ times, force replan
    
    def __init__(self):
        """Initialize tool loop manager."""
        self._extractor = DataExtractor()
        self._loop_guard = ToolLoopGuard(max_consecutive_repeats=2)
    
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
        
        # CRITICAL: If query_tree_dataset has been called 2+ times with valid results,
        # FORCE a response using those results instead of allowing more calls
        dataset_call_count = tool_call_counts.get("query_tree_dataset", 0)
        if dataset_call_count >= 2:
            # Check if we have valid results to use
            dataset_results = self._extractor.extract_dataset_results(messages)
            if dataset_results:
                # We have results! Force a response NOW
                response = self._format_dataset_results(dataset_results, messages)
                return {
                    "messages": [AIMessage(content=response)],
                    "tool_loop_detected": True,
                    "tool_loop_action": "stop",
                    "tool_loop_details": {"forced_response": True, "results_count": len(dataset_results)},
                    "tool_call_counts": tool_call_counts,
                }
        
        # CRITICAL: If generate_chart has been called 2+ times with successful result,
        # FORCE a response with the chart instead of allowing more calls
        chart_call_count = tool_call_counts.get("generate_chart", 0)
        if chart_call_count >= 2:
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
        
        # CRITICAL: If generate_map has been called 2+ times with successful result,
        # FORCE a response with the map instead of allowing more calls
        map_call_count = tool_call_counts.get("generate_map", 0)
        if map_call_count >= 2:
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
                response = self._format_dataset_results(dataset_results, messages)
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
        """Create a self-reflection prompt to recover from tool loops.
        
        Args:
            state: Current agent state
            messages: Current conversation messages
            
        Returns:
            System message with replan prompt
        """
        current = int(state.get("tool_loop_replan_count") or 0)
        details: Dict[str, Any] = state.get("tool_loop_details") or {}
        tool_calls = details.get("tool_calls") or []
        abuse_detected = details.get("abuse_detected", False)
        call_count = details.get("call_count", 0)

        # Extract tool results from recent messages for self-evaluation
        recent_tool_results = []
        for msg in reversed(list(messages)[-15:]):
            if isinstance(msg, ToolMessage):
                content = str(msg.content)[:400]
                recent_tool_results.append(content)
        
        tool_results_summary = "\n---\n".join(recent_tool_results[:3]) if recent_tool_results else "Nessun risultato recente"
        
        # Get the abused tool name
        abused_tool = tool_calls[0].get("name") if tool_calls else "questo tool"

        # If tool abuse detected (same tool called many times with different args)
        if abuse_detected or call_count >= 5:
            prompt = f"""🛑 **STOP - HAI CHIAMATO `{abused_tool}` {call_count} VOLTE**

Stai chiamando lo stesso tool ripetutamente con query diverse, ma non stai facendo progressi.

**RISULTATI CHE HAI GIÀ OTTENUTO:**
{tool_results_summary}

**ANALIZZA LA SITUAZIONE:**
- Hai già cercato {call_count} volte - se non hai trovato quello che cerchi, probabilmente non c'è
- Guarda i risultati sopra: contengono informazioni utili?
- Puoi rispondere con quello che hai, anche se parziale?

**SCEGLI UNA DI QUESTE AZIONI (OBBLIGATORIO):**

1. **RISPONDI CON QUELLO CHE HAI**: Usa i paper/risultati che hai trovato per dare una risposta.
2. **AMMETTI I LIMITI E OFFRI ALTERNATIVE**: Se non hai trovato esattamente quello che l'utente cerca.
3. **CHIEDI CHIARIMENTI**: Se hai bisogno di più contesto.

**⛔ NON PUOI chiamare di nuovo `{abused_tool}`. Devi rispondere ora.**
"""
        # Progressive assertiveness for exact fingerprint repeats
        elif current < 2:
            prompt = f"""🔄 **MOMENTO DI AUTO-RIFLESSIONE**

Hai chiamato lo stesso tool più volte. Prima di procedere, chiediti:

**Risultati ottenuti finora:**
{tool_results_summary}

**Domande da porti:**
1. Questi risultati rispondono (anche parzialmente) alla domanda dell'utente?
2. Sto cercando qualcosa che potrebbe non esistere nei dati disponibili?
3. Posso dare una risposta utile con quello che ho?

**Azioni possibili:**
A) **RISPONDO**: Formula una risposta con ciò che hai trovato (anche se parziale)
B) **CHIEDO**: Fai una domanda specifica all'utente per capire meglio
C) **CAMBIO STRATEGIA**: Usa un tool diverso

NON richiamare lo stesso tool con la stessa query.
"""
        else:
            prompt = f"""🛑 **STOP - RISPOSTA OBBLIGATORIA**

Hai tentato {current + 1} volte senza successo. È il momento di rispondere all'utente.

**Risultati disponibili:**
{tool_results_summary}

**ISTRUZIONI FINALI:**
Scrivi ORA una risposta all'utente che:
1. Spiega onestamente cosa hai cercato e cosa hai (o non hai) trovato
2. Offre alternative concrete: "Non ho trovato X, ma posso aiutarti con Y..."
3. Chiede se l'utente vuole procedere diversamente

**RISPONDI ORA - Non chiamare altri tool.**
"""

        return SystemMessage(content=prompt)
    
    def _format_dataset_results(self, results: List[dict], messages: Sequence[BaseMessage]) -> str:
        """Format dataset results as user-friendly response."""
        from streamlit_app.agent.response_builder import ResponseBuilder
        return ResponseBuilder.format_dataset_results(results, messages)
    
    def _format_chart_results(self, chart_results: List[dict], messages: Sequence[BaseMessage]) -> str:
        """Format chart results as user-friendly response."""
        if not chart_results:
            return "Non sono riuscito a generare il grafico richiesto."
        
        # Get the most recent successful chart
        chart = chart_results[0]
        
        # Build response with chart markers for UI parsing
        chart_type = chart.get("chart_type", "grafico")
        data_points = chart.get("data_points", 0)
        title = chart.get("title", "Grafico")
        description = chart.get("description", f"Grafico {chart_type} generato con successo")
        
        response = f"Ecco il {chart_type} che hai richiesto: **{title}**\n\n"
        response += f"{description} con {data_points} punti dati.\n\n"
        
        # Add chart data markers for UI
        chart_json_str = json.dumps(chart, ensure_ascii=False, indent=2)
        response += f"\nCHART_DATA_START\n{chart_json_str}\nCHART_DATA_END\n"
        
        return response
    
    def _format_map_results(self, map_results: List[dict], messages: Sequence[BaseMessage]) -> str:
        """Format map results as user-friendly response."""
        if not map_results:
            return "Non sono riuscito a generare la mappa richiesta."
        
        # Get the most recent successful map
        map_data = map_results[0]
        
        # Build response with map markers for UI parsing
        map_type = map_data.get("map_type", "mappa")
        data_points = map_data.get("data_points", 0)
        title = map_data.get("title", "Mappa")
        description = map_data.get("description", f"Mappa {map_type} generata con successo")
        
        response = f"Ecco la {map_type} che hai richiesto: **{title}**\n\n"
        response += f"{description} con {data_points} punti visualizzati.\n\n"
        
        # Add map data markers for UI
        map_json_str = json.dumps(map_data, ensure_ascii=False, indent=2)
        response += f"\nMAP_DATA_START\n{map_json_str}\nMAP_DATA_END\n"
        
        return response

