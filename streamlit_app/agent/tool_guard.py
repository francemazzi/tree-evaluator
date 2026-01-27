"""Tool loop detection and recovery mechanisms."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage

from streamlit_app.agent.extraction import DataExtractor
from streamlit_app.llm.tool_loop_guard import ToolLoopGuard


class SemanticToolLoopDetector:
    """Detects tool loops using semantic similarity instead of just counting.

    This detector compares tool arguments to determine if calls are truly
    repetitive (high similarity = loop) vs legitimate follow-up calls
    (low similarity = different queries).
    """

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
        
        # Get detected language from state
        detected_language = state.get("detected_language", "it")
        if detected_language not in ["it", "en"]:
            detected_language = "it"

        # Extract tool results from recent messages for self-evaluation
        recent_tool_results = []
        for msg in reversed(list(messages)[-15:]):
            if isinstance(msg, ToolMessage):
                content = str(msg.content)[:400]
                recent_tool_results.append(content)
        
        if detected_language == "en":
            tool_results_summary = "\n---\n".join(recent_tool_results[:3]) if recent_tool_results else "No recent results"
            abused_tool = tool_calls[0].get("name") if tool_calls else "this tool"
        else:
            tool_results_summary = "\n---\n".join(recent_tool_results[:3]) if recent_tool_results else "Nessun risultato recente"
            abused_tool = tool_calls[0].get("name") if tool_calls else "questo tool"

        # If tool abuse detected (same tool called many times with different args)
        if abuse_detected or call_count >= 5:
            if detected_language == "en":
                prompt = f"""🛑 **STOP - YOU HAVE CALLED `{abused_tool}` {call_count} TIMES**

You are calling the same tool repeatedly with different queries, but you are not making progress.

**RESULTS YOU HAVE ALREADY OBTAINED:**
{tool_results_summary}

**ANALYZE THE SITUATION:**
- You have already searched {call_count} times - if you haven't found what you're looking for, it probably doesn't exist
- Look at the results above: do they contain useful information?
- Can you respond with what you have, even if partial?

**CHOOSE ONE OF THESE ACTIONS (MANDATORY):**

1. **RESPOND WITH WHAT YOU HAVE**: Use the papers/results you found to give an answer.
2. **ADMIT LIMITATIONS AND OFFER ALTERNATIVES**: If you haven't found exactly what the user is looking for.
3. **ASK FOR CLARIFICATION**: If you need more context.

**⛔ YOU CANNOT call `{abused_tool}` again. You must respond now.**
"""
            else:
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
            if detected_language == "en":
                prompt = f"""🔄 **MOMENT OF SELF-REFLECTION**

You have called the same tool multiple times. Before proceeding, ask yourself:

**Results obtained so far:**
{tool_results_summary}

**Questions to ask yourself:**
1. Do these results answer (even partially) the user's question?
2. Am I looking for something that might not exist in the available data?
3. Can I give a useful answer with what I have?

**Possible actions:**
A) **RESPOND**: Formulate a response with what you found (even if partial)
B) **ASK**: Ask the user a specific question to understand better
C) **CHANGE STRATEGY**: Use a different tool

DO NOT call the same tool with the same query.
"""
            else:
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
            if detected_language == "en":
                prompt = f"""🛑 **STOP - MANDATORY RESPONSE**

You have tried {current + 1} times without success. It's time to respond to the user.

**Available results:**
{tool_results_summary}

**FINAL INSTRUCTIONS:**
Write NOW a response to the user that:
1. Honestly explains what you searched for and what you found (or didn't find)
2. Offers concrete alternatives: "I didn't find X, but I can help you with Y..."
3. Asks if the user wants to proceed differently

**RESPOND NOW - Do not call other tools.**
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
    
    def _format_dataset_results(self, results: List[dict], messages: Sequence[BaseMessage], language: str = "it") -> str:
        """Format dataset results as user-friendly response."""
        from streamlit_app.agent.response_builder import ResponseBuilder
        return ResponseBuilder.format_dataset_results(results, messages, language)
    
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
    
    def _format_co2_aggregate_results(self, result: dict, language: str = "it") -> str:
        """Format CO2 aggregate results as user-friendly response with CARBON value prominently.
        
        Args:
            result: CO2 aggregate result dictionary
            language: Response language
            
        Returns:
            Formatted response string
        """
        # If answer_hint is present, use it directly
        if "answer_hint" in result:
            return result["answer_hint"]
        
        # Otherwise build response manually
        carbon_stock = result.get("carbon_stock_t", 0)
        co2_stock = result.get("co2_stock_t", 0)
        tree_count = result.get("tree_count", 0)
        total_biomass = result.get("total_biomass_t", 0)
        agb = result.get("above_ground_biomass_t", 0)
        bgb = result.get("below_ground_biomass_t", 0)
        species = result.get("dominant_species", "")
        
        # Get parameters
        params = result.get("parameters", {})
        cf = params.get("carbon_fraction", {}).get("value", 0.47)
        rs = params.get("root_shoot_ratio", {}).get("value", 0.24)
        
        if language == "en":
            response = f"""The carbon stock of {species} is **{carbon_stock:,.2f} t C** (tonnes of carbon).

Details:
- Trees analyzed: {tree_count:,}
- Carbon stock: {carbon_stock:,.2f} t C
- CO2 equivalent: {co2_stock:,.2f} t CO2
- Total biomass: {total_biomass:,.2f} t
  - Above-ground biomass (AGB): {agb:,.2f} t
  - Below-ground biomass (BGB): {bgb:,.2f} t

**Formulas used:**
- AGB = 0.0673 × (WD × DBH² × H)^0.976 (Chave et al., 2014)
- BGB = AGB × R/S
- C = Biomass × CF
- CO2 = C × (44/12)

**Parameters:**
- Wood density (WD): 0.6 g/cm³
- Carbon fraction (CF): {cf} ({cf*100:.1f}%)
- Root-to-shoot ratio (R/S): {rs}

Tools used: calculate_co2_aggregate"""
        else:
            response = f"""Lo stock di carbonio di {species} è di **{carbon_stock:,.2f} t C** (tonnellate di carbonio).

Dettagli:
- Alberi analizzati: {tree_count:,}
- Stock di carbonio: {carbon_stock:,.2f} t C
- CO2 equivalente: {co2_stock:,.2f} t CO2
- Biomassa totale: {total_biomass:,.2f} t
  - Biomassa epigea (AGB): {agb:,.2f} t
  - Biomassa ipogea (BGB): {bgb:,.2f} t

**Formule utilizzate:**
- AGB = 0.0673 × (WD × DBH² × H)^0.976 (Chave et al., 2014)
- BGB = AGB × R/S
- C = Biomassa × CF
- CO2 = C × (44/12)

**Parametri:**
- Densità legno (WD): 0.6 g/cm³
- Frazione carbonio (CF): {cf} ({cf*100:.1f}%)
- Rapporto R/S: {rs}

Tool utilizzati: calculate_co2_aggregate"""
        
        return response

