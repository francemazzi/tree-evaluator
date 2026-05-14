from __future__ import annotations

from typing import Any, Dict, Sequence

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage


def create_replan_prompt(state: dict, messages: Sequence[BaseMessage]) -> SystemMessage:
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

