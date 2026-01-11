"""Translation dictionary for agent UI messages."""

from __future__ import annotations

from typing import Dict, Literal

Language = Literal["it", "en"]

TRANSLATIONS: Dict[str, Dict[Language, str]] = {
    # Language detector
    "language_detected": {
        "it": "**Italiano**",
        "en": "**English**"
    },
    
    # Context manager
    "context_management": {
        "it": "🧹 **Gestione Contesto**",
        "en": "🧹 **Context Management**"
    },
    "original_messages": {
        "it": "Messaggi originali",
        "en": "Original messages"
    },
    "optimized_messages": {
        "it": "Messaggi ottimizzati",
        "en": "Optimized messages"
    },
    "context_compressed": {
        "it": "Contesto lungo compresso per evitare limiti di token.",
        "en": "Long context compressed to avoid token limits."
    },
    
    # Query optimizer
    "query_optimization": {
        "it": "🔍 **Ottimizzazione Query**",
        "en": "🔍 **Query Optimization**"
    },
    "optimized_query_label": {
        "it": "Query ottimizzata",
        "en": "Optimized query"
    },
    "tasks_identified": {
        "it": "**Task identificati:**",
        "en": "**Identified tasks:**"
    },
    
    # Tool calls
    "tool_call": {
        "it": "🛠️ **Chiamata Tool**",
        "en": "🛠️ **Tool Call**"
    },
    "tool": {
        "it": "Tool",
        "en": "Tool"
    },
    "query": {
        "it": "Query",
        "en": "Query"
    },
    "dbh": {
        "it": "DBH",
        "en": "DBH"
    },
    "height": {
        "it": "Altezza",
        "en": "Height"
    },
    "wood_density": {
        "it": "Densità legno",
        "en": "Wood density"
    },
    "chart_type": {
        "it": "Tipo grafico",
        "en": "Chart type"
    },
    
    # Tool results
    "tool_results": {
        "it": "✅ **Risultati Tool**",
        "en": "✅ **Tool Results**"
    },
    "tool_executed": {
        "it": "✅ **Tool Eseguito**",
        "en": "✅ **Tool Executed**"
    },
    "processing_results": {
        "it": "Elaborazione risultati...",
        "en": "Processing results..."
    },
    "sql_query_generated": {
        "it": "**Query SQL generata:**",
        "en": "**SQL query generated:**"
    },
    "vector_search_applied": {
        "it": "🔍 **Vector Search Applicata**",
        "en": "🔍 **Vector Search Applied**"
    },
    "total_rows_found": {
        "it": "**Righe totali trovate**",
        "en": "**Total rows found**"
    },
    "top_relevant_results": {
        "it": "**Top risultati più rilevanti**",
        "en": "**Top most relevant results**"
    },
    "rows_found": {
        "it": "**Righe trovate**",
        "en": "**Rows found**"
    },
    "warning": {
        "it": "**Attenzione**",
        "en": "**Warning**"
    },
    "first_results": {
        "it": "**Primi risultati:**",
        "en": "**First results:**"
    },
    "species": {
        "it": "Specie",
        "en": "Species"
    },
    "count": {
        "it": "Count",
        "en": "Count"
    },
    "district": {
        "it": "Distretto",
        "en": "District"
    },
    "circumference": {
        "it": "Circonferenza",
        "en": "Circumference"
    },
    "and_others": {
        "it": "... e altri {count} risultati",
        "en": "... and {count} more results"
    },
    "co2_sequestered": {
        "it": "**CO2 sequestrato**",
        "en": "**CO2 sequestered**"
    },
    
    # Budget check
    "budget_limit": {
        "it": "⚠️ **Budget Limit**",
        "en": "⚠️ **Budget Limit**"
    },
    "execution_limit_reached": {
        "it": "Limite di esecuzione raggiunto. Interruzione per prevenire loop infiniti.",
        "en": "Execution limit reached. Stopping to prevent infinite loops."
    },
    "budget_check": {
        "it": "✓ **Budget Check**",
        "en": "✓ **Budget Check**"
    },
    "tool_calls": {
        "it": "Tool calls",
        "en": "Tool calls"
    },
    "time": {
        "it": "Tempo",
        "en": "Time"
    },
    
    # Tool loop guard
    "stop_anti_loop": {
        "it": "🛑 **Stop Anti-Loop**",
        "en": "🛑 **Stop Anti-Loop**"
    },
    "repetition_detected": {
        "it": "Rilevata ripetizione della stessa chiamata tool. Interrompo ed entro in modalità chiarimento.",
        "en": "Detected repetition of the same tool call. Stopping and entering clarification mode."
    },
    "recovery_anti_loop": {
        "it": "🔁 **Recovery Anti-Loop**",
        "en": "🔁 **Recovery Anti-Loop**"
    },
    "repetition_detected_replan": {
        "it": "Rilevata ripetizione della stessa chiamata tool. Provo a cambiare strategia (replanning).",
        "en": "Detected repetition of the same tool call. Trying to change strategy (replanning)."
    },
    
    # Replanning
    "replanning": {
        "it": "🧠 **Replanning**",
        "en": "🧠 **Replanning**"
    },
    "reformulating_step": {
        "it": "Sto riformulando il prossimo passo per evitare ripetizioni.",
        "en": "Reformulating the next step to avoid repetitions."
    },
    
    # Validator
    "validation_completed": {
        "it": "✓ **Validazione Completata**",
        "en": "✓ **Validation Completed**"
    },
    "response_complete_accurate": {
        "it": "La risposta è completa e accurata.",
        "en": "The response is complete and accurate."
    },
    "validation": {
        "it": "⚠️ **Validazione**",
        "en": "⚠️ **Validation**"
    },
    "retry_limit_reached": {
        "it": "Raggiunto limite retry. Proseguo con la risposta attuale.",
        "en": "Retry limit reached. Proceeding with current response."
    },
    "validation_attempt": {
        "it": "⚠️ **Validazione (Tentativo {count})**",
        "en": "⚠️ **Validation (Attempt {count})**"
    },
    "missing_tasks": {
        "it": "Task mancanti",
        "en": "Missing tasks"
    },
    "reprocessing_response": {
        "it": "Rielaborazione risposta...",
        "en": "Reprocessing response..."
    },
}


def get_translation(key: str, language: Language = "it") -> str:
    """Get translated text for a given key and language.
    
    Args:
        key: Translation key
        language: Language code ("it" or "en")
        
    Returns:
        Translated text, or the key itself if not found
    """
    if key not in TRANSLATIONS:
        return key
    
    translations = TRANSLATIONS[key]
    return translations.get(language, translations.get("it", key))


def format_translation(key: str, language: Language = "it", **kwargs) -> str:
    """Get and format translated text with placeholders.
    
    Args:
        key: Translation key
        language: Language code ("it" or "en")
        **kwargs: Format arguments
        
    Returns:
        Formatted translated text
    """
    text = get_translation(key, language)
    try:
        return text.format(**kwargs)
    except KeyError:
        return text

