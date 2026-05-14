from __future__ import annotations

from typing import List

import streamlit as st

from streamlit_app.llm.ollama_client import OllamaClient


def render_ollama_settings(ui, settings) -> None:
    """Render local Ollama settings."""
    st.info("Stai usando **Ollama in locale**. La chiave OpenAI non è necessaria.")

    base_url = st.text_input(
        "Ollama base URL",
        value=settings.ollama_base_url,
        help=(
            "Suggerito: macOS/Windows Docker Desktop → http://host.docker.internal:11434; "
            "Linux → http://172.17.0.1:11434; fuori da Docker → http://localhost:11434"
        ),
    )

    if "ollama_models_cache" not in st.session_state:
        st.session_state.ollama_models_cache = []

    cols = st.columns([1, 1, 2])
    with cols[0]:
        refresh = st.button("↻ Aggiorna modelli", use_container_width=True)
    with cols[1]:
        use_cached = st.checkbox("Usa cache", value=True)
    if refresh or (not use_cached) or not st.session_state.ollama_models_cache:
        client = OllamaClient(base_url=base_url.strip() or settings.ollama_base_url)
        st.session_state.ollama_models_cache = client.list_model_names()

    model_names: List[str] = st.session_state.ollama_models_cache
    if not model_names:
        st.error("Non riesco a leggere **nessun modello** da Ollama (o Ollama non è raggiungibile).")
        st.markdown("Se non hai ancora scaricato nulla, esegui sul tuo host:")
        st.code("ollama pull nomic-embed-text\nollama pull gpt-oss:20b", language="bash")
        st.session_state.llm_block_reason = (
            "Ollama non ha modelli disponibili (o non è raggiungibile). "
            "Scarica almeno un modello chat e un modello embeddings, poi premi “Aggiorna modelli”."
        )

    chat_model = st.selectbox(
        "Modello Ollama (chat)",
        options=model_names if model_names else [settings.ollama_chat_model],
        index=(
            model_names.index(settings.ollama_chat_model)
            if model_names and settings.ollama_chat_model in model_names
            else 0
        ),
    )

    embedding_candidates = [
        name for name in model_names if any(key in name.lower() for key in ("embed", "bge", "mxbai"))
    ]
    if model_names and not embedding_candidates:
        st.warning("Non vedo embedding model tra quelli installati in Ollama.")
        st.markdown("Suggerimento (host):")
        st.code("ollama pull nomic-embed-text", language="bash")
    embedding_options = embedding_candidates or (
        model_names if model_names else [settings.ollama_embedding_model]
    )
    embed_model = st.selectbox(
        "Modello Ollama (embeddings)",
        options=embedding_options,
        index=(
            embedding_options.index(settings.ollama_embedding_model)
            if settings.ollama_embedding_model in embedding_options
            else 0
        ),
    )
    if embed_model and not any(key in embed_model.lower() for key in ("embed", "bge", "mxbai")):
        st.warning("Il modello embeddings selezionato non sembra un embedding model. Consigliato: **nomic-embed-text** (o simili).")

    if st.button("Salva impostazioni Ollama"):
        if not model_names:
            st.error("Impossibile salvare: Ollama non ha modelli disponibili (o non è raggiungibile).")
            st.stop()
        settings.ollama_base_url = base_url.strip() or settings.ollama_base_url
        settings.ollama_chat_model = chat_model
        settings.ollama_embedding_model = embed_model
        ui._service.save_user_llm_settings(settings)
        ui._service._agent = None
        st.success("✅ Impostazioni Ollama salvate!")
        st.rerun()
