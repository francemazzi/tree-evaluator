from __future__ import annotations

import streamlit as st


def render_openai_settings(ui, settings) -> None:
    """Render OpenAI authentication and model settings."""
    auth_labels = list(ui._OPENAI_AUTH_METHODS.keys())
    current_auth_label = ui._openai_auth_method_label(settings.openai_auth_method)
    selected_auth_label = st.radio(
        "Metodo autenticazione OpenAI",
        options=auth_labels,
        index=auth_labels.index(current_auth_label),
        help=(
            "Puoi usare una API key della piattaforma OpenAI oppure un login ChatGPT OAuth "
            "con refresh token."
        ),
    )
    selected_auth_method = ui._OPENAI_AUTH_METHODS[selected_auth_label]
    if selected_auth_method != settings.openai_auth_method:
        settings.openai_auth_method = selected_auth_method
        ui._service.save_user_llm_settings(settings)
        ui._service._agent = None
        st.rerun()

    if settings.openai_auth_method == "api_key":
        _render_openai_api_key_settings(ui, settings)
    else:
        _render_openai_oauth_settings(ui, settings)

    _render_openai_model_settings(ui, settings)


def _render_openai_api_key_settings(ui, settings) -> None:
    col1, col2 = st.columns([2, 1])
    with col1:
        new_api_key = st.text_input(
            "OpenAI API Key",
            value=settings.openai_api_key,
            type="password",
            key="api_key_input",
            help="Inserisci la tua chiave API OpenAI (sk-...). Verrà salvata in modo persistente.",
            placeholder="sk-...",
        )
        st.link_button(
            "Apri pagina API keys",
            ui._OPENAI_API_KEYS_URL,
            use_container_width=True,
        )
    with col2:
        st.write("")
        st.write("")
        if st.button("Salva", use_container_width=True):
            settings.openai_api_key = new_api_key.strip()
            settings.openai_auth_method = "api_key"
            ui._service.save_user_llm_settings(settings)
            ui._service._agent = None
            st.success("✅ Impostazioni salvate!")
            st.rerun()


def _render_openai_oauth_settings(ui, settings) -> None:
    if st.button("Genera codice dispositivo", use_container_width=True):
        try:
            from streamlit_app.llm.openai_oauth import request_device_code

            st.session_state.openai_device_pairing = request_device_code()
        except Exception as exc:
            st.error(f"Impossibile generare il codice dispositivo: {exc}")

    device_pairing = st.session_state.get("openai_device_pairing")
    if device_pairing:
        st.caption("Inserisci questo codice nella pagina di device pairing.")
        st.code(str(device_pairing.get("user_code") or ""), language="text")
        if device_pairing.get("expires_at"):
            st.caption(f"Scade: {device_pairing['expires_at']}")
        st.link_button(
            "OpenAI Codex Device Pairing",
            ui._OPENAI_CODEX_DEVICE_PAIRING_URL,
            use_container_width=True,
        )
        st.caption("Pair in browser with a device code.")
        if st.button("Ho completato il pairing: salva token", use_container_width=True):
            _complete_openai_device_pairing(ui, settings, device_pairing)

    with st.expander("Ho già un refresh token", expanded=False):
        col1, col2 = st.columns([2, 1])
        with col1:
            new_refresh_token = st.text_input(
                "Refresh token OAuth",
                value=settings.openai_codex_oauth_token,
                type="password",
                key="openai_codex_oauth_token_input",
                help="Se hai già un refresh token, puoi incollarlo direttamente senza rifare il login.",
                placeholder="refresh token",
            )
        with col2:
            st.write("")
            st.write("")
            if st.button("Salva token", use_container_width=True):
                settings.openai_codex_oauth_token = new_refresh_token.strip()
                settings.openai_auth_method = "codex_oauth"
                ui._service.save_user_llm_settings(settings)
                ui._service._agent = None
                st.success("✅ Refresh token salvato!")
                st.rerun()

    st.info(
        "Il codice ora viene generato direttamente da questa sidebar. "
        "Apri il link di device pairing, inserisci il codice visualizzato qui e poi premi "
        "“Ho completato il pairing: salva token”."
    )
    if settings.openai_codex_oauth_token:
        st.success("Device pairing configurato.")
        st.caption("La chat userà il backend ChatGPT/Codex associato al tuo account.")
        st.session_state.llm_block_reason = None
    else:
        st.session_state.llm_block_reason = (
            "Completa il login ChatGPT OAuth o incolla un refresh token, "
            "oppure usa una API key OpenAI Platform."
        )


def _complete_openai_device_pairing(ui, settings, device_pairing) -> None:
    try:
        from streamlit_app.llm.openai_oauth import (
            DeviceAuthorizationPending,
            complete_device_code_login,
        )

        token_payload = complete_device_code_login(
            str(device_pairing.get("device_auth_id") or ""),
            str(device_pairing.get("user_code") or ""),
        )
        refresh_token = str(token_payload.get("refresh_token") or "").strip()
        if not refresh_token:
            raise ValueError("il refresh token non è presente nella risposta OAuth.")
        settings.openai_codex_oauth_token = refresh_token
        settings.openai_auth_method = "codex_oauth"
        ui._service.save_user_llm_settings(settings)
        ui._service._agent = None
        del st.session_state.openai_device_pairing
        st.success("✅ Login ChatGPT salvato!")
        st.rerun()
    except DeviceAuthorizationPending:
        st.warning("Il pairing non risulta ancora completato. Inserisci il codice nel browser e riprova.")
    except Exception as exc:
        st.error(f"Impossibile completare il pairing: {exc}")


def _render_openai_model_settings(ui, settings) -> None:
    is_codex_oauth = settings.openai_auth_method == "codex_oauth"
    chat_options = ui._build_openai_chat_model_options(
        settings.openai_chat_model,
        for_codex_oauth=is_codex_oauth,
    )
    chat_help = (
        "Con login ChatGPT (OAuth) sono ammessi solo i modelli del Codex CLI "
        "(es. gpt-5.5, gpt-5.4). I modelli della Platform (gpt-5, gpt-4o…) "
        "sono accettati solo con API key."
        if is_codex_oauth
        else "Seleziona il modello chat OpenAI. Puoi scegliere “Altro…” per inserire un nome custom."
    )
    chat_selected = st.selectbox(
        "Modello OpenAI (chat)",
        options=chat_options,
        index=(
            chat_options.index(settings.openai_chat_model)
            if settings.openai_chat_model in chat_options
            else 0
        ),
        help=chat_help,
    )
    custom_chat = ""
    if chat_selected == "Altro…":
        custom_chat = st.text_input(
            "Nome modello chat (custom)",
            value=settings.openai_chat_model,
            help="Inserisci il nome esatto del modello (es. gpt-5.5 per OAuth, gpt-5 per API key).",
        )

    embed_options = ui._build_openai_embedding_model_options(settings.openai_embedding_model)
    embed_selected = st.selectbox(
        "Modello OpenAI (embeddings)",
        options=embed_options,
        index=(
            embed_options.index(settings.openai_embedding_model)
            if settings.openai_embedding_model in embed_options
            else 0
        ),
        help="Seleziona il modello embeddings OpenAI. Puoi scegliere “Altro…” per inserire un nome custom.",
    )
    custom_embed = ""
    if embed_selected == "Altro…":
        custom_embed = st.text_input(
            "Nome modello embeddings (custom)",
            value=settings.openai_embedding_model,
            help="Inserisci il nome esatto del modello embeddings (es. text-embedding-3-small).",
        )

    resolved_chat = ui._resolve_openai_model_selection(chat_selected, custom_chat)
    resolved_embed = ui._resolve_openai_model_selection(embed_selected, custom_embed)
    changed = (
        resolved_chat
        and resolved_embed
        and (
            resolved_chat != (settings.openai_chat_model or "").strip()
            or resolved_embed != (settings.openai_embedding_model or "").strip()
        )
    )
    if changed and st.button("Applica modelli OpenAI"):
        settings.openai_chat_model = resolved_chat
        settings.openai_embedding_model = resolved_embed
        ui._service.save_user_llm_settings(settings)
        ui._service._agent = None
        st.success("✅ Modelli OpenAI aggiornati!")
        st.rerun()
