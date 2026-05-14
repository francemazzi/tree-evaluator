from __future__ import annotations

import streamlit as st


def render_anthropic_settings(ui, settings) -> None:
    """Render Anthropic authentication and model settings."""
    auth_labels = list(ui._ANTHROPIC_AUTH_METHODS.keys())
    current_auth_label = ui._anthropic_auth_method_label(settings.anthropic_auth_method)
    selected_auth_label = st.radio(
        "Metodo autenticazione Anthropic",
        options=auth_labels,
        index=auth_labels.index(current_auth_label),
        help="Puoi usare il login Claude OAuth oppure una API key Anthropic.",
    )
    selected_auth_method = ui._ANTHROPIC_AUTH_METHODS[selected_auth_label]
    if selected_auth_method != settings.anthropic_auth_method:
        settings.anthropic_auth_method = selected_auth_method
        ui._service.save_user_llm_settings(settings)
        ui._service._agent = None
        st.rerun()

    if settings.anthropic_auth_method == "api_key":
        _render_anthropic_api_key_settings(ui, settings)
    else:
        _render_anthropic_oauth_settings(ui, settings)

    _render_anthropic_model_settings(ui, settings)


def _render_anthropic_api_key_settings(ui, settings) -> None:
    col1, col2 = st.columns([2, 1])
    with col1:
        anthropic_api_key = st.text_input(
            "Anthropic API Key",
            value=settings.anthropic_api_key,
            type="password",
            key="anthropic_api_key_input",
            help="Inserisci la tua chiave API Anthropic (sk-ant-...).",
            placeholder="sk-ant-...",
        )
        st.link_button(
            "Apri Console Anthropic",
            ui._ANTHROPIC_CONSOLE_KEYS_URL,
            use_container_width=True,
        )
    with col2:
        st.write("")
        st.write("")
        if st.button("Salva Anthropic", use_container_width=True):
            settings.anthropic_api_key = anthropic_api_key.strip()
            settings.anthropic_auth_method = "api_key"
            ui._service.save_user_llm_settings(settings)
            ui._service._agent = None
            st.success("✅ Impostazioni Anthropic salvate!")
            st.rerun()


def _render_anthropic_oauth_settings(ui, settings) -> None:
    from streamlit_app.llm.anthropic_oauth import (
        MANUAL_REDIRECT_URI,
        build_authorize_url,
        exchange_code_for_tokens,
        generate_pkce,
        generate_state,
        parse_pasted_code,
        start_loopback_server,
    )

    if st.button("Avvia login Claude", use_container_width=True):
        _start_claude_oauth_flow(build_authorize_url, generate_pkce, generate_state, start_loopback_server)

    oauth_flow = st.session_state.get("anthropic_oauth_flow") or {}
    if oauth_flow:
        _render_active_oauth_flow(
            ui,
            settings,
            oauth_flow,
            exchange_code_for_tokens,
            parse_pasted_code,
            build_authorize_url,
            generate_pkce,
            generate_state,
            MANUAL_REDIRECT_URI,
        )

    if settings.anthropic_oauth_refresh_token:
        st.success("Login Claude OAuth configurato.")
        st.caption("La chat userà il backend Claude associato al tuo account.")
        st.session_state.llm_block_reason = None
    else:
        st.session_state.llm_block_reason = (
            "Completa il login Claude OAuth oppure usa una API key Anthropic."
        )


def _start_claude_oauth_flow(build_authorize_url, generate_pkce, generate_state, start_loopback_server) -> None:
    try:
        existing_flow = st.session_state.get("anthropic_oauth_flow") or {}
        existing_server = existing_flow.get("server")
        if existing_server:
            existing_server.close()
        pkce = generate_pkce()
        state = generate_state()
        loopback = start_loopback_server(state)
        st.session_state.anthropic_oauth_flow = {
            "mode": "loopback",
            "state": state,
            "verifier": pkce.verifier,
            "redirect_uri": loopback.redirect_uri,
            "server": loopback,
            "authorize_url": build_authorize_url(
                redirect_uri=loopback.redirect_uri,
                code_challenge=pkce.challenge,
                state=state,
            ),
        }
        st.rerun()
    except Exception as exc:
        st.warning("Loopback OAuth non disponibile. Uso il fallback copy/paste.")
        pkce = generate_pkce()
        state = generate_state()
        from streamlit_app.llm.anthropic_oauth import MANUAL_REDIRECT_URI

        st.session_state.anthropic_oauth_flow = {
            "mode": "manual",
            "state": state,
            "verifier": pkce.verifier,
            "redirect_uri": MANUAL_REDIRECT_URI,
            "authorize_url": build_authorize_url(
                redirect_uri=MANUAL_REDIRECT_URI,
                code_challenge=pkce.challenge,
                state=state,
            ),
            "error": str(exc),
        }
        st.rerun()


def _render_active_oauth_flow(
    ui,
    settings,
    oauth_flow,
    exchange_code_for_tokens,
    parse_pasted_code,
    build_authorize_url,
    generate_pkce,
    generate_state,
    manual_redirect_uri,
) -> None:
    if oauth_flow.get("error"):
        st.caption(f"Fallback attivo: {oauth_flow['error']}")
    st.link_button(
        "Login Claude",
        str(oauth_flow.get("authorize_url") or ""),
        use_container_width=True,
    )
    if oauth_flow.get("mode") == "loopback":
        _render_loopback_completion(ui, settings, oauth_flow, exchange_code_for_tokens)

    with st.expander("Fallback copy/paste", expanded=oauth_flow.get("mode") == "manual"):
        st.caption(
            "Se il loopback non funziona, usa questo link e incolla qui il codice "
            "mostrato dal browser (CODE oppure CODE#STATE)."
        )
        if st.button("Usa fallback copy/paste", use_container_width=True):
            pkce = generate_pkce()
            state = generate_state()
            st.session_state.anthropic_oauth_flow = {
                "mode": "manual",
                "state": state,
                "verifier": pkce.verifier,
                "redirect_uri": manual_redirect_uri,
                "authorize_url": build_authorize_url(
                    redirect_uri=manual_redirect_uri,
                    code_challenge=pkce.challenge,
                    state=state,
                ),
            }
            st.rerun()
        pasted_code = st.text_input(
            "Codice autorizzazione Claude",
            value="",
            key="anthropic_oauth_code_input",
            placeholder="CODE oppure CODE#STATE",
        )
        if st.button("Completa pairing Claude", use_container_width=True):
            _complete_manual_pairing(
                ui,
                settings,
                oauth_flow,
                pasted_code,
                exchange_code_for_tokens,
                parse_pasted_code,
                manual_redirect_uri,
            )


def _render_loopback_completion(ui, settings, oauth_flow, exchange_code_for_tokens) -> None:
    st.info(
        "Dopo il login Claude, torna qui e premi “Controlla callback” "
        "se la pagina non si aggiorna automaticamente."
    )
    server = oauth_flow.get("server")
    callback = server.poll() if server else None
    if st.button("Controlla callback Claude", use_container_width=True):
        callback = server.poll() if server else callback
    if callback:
        try:
            token_payload = exchange_code_for_tokens(
                code=str(callback.get("code") or ""),
                state=str(callback.get("state") or ""),
                verifier=str(oauth_flow.get("verifier") or ""),
                redirect_uri=str(oauth_flow.get("redirect_uri") or ""),
            )
            _save_anthropic_refresh_token(ui, settings, token_payload)
            if server:
                server.close()
            del st.session_state.anthropic_oauth_flow
            st.success("✅ Login Claude salvato!")
            st.rerun()
        except Exception as exc:
            st.error(f"Impossibile completare il login Claude: {exc}")


def _complete_manual_pairing(
    ui,
    settings,
    oauth_flow,
    pasted_code: str,
    exchange_code_for_tokens,
    parse_pasted_code,
    manual_redirect_uri: str,
) -> None:
    try:
        parsed = parse_pasted_code(
            pasted_code,
            default_state=str(oauth_flow.get("state") or ""),
        )
        token_payload = exchange_code_for_tokens(
            code=parsed["code"],
            state=parsed.get("state") or str(oauth_flow.get("state") or ""),
            verifier=str(oauth_flow.get("verifier") or ""),
            redirect_uri=str(oauth_flow.get("redirect_uri") or manual_redirect_uri),
        )
        _save_anthropic_refresh_token(ui, settings, token_payload)
        server = oauth_flow.get("server")
        if server:
            server.close()
        del st.session_state.anthropic_oauth_flow
        st.success("✅ Login Claude salvato!")
        st.rerun()
    except Exception as exc:
        st.error(f"Impossibile completare il pairing Claude: {exc}")


def _save_anthropic_refresh_token(ui, settings, token_payload) -> None:
    refresh_token = str(token_payload.get("refresh_token") or "").strip()
    if not refresh_token:
        raise ValueError("il refresh token non è presente nella risposta OAuth.")
    settings.anthropic_oauth_refresh_token = refresh_token
    settings.anthropic_auth_method = "oauth"
    ui._service.save_user_llm_settings(settings)
    ui._service._agent = None


def _render_anthropic_model_settings(ui, settings) -> None:
    is_anthropic_oauth = settings.anthropic_auth_method == "oauth"
    anthropic_options = ui._build_anthropic_chat_model_options(
        settings.anthropic_chat_model,
        for_oauth=is_anthropic_oauth,
    )
    anthropic_selected = st.selectbox(
        "Modello Claude (chat)",
        options=anthropic_options,
        index=(
            anthropic_options.index(settings.anthropic_chat_model)
            if settings.anthropic_chat_model in anthropic_options
            else 0
        ),
        help=(
            "Con login Claude OAuth sono mostrati solo i modelli compatibili. "
            "Con API key puoi inserire un modello custom."
        ),
    )
    custom_anthropic_model = ""
    if anthropic_selected == "Altro…":
        custom_anthropic_model = st.text_input(
            "Nome modello Claude (custom)",
            value=settings.anthropic_chat_model,
            help="Inserisci il nome esatto del modello Claude.",
        )
    resolved_anthropic_model = ui._resolve_model_selection(
        anthropic_selected,
        custom_anthropic_model,
    )
    if resolved_anthropic_model and resolved_anthropic_model != (
        settings.anthropic_chat_model or ""
    ).strip():
        if st.button("Applica modello Claude"):
            settings.anthropic_chat_model = resolved_anthropic_model
            ui._service.save_user_llm_settings(settings)
            ui._service._agent = None
            st.success("✅ Modello Claude aggiornato!")
            st.rerun()
