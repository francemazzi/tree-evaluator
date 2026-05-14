from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st

from streamlit_app.services.data_manager import DynamicDataManager

_CUSTOM_DATASET_KEYS = [
    "custom_db_path",
    "custom_table_name",
    "stored_data_description",
    "uploaded_file_name",
    "uploaded_file_signature",
    "dataset_metadata",
]


def render_dataset_settings(ui) -> None:
    """Render dataset source controls."""
    st.divider()
    st.header("📊 Gestione Dataset")

    data_source = st.radio(
        "Fonte Dati",
        [
            "🇦🇹 Dataset Vienna (229K alberi)",
            "🇮🇹 Dataset Milano (251K alberi)",
            "📁 Carica CSV Personalizzato",
        ],
        key="data_source_radio",
    )

    if data_source == "🇦🇹 Dataset Vienna (229K alberi)":
        _select_vienna_dataset(ui)
    elif data_source == "🇮🇹 Dataset Milano (251K alberi)":
        _select_milano_dataset(ui)
    elif data_source == "📁 Carica CSV Personalizzato":
        _render_custom_csv_upload(ui)


def _clear_custom_dataset_state(include_selected_preset: bool = False) -> None:
    keys = list(_CUSTOM_DATASET_KEYS)
    if include_selected_preset:
        keys.append("selected_preset")
    for key in keys:
        if key in st.session_state:
            del st.session_state[key]


def _select_vienna_dataset(ui) -> None:
    _clear_custom_dataset_state(include_selected_preset=True)
    if st.session_state.get("selected_preset") != "vienna":
        st.session_state.selected_preset = "vienna"
        ui._service._agent = None

    st.info("""🌳 **Dataset Vienna Trees (BAUMKATOGD)**

- **Alberi totali:** 229.298
- **Distretti:** 23
- **Colonne principali:** specie, anno piantumazione, circonferenza, altezza, via
- **Periodo:** Dati storici fino ad oggi
    """)


def _select_milano_dataset(ui) -> None:
    _clear_custom_dataset_state()
    if st.session_state.get("selected_preset") != "milano":
        st.session_state.selected_preset = "milano"
        ui._service._agent = None

    st.info("""🌳 **Dataset Milano Trees**

- **Alberi totali:** 251.165
- **Municipi:** 9
- **Specie uniche:** 338
- **Colonne principali:** genere, specie, varietà, diametro tronco, altezza, via, coordinate GPS
- **Periodo:** Dal 1984 ad oggi
    """)


def _render_custom_csv_upload(ui) -> None:
    st.info("📁 Carica un file CSV per analizzarlo con l'AI")

    uploaded_file = st.file_uploader(
        "Seleziona file CSV",
        type=["csv"],
        key="csv_uploader",
        help="Il file verrà automaticamente convertito in database SQL",
    )

    stored_value = st.session_state.get("stored_data_description", "")
    description = st.text_area(
        "Descrizione dati (opzionale)",
        value=stored_value,
        placeholder="Es: Questo dataset contiene vendite mensili per regione dal 2020 al 2024...",
        key="data_description_input",
        help="Fornisci un contesto che aiuti l'AI a comprendere meglio i tuoi dati",
        height=100,
    )

    if uploaded_file:
        _process_uploaded_file(ui, uploaded_file, description)

    if st.button("🔄 Torna al Dataset Vienna", use_container_width=True):
        _clear_custom_dataset_state(include_selected_preset=True)
        ui._service._agent = None
        st.rerun()


def _process_uploaded_file(ui, uploaded_file, description: str) -> None:
    current_file_name = st.session_state.get("uploaded_file_name", None)
    file_bytes = uploaded_file.getbuffer()
    uploaded_file_signature = hashlib.sha256(
        file_bytes.tobytes() if hasattr(file_bytes, "tobytes") else bytes(file_bytes)
    ).hexdigest()
    current_file_signature = st.session_state.get("uploaded_file_signature", None)

    if current_file_name != uploaded_file.name or current_file_signature != uploaded_file_signature:
        _convert_and_store_uploaded_file(ui, uploaded_file, description, uploaded_file_signature)
    else:
        st.success(f"✅ Dataset attivo: {uploaded_file.name}")
        metadata = st.session_state.get("dataset_metadata")
        if metadata:
            _render_dataset_metadata(metadata)


def _convert_and_store_uploaded_file(ui, uploaded_file, description: str, uploaded_file_signature: str) -> None:
    with st.spinner("📥 Caricamento e conversione CSV in corso..."):
        try:
            manager = DynamicDataManager(Path("temp_data"))
            db_path, table_name, metadata = manager.process_uploaded_file(uploaded_file)

            st.session_state.custom_db_path = str(db_path)
            st.session_state.custom_table_name = table_name
            st.session_state.stored_data_description = description
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.uploaded_file_signature = metadata.get("file_hash", uploaded_file_signature)
            st.session_state.dataset_metadata = metadata

            ui._service._agent = None
            st.success("✅ Dataset caricato con successo!")
            _render_dataset_metadata(metadata, table_name=table_name)
        except Exception as exc:
            st.error(f"❌ Errore nel caricamento: {str(exc)}")
            if "custom_db_path" in st.session_state:
                del st.session_state.custom_db_path


def _render_dataset_metadata(metadata: dict, table_name: str | None = None) -> None:
    with st.expander("📋 Info Dataset"):
        st.write(f"**File:** {metadata['original_filename']}")
        st.write(f"**Righe:** {metadata['row_count']:,}")
        st.write(f"**Colonne:** {metadata['column_count']}")
        st.write(f"**Separatore rilevato:** `{metadata.get('detected_delimiter', ',')}`")
        st.write(f"**Encoding:** `{metadata.get('detected_encoding', 'n/d')}`")
        if table_name:
            st.write(f"**Tabella SQL:** `{table_name}`")
        if metadata.get("warnings"):
            st.warning("\n".join(metadata["warnings"]))
        if metadata.get("profile_summary"):
            st.text_area(
                "Profilo automatico",
                value=metadata["profile_summary"],
                height=180,
                disabled=True,
            )
        if metadata.get("column_mapping"):
            st.write("\n**Colonne:**")
            for original, sql_name in metadata["column_mapping"].items():
                st.write(f"- {original} → `{sql_name}`")
