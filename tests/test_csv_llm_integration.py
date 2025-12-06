"""
Realistic LLM integration test for CSV upload + text-to-SQL.

This test simula un utente che chiede al modello una query in linguaggio
naturale; l'LLM genera SQL sul database creato dal CSV caricato. Se non è
presente una OPENAI_API_KEY, il test viene saltato (per evitare dipendenze
hardcodate).
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
import tempfile

import pandas as pd
import pytest
from langchain_openai import ChatOpenAI

from streamlit_app.services.data_manager import DynamicDataManager
from streamlit_app.tools.dataset_tool import DatasetQueryTool


class FakeUploadedFile:
    """Mock Streamlit uploaded file for testing."""

    def __init__(self, name: str, content: str):
        self.name = name
        self._content = content

    def getbuffer(self):
        return self._content.encode("utf-8")


@pytest.fixture(scope="module")
def sample_csv_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a realistic sample CSV on disk (no hardcoded numbers in assertions)."""
    csv_content = """Regione,Mese,Anno,Vendite,Prodotto
Lombardia,Gennaio,2023,15000,Laptop
Lazio,Gennaio,2023,12000,Laptop
Toscana,Gennaio,2023,8500,Laptop
Lombardia,Febbraio,2023,16500,Laptop
Lazio,Febbraio,2023,13200,Laptop
Toscana,Febbraio,2023,9100,Laptop
Lombardia,Marzo,2023,18000,Laptop
Lazio,Marzo,2023,14500,Laptop
Toscana,Marzo,2023,10200,Laptop
Lombardia,Gennaio,2023,8000,Tablet
Lazio,Gennaio,2023,6500,Tablet
Toscana,Gennaio,2023,4200,Tablet
Lombardia,Febbraio,2023,8500,Tablet
Lazio,Febbraio,2023,7000,Tablet
Toscana,Febbraio,2023,4800,Tablet
Lombardia,Marzo,2023,9200,Tablet
Lazio,Marzo,2023,7500,Tablet
Toscana,Marzo,2023,5100,Tablet
"""
    csv_path = tmp_path_factory.mktemp("data") / "vendite_llm.csv"
    csv_path.write_text(csv_content, encoding="utf-8")
    return csv_path


@pytest.fixture(scope="module")
def df_expected(sample_csv_path: Path) -> pd.DataFrame:
    """DataFrame per calcolare aspettative dinamiche (senza numeri hardcoded nel test)."""
    return pd.read_csv(sample_csv_path)


def _build_tool_from_csv(csv_path: Path, upload_dir: Path, llm: ChatOpenAI) -> DatasetQueryTool:
    """Carica il CSV via DynamicDataManager e restituisce il tool configurato."""
    manager = DynamicDataManager(upload_dir)
    fake_file = FakeUploadedFile(csv_path.name, csv_path.read_text(encoding="utf-8"))
    db_path, table_name, _ = manager.process_uploaded_file(fake_file)
    return DatasetQueryTool(
        db_path=db_path,
        table_name=table_name,
        user_description="Dataset di vendite mensili per regione e prodotto",
        llm=llm,
    )


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Richiede OPENAI_API_KEY per testare LLM reale (no hardcode).",
)
def test_llm_generates_correct_sum_vendite(sample_csv_path: Path, df_expected: pd.DataFrame):
    """Test end-to-end con LLM: domanda in italiano -> SQL generato -> risultato corretto."""
    with tempfile.TemporaryDirectory() as tmpdir:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
        tool = _build_tool_from_csv(sample_csv_path, Path(tmpdir), llm)

        # Calcola aspettativa dal CSV dinamicamente
        expected_sum = int(df_expected["Vendite"].sum())

        # Recupera schema e traduce la domanda in SQL
        conn = tool._get_connection()
        schema_info = tool._get_schema_info(conn)
        question = "Qual è il totale delle vendite?"
        sql = tool._translate_to_sql(question, schema_info)

        # Esegui SQL e confronta
        result = tool._execute_sql(conn, sql, natural_query=question)
        conn.close()

        assert result.get("result") == expected_sum, f"Totale vendite atteso {expected_sum}, ottenuto {result}"


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Richiede OPENAI_API_KEY per testare LLM reale (no hardcode).",
)
def test_llm_group_by_regione(sample_csv_path: Path, df_expected: pd.DataFrame):
    """Test LLM: raggruppa vendite per regione e confronta con attesi dinamici."""
    with tempfile.TemporaryDirectory() as tmpdir:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
        tool = _build_tool_from_csv(sample_csv_path, Path(tmpdir), llm)

        expected_by_region = df_expected.groupby("Regione")["Vendite"].sum().to_dict()

        conn = tool._get_connection()
        schema_info = tool._get_schema_info(conn)
        question = "Mostrami il totale delle vendite per regione, ordinate dal valore più alto."
        sql = tool._translate_to_sql(question, schema_info)

        result = tool._execute_sql(conn, sql, natural_query=question)
        conn.close()

        # Normalizza risultato
        rows = result.get("results") or []
        obtained = {row["regione"]: row["totale"] for row in rows if "regione" in row and "totale" in row}

        assert obtained == expected_by_region, f"Atteso {expected_by_region}, ottenuto {obtained}"


