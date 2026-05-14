from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


class FakeUploadedFile:
    """Mock Streamlit uploaded file for testing."""

    def __init__(self, name: str, content: str | bytes):
        self.name = name
        self._content = content

    def getbuffer(self):
        if isinstance(self._content, bytes):
            return self._content
        return self._content.encode("utf-8")


@pytest.fixture
def temp_upload_dir():
    """Create a temporary directory for uploads."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_csv_content():
    """Sample CSV data for testing."""
    return """Regione,Mese,Anno,Vendite,Prodotto
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
Toscana,Marzo,2023,5100,Tablet"""
