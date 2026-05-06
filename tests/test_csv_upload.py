"""
Test suite for CSV upload and dynamic dataset functionality.

This module tests the complete flow from CSV upload to query execution.
"""

import os
import sqlite3
from pathlib import Path
import tempfile
import pytest

from streamlit_app.services.data_manager import DatasetValidationError, DynamicDataManager
from streamlit_app.tools.dataset_tool import DatasetQueryTool


class FakeUploadedFile:
    """Mock Streamlit uploaded file for testing."""
    
    def __init__(self, name: str, content: str | bytes):
        self.name = name
        self._content = content
    
    def getbuffer(self):
        if isinstance(self._content, bytes):
            return self._content
        return self._content.encode('utf-8')


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


class TestDynamicDataManager:
    """Test the DynamicDataManager service."""
    
    def test_initialization(self, temp_upload_dir):
        """Test manager initialization creates directory."""
        manager = DynamicDataManager(temp_upload_dir)
        assert manager.upload_dir.exists()
        assert manager.upload_dir == temp_upload_dir
    
    def test_sanitize_column_name(self, temp_upload_dir):
        """Test column name sanitization."""
        manager = DynamicDataManager(temp_upload_dir)
        
        # Test various column names
        assert manager._sanitize_column_name("Simple") == "simple"
        assert manager._sanitize_column_name("With Spaces") == "with_spaces"
        assert manager._sanitize_column_name("With-Dashes") == "with_dashes"
        assert manager._sanitize_column_name("With__Multiple___Underscores") == "with_multiple_underscores"
        assert manager._sanitize_column_name("CamelCase") == "camelcase"
        assert manager._sanitize_column_name("with.dots") == "with_dots"
    
    def test_process_uploaded_file(self, temp_upload_dir, sample_csv_content):
        """Test complete CSV processing pipeline."""
        manager = DynamicDataManager(temp_upload_dir)
        
        # Create fake uploaded file
        fake_file = FakeUploadedFile("test_vendite.csv", sample_csv_content)
        
        # Process file
        db_path, table_name, metadata = manager.process_uploaded_file(fake_file)
        
        # Verify database was created
        assert db_path.exists()
        assert db_path.suffix == ".db"
        
        # Verify table name
        assert table_name == "uploaded_data"
        
        # Verify metadata
        assert metadata["original_filename"] == "test_vendite.csv"
        assert metadata["row_count"] == 18
        assert metadata["column_count"] == 5
        assert "regione" in metadata["columns"]
        assert "mese" in metadata["columns"]
        assert "anno" in metadata["columns"]
        assert "vendite" in metadata["columns"]
        assert "prodotto" in metadata["columns"]
        
        # Verify column mapping
        assert metadata["column_mapping"]["Regione"] == "regione"
        assert metadata["column_mapping"]["Vendite"] == "vendite"
        
        # Verify dtypes
        assert "anno" in metadata["dtypes"]
        assert "vendite" in metadata["dtypes"]
        assert metadata["detected_delimiter"] == ","
        assert metadata["detected_encoding"] in {"utf-8", "utf-8-sig"}
        assert metadata["file_hash"]
    
    def test_database_content(self, temp_upload_dir, sample_csv_content):
        """Test that database contains correct data."""
        manager = DynamicDataManager(temp_upload_dir)
        fake_file = FakeUploadedFile("test_data.csv", sample_csv_content)
        db_path, table_name, _ = manager.process_uploaded_file(fake_file)
        
        # Connect to database and verify content
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Count rows
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        assert count == 18
        
        # Check column names
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cursor.fetchall()]
        assert "regione" in columns
        assert "mese" in columns
        assert "anno" in columns
        assert "vendite" in columns
        assert "prodotto" in columns
        
        # Check data types
        cursor.execute(f"SELECT anno, vendite FROM {table_name} LIMIT 1")
        row = cursor.fetchone()
        assert isinstance(row[0], int)  # anno should be INTEGER
        assert isinstance(row[1], int)  # vendite should be INTEGER
        
        # Check specific values
        cursor.execute(f"SELECT vendite FROM {table_name} WHERE regione = 'Lombardia' AND mese = 'Gennaio' AND prodotto = 'Laptop'")
        vendite = cursor.fetchone()[0]
        assert vendite == 15000
        
        conn.close()
    
    def test_get_schema_info(self, temp_upload_dir, sample_csv_content):
        """Test schema information retrieval."""
        manager = DynamicDataManager(temp_upload_dir)
        fake_file = FakeUploadedFile("test.csv", sample_csv_content)
        db_path, table_name, _ = manager.process_uploaded_file(fake_file)
        
        schema = manager.get_schema_info(db_path, table_name)
        
        assert "CREATE TABLE" in schema
        assert table_name in schema
        assert "regione" in schema.lower()
        assert "vendite" in schema.lower()

    def test_semicolon_csv_detected_as_multiple_columns(self, temp_upload_dir):
        """European CSV files with semicolon delimiters should not collapse to one column."""
        csv_content = """Regione;Vendite;Prezzo
Lombardia;10;1,25
Lazio;20;3,50"""

        manager = DynamicDataManager(temp_upload_dir)
        fake_file = FakeUploadedFile("semicolon.csv", csv_content)
        db_path, table_name, metadata = manager.process_uploaded_file(fake_file)

        assert db_path.exists()
        assert table_name == "uploaded_data"
        assert metadata["detected_delimiter"] == ";"
        assert metadata["decimal_separator"] == ","
        assert metadata["columns"] == ["regione", "vendite", "prezzo"]

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"SELECT prezzo FROM {table_name} WHERE regione = 'Lombardia'")
        assert cursor.fetchone()[0] == 1.25
        conn.close()

    def test_duplicate_sanitized_columns_are_made_unique(self, temp_upload_dir):
        """Columns that sanitize to the same SQL name should not fail SQLite import."""
        csv_content = """A-B,A B
1,2
3,4"""

        manager = DynamicDataManager(temp_upload_dir)
        fake_file = FakeUploadedFile("duplicate_columns.csv", csv_content)
        db_path, table_name, metadata = manager.process_uploaded_file(fake_file)

        assert db_path.exists()
        assert metadata["columns"] == ["a_b", "a_b_2"]
        assert metadata["column_mapping"]["A-B"] == "a_b"
        assert metadata["column_mapping"]["A B"] == "a_b_2"
        assert metadata["warnings"]

    def test_empty_csv_is_rejected(self, temp_upload_dir):
        manager = DynamicDataManager(temp_upload_dir)
        fake_file = FakeUploadedFile("empty.csv", "")

        with pytest.raises(DatasetValidationError, match="vuoto"):
            manager.process_uploaded_file(fake_file)

    def test_upload_size_limit_is_enforced(self, temp_upload_dir):
        manager = DynamicDataManager(temp_upload_dir, max_upload_size_mb=0)
        fake_file = FakeUploadedFile("too_big.csv", "a,b\n1,2\n")

        with pytest.raises(DatasetValidationError, match="limite"):
            manager.process_uploaded_file(fake_file)

    def test_upload_filename_cannot_escape_upload_dir(self, temp_upload_dir):
        manager = DynamicDataManager(temp_upload_dir)
        fake_file = FakeUploadedFile("../escape.csv", "a,b\n1,2\n")

        db_path, _table_name, metadata = manager.process_uploaded_file(fake_file)

        assert db_path.parent.resolve() == temp_upload_dir.resolve()
        assert metadata["stored_filename"] == "escape.csv"

    def test_metadata_includes_dataset_profile(self, temp_upload_dir, sample_csv_content):
        manager = DynamicDataManager(temp_upload_dir)
        fake_file = FakeUploadedFile("profile.csv", sample_csv_content)

        _db_path, _table_name, metadata = manager.process_uploaded_file(fake_file)

        profile = metadata["profile"]
        assert profile["row_count"] == 18
        assert profile["column_count"] == 5
        assert "vendite" in profile["numeric_columns"]
        assert profile["columns"]["vendite"]["semantic_type"] == "numeric"
        assert profile["columns"]["vendite"]["null_count"] == 0
        assert profile["columns"]["regione"]["semantic_type"] == "categorical"
        assert "Profilo" not in metadata["profile_summary"]
        assert "Colonne numeriche" in metadata["profile_summary"]

    def test_profile_detects_common_dataset_roles(self, temp_upload_dir):
        csv_content = """Species,Latitude,Longitude,Height m,Plant Date,Score,Category
Acer platanoides,45.1,9.1,12.5,2024-01-02,5,A
Tilia cordata,45.2,9.2,10.0,2024-02-03,8,B"""

        manager = DynamicDataManager(temp_upload_dir)
        fake_file = FakeUploadedFile("roles.csv", csv_content)
        _db_path, _table_name, metadata = manager.process_uploaded_file(fake_file)

        profile = metadata["profile"]
        roles = profile["roles"]
        assert roles["latitude_candidates"] == ["latitude"]
        assert roles["longitude_candidates"] == ["longitude"]
        assert roles["species_candidates"] == ["species"]
        assert roles["height_candidates"] == ["height_m"]
        assert "plant_date" in profile["datetime_columns"]
        assert "score" in profile["numeric_columns"]
        assert "Ruoli rilevati" in metadata["profile_summary"]


class TestDatasetQueryToolWithCustomDB:
    """Test DatasetQueryTool with custom uploaded database."""
    
    def test_tool_with_custom_database(self, temp_upload_dir, sample_csv_content):
        """Test querying a custom uploaded database."""
        # Setup: Create custom database
        manager = DynamicDataManager(temp_upload_dir)
        fake_file = FakeUploadedFile("vendite.csv", sample_csv_content)
        db_path, table_name, _ = manager.process_uploaded_file(fake_file)
        
        # Create tool with custom database (without LLM for basic test)
        tool = DatasetQueryTool(
            db_path=db_path,
            table_name=table_name,
            user_description="Dataset di vendite per regione e prodotto"
        )
        
        # Verify tool configuration
        assert tool._db_path == db_path
        assert tool._table_name == table_name
        assert tool._user_description == "Dataset di vendite per regione e prodotto"
    
    def test_direct_sql_execution(self, temp_upload_dir, sample_csv_content):
        """Test direct SQL execution on custom database."""
        # Setup
        manager = DynamicDataManager(temp_upload_dir)
        fake_file = FakeUploadedFile("vendite.csv", sample_csv_content)
        db_path, table_name, _ = manager.process_uploaded_file(fake_file)
        
        tool = DatasetQueryTool(db_path=db_path, table_name=table_name)
        
        # Get connection
        conn = tool._get_connection()
        
        # Execute SQL directly
        result = tool._execute_sql(conn, f"SELECT COUNT(*) as total FROM {table_name}", "")
        
        conn.close()
        
        # Verify result
        assert "result" in result
        assert result["result"] == 18
        assert result["column"] == "total"
    
    def test_schema_retrieval(self, temp_upload_dir, sample_csv_content):
        """Test schema retrieval from custom database."""
        # Setup
        manager = DynamicDataManager(temp_upload_dir)
        fake_file = FakeUploadedFile("data.csv", sample_csv_content)
        db_path, table_name, _ = manager.process_uploaded_file(fake_file)
        
        tool = DatasetQueryTool(db_path=db_path, table_name=table_name)
        
        # Get connection and schema
        conn = tool._get_connection()
        schema = tool._get_schema_info(conn)
        conn.close()
        
        # Verify schema
        assert schema != "Schema not found"
        assert "CREATE TABLE" in schema
        assert table_name in schema


class TestIntegration:
    """Integration tests for the complete CSV upload flow."""
    
    def test_complete_flow(self, temp_upload_dir, sample_csv_content):
        """Test complete flow from CSV upload to query execution."""
        # Step 1: Upload and process CSV
        manager = DynamicDataManager(temp_upload_dir)
        fake_file = FakeUploadedFile("integration_test.csv", sample_csv_content)
        db_path, table_name, metadata = manager.process_uploaded_file(fake_file)
        
        # Verify upload
        assert db_path.exists()
        assert metadata["row_count"] == 18
        
        # Step 2: Create tool with custom database
        tool = DatasetQueryTool(
            db_path=db_path,
            table_name=table_name,
            user_description="Dataset vendite mensili"
        )
        
        # Step 3: Execute queries
        conn = tool._get_connection()
        
        # Query 1: Count total records
        result1 = tool._execute_sql(conn, f"SELECT COUNT(*) as total FROM {table_name}", "")
        assert result1["result"] == 18
        
        # Query 2: Sum vendite
        result2 = tool._execute_sql(conn, f"SELECT SUM(vendite) as totale FROM {table_name}", "")
        assert result2["result"] > 0
        
        # Query 3: Group by regione
        result3 = tool._execute_sql(
            conn, 
            f"SELECT regione, SUM(vendite) as totale FROM {table_name} GROUP BY regione ORDER BY totale DESC",
            ""
        )
        assert "results" in result3
        assert len(result3["results"]) == 3  # 3 regions
        
        conn.close()
    
    def test_multiple_uploads(self, temp_upload_dir):
        """Test handling multiple CSV uploads."""
        manager = DynamicDataManager(temp_upload_dir)
        
        # Upload first CSV
        csv1 = """Name,Age,City
Alice,30,Rome
Bob,25,Milan"""
        file1 = FakeUploadedFile("users.csv", csv1)
        db_path1, table_name1, meta1 = manager.process_uploaded_file(file1)
        
        # Upload second CSV
        csv2 = """Product,Price,Stock
Laptop,1000,50
Mouse,20,200"""
        file2 = FakeUploadedFile("products.csv", csv2)
        db_path2, table_name2, meta2 = manager.process_uploaded_file(file2)
        
        # Verify both databases exist
        assert db_path1.exists()
        assert db_path2.exists()
        assert db_path1 != db_path2  # Different databases
        
        # Verify metadata
        assert meta1["row_count"] == 2
        assert meta2["row_count"] == 2
        assert "name" in meta1["columns"]
        assert "product" in meta2["columns"]


def test_csv_with_special_characters(temp_upload_dir):
    """Test CSV with special characters and encoding."""
    csv_content = """Nome,Città,Prezzo
Mario,Milano,€100
Giuseppe,Torino,€200
François,Paris,€150"""
    
    manager = DynamicDataManager(temp_upload_dir)
    fake_file = FakeUploadedFile("special.csv", csv_content)
    
    db_path, table_name, metadata = manager.process_uploaded_file(fake_file)
    
    # Verify processing
    assert db_path.exists()
    assert metadata["row_count"] == 3
    
    # Check columns (sanitizer preserves accents, only converts spaces to _)
    assert "nome" in metadata["columns"]
    assert "città" in metadata["columns"]  # Accents are preserved
    assert "prezzo" in metadata["columns"]


def test_csv_with_numeric_types(temp_upload_dir):
    """Test CSV with different numeric types."""
    csv_content = """ID,Value,Decimal,Text
1,100,10.5,test1
2,200,20.7,test2
3,300,30.9,test3"""
    
    manager = DynamicDataManager(temp_upload_dir)
    fake_file = FakeUploadedFile("numeric.csv", csv_content)
    
    db_path, table_name, metadata = manager.process_uploaded_file(fake_file)
    
    # Verify data types in database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute(f"SELECT id, value, decimal, text FROM {table_name} LIMIT 1")
    row = cursor.fetchone()
    
    # id and value should be integers
    assert isinstance(row[0], int)
    assert isinstance(row[1], int)
    # decimal should be float
    assert isinstance(row[2], float)
    # text should be string
    assert isinstance(row[3], str)
    
    conn.close()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
