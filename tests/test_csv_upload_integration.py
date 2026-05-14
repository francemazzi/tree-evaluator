import sqlite3

from streamlit_app.services.data_manager import DynamicDataManager
from streamlit_app.tools.dataset_tool import DatasetQueryTool
from tests.csv_upload_fixtures import FakeUploadedFile, sample_csv_content, temp_upload_dir


class TestIntegration:
    """Integration tests for the complete CSV upload flow."""

    def test_complete_flow(self, temp_upload_dir, sample_csv_content):
        """Test complete flow from CSV upload to query execution."""
        manager = DynamicDataManager(temp_upload_dir)
        fake_file = FakeUploadedFile("integration_test.csv", sample_csv_content)
        db_path, table_name, metadata = manager.process_uploaded_file(fake_file)

        assert db_path.exists()
        assert metadata["row_count"] == 18

        tool = DatasetQueryTool(
            db_path=db_path,
            table_name=table_name,
            user_description="Dataset vendite mensili",
        )

        conn = tool._get_connection()
        result1 = tool._execute_sql(conn, f"SELECT COUNT(*) as total FROM {table_name}", "")
        assert result1["result"] == 18

        result2 = tool._execute_sql(conn, f"SELECT SUM(vendite) as totale FROM {table_name}", "")
        assert result2["result"] > 0

        result3 = tool._execute_sql(
            conn,
            f"SELECT regione, SUM(vendite) as totale FROM {table_name} GROUP BY regione ORDER BY totale DESC",
            "",
        )
        assert "results" in result3
        assert len(result3["results"]) == 3
        conn.close()

    def test_multiple_uploads(self, temp_upload_dir):
        """Test handling multiple CSV uploads."""
        manager = DynamicDataManager(temp_upload_dir)

        csv1 = """Name,Age,City
Alice,30,Rome
Bob,25,Milan"""
        db_path1, _table_name1, meta1 = manager.process_uploaded_file(FakeUploadedFile("users.csv", csv1))

        csv2 = """Product,Price,Stock
Laptop,1000,50
Mouse,20,200"""
        db_path2, _table_name2, meta2 = manager.process_uploaded_file(FakeUploadedFile("products.csv", csv2))

        assert db_path1.exists()
        assert db_path2.exists()
        assert db_path1 != db_path2
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
    db_path, _table_name, metadata = manager.process_uploaded_file(FakeUploadedFile("special.csv", csv_content))

    assert db_path.exists()
    assert metadata["row_count"] == 3
    assert "nome" in metadata["columns"]
    assert "città" in metadata["columns"]
    assert "prezzo" in metadata["columns"]


def test_csv_with_numeric_types(temp_upload_dir):
    """Test CSV with different numeric types."""
    csv_content = """ID,Value,Decimal,Text
1,100,10.5,test1
2,200,20.7,test2
3,300,30.9,test3"""

    manager = DynamicDataManager(temp_upload_dir)
    db_path, table_name, _metadata = manager.process_uploaded_file(FakeUploadedFile("numeric.csv", csv_content))

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"SELECT id, value, decimal, text FROM {table_name} LIMIT 1")
    row = cursor.fetchone()

    assert isinstance(row[0], int)
    assert isinstance(row[1], int)
    assert isinstance(row[2], float)
    assert isinstance(row[3], str)
    conn.close()
