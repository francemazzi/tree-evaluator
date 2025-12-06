"""
Dynamic Data Manager Service

This module provides functionality to dynamically load CSV files,
convert them to SQLite databases, and make them available for the chat agent.
"""

from __future__ import annotations

import pandas as pd
import sqlite3
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import re


class DynamicDataManager:
    """
    Handles dynamic loading of CSV data into SQLite for the chat agent.
    
    This class provides methods to upload CSV files, automatically infer
    data types, and create SQLite databases that can be queried by the agent.
    """
    
    def __init__(self, upload_dir: Path | str = "temp_data"):
        """
        Initialize the Data Manager.
        
        Args:
            upload_dir: Directory where uploaded files and databases will be stored
        """
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    def _sanitize_column_name(self, column_name: str) -> str:
        """
        Sanitize column name for SQL compatibility.
        
        Args:
            column_name: Original column name
            
        Returns:
            Sanitized column name safe for SQL
        """
        # Replace spaces and special characters with underscores
        sanitized = re.sub(r'[^\w]', '_', column_name)
        # Remove consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        return sanitized.lower()
    
    def process_uploaded_file(
        self, 
        uploaded_file, 
        table_name: Optional[str] = None
    ) -> Tuple[Path, str, Dict[str, Any]]:
        """
        Processes an uploaded CSV file and converts it to SQLite database.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            table_name: Optional custom table name (default: 'uploaded_data')
            
        Returns:
            Tuple containing:
                - Path to the created SQLite database
                - Table name used in the database
                - Dictionary with metadata (row_count, columns, etc.)
        """
        # 1. Save CSV temporarily
        file_path = self.upload_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # 2. Load into DataFrame with automatic type detection
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            # Try with different encodings
            df = pd.read_csv(file_path, encoding='latin-1')
        
        # 3. Clean column names for SQL compatibility
        original_columns = df.columns.tolist()
        df.columns = [self._sanitize_column_name(c) for c in df.columns]
        
        # 4. Create metadata
        metadata = {
            "original_filename": uploaded_file.name,
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "original_columns": original_columns,
            "column_mapping": dict(zip(original_columns, df.columns)),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "sample_data": df.head(3).to_dict('records')
        }
        
        # 5. Create SQLite database
        db_name = file_path.stem + ".db"
        db_path = self.upload_dir / db_name
        
        # Use provided table name or default
        if table_name is None:
            table_name = "uploaded_data"
        else:
            table_name = self._sanitize_column_name(table_name)
        
        # 6. Write to SQLite
        conn = sqlite3.connect(db_path)
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.close()
        
        return db_path, table_name, metadata
    
    def get_schema_info(self, db_path: Path, table_name: str) -> str:
        """
        Get schema information from a SQLite database.
        
        Args:
            db_path: Path to the SQLite database
            table_name: Name of the table
            
        Returns:
            Schema information as a string
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'")
            schema = cursor.fetchone()
            return schema[0] if schema else "Schema not found"
        finally:
            conn.close()
    
    def cleanup_old_files(self, keep_latest: int = 5) -> None:
        """
        Clean up old uploaded files to save disk space.
        
        Args:
            keep_latest: Number of most recent files to keep
        """
        # Get all files sorted by modification time
        all_files = sorted(
            self.upload_dir.glob("*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        # Remove old files beyond the keep_latest limit
        for file_path in all_files[keep_latest:]:
            try:
                file_path.unlink()
            except Exception:
                pass  # Ignore errors during cleanup

