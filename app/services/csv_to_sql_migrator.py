"""
CSV to SQL Migrator Service

This module provides functionality to migrate CSV datasets to SQL format,
maintaining the same columns and titles.
"""

import csv
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import re


class CSVToSQLMigrator:
    """
    Handles the migration of CSV files to SQL format.
    
    This class reads CSV files from a specified directory and generates
    SQL files with CREATE TABLE and INSERT statements.
    """
    
    def __init__(self, dataset_dir: str = "dataset"):
        """
        Initialize the migrator.
        
        Args:
            dataset_dir: Directory containing CSV files
        """
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.exists():
            raise ValueError(f"Dataset directory '{dataset_dir}' does not exist")
    
    def get_csv_files(self) -> List[Path]:
        """
        Get all CSV files in the dataset directory.
        
        Returns:
            List of Path objects for CSV files
        """
        return list(self.dataset_dir.glob("*.csv"))
    
    def sanitize_column_name(self, column_name: str) -> str:
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
    
    def infer_sql_type(self, values: List[str]) -> str:
        """
        Infer SQL data type from a sample of values.
        
        Args:
            values: List of sample values from the column
            
        Returns:
            SQL data type string
        """
        # Remove empty values for analysis
        non_empty_values = [v for v in values if v and v.strip()]
        
        if not non_empty_values:
            return "TEXT"
        
        # Check if all values are integers
        all_integers = True
        all_floats = True
        max_length = 0
        
        for value in non_empty_values[:100]:  # Sample first 100 non-empty values
            value = value.strip()
            max_length = max(max_length, len(value))
            
            try:
                int(value)
            except ValueError:
                all_integers = False
            
            try:
                float(value)
            except ValueError:
                all_floats = False
        
        if all_integers:
            return "INTEGER"
        elif all_floats:
            return "REAL"
        else:
            # Use VARCHAR with appropriate length
            varchar_length = min(max(max_length * 2, 50), 1000)
            return f"VARCHAR({varchar_length})"
    
    def escape_sql_string(self, value: str) -> str:
        """
        Escape string value for SQL INSERT statement.
        
        Args:
            value: String value to escape
            
        Returns:
            Escaped string value
        """
        if value is None or value == '':
            return 'NULL'
        
        # Escape single quotes by doubling them
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    
    def create_table_statement(self, table_name: str, columns: List[str], 
                              column_types: Dict[str, str]) -> str:
        """
        Generate CREATE TABLE SQL statement.
        
        Args:
            table_name: Name of the table
            columns: List of column names
            column_types: Dictionary mapping column names to SQL types
            
        Returns:
            CREATE TABLE SQL statement
        """
        sanitized_table = self.sanitize_column_name(table_name)
        
        column_defs = []
        for col in columns:
            sanitized_col = self.sanitize_column_name(col)
            col_type = column_types.get(col, "TEXT")
            column_defs.append(f"    {sanitized_col} {col_type}")
        
        sql = f"-- Table: {sanitized_table}\n"
        sql += f"DROP TABLE IF EXISTS {sanitized_table};\n\n"
        sql += f"CREATE TABLE {sanitized_table} (\n"
        sql += ",\n".join(column_defs)
        sql += "\n);\n\n"
        
        return sql
    
    def create_insert_statements(self, table_name: str, columns: List[str], 
                                 rows: List[Dict[str, str]], 
                                 batch_size: int = 100) -> str:
        """
        Generate INSERT SQL statements.
        
        Args:
            table_name: Name of the table
            columns: List of column names
            rows: List of row dictionaries
            batch_size: Number of rows per INSERT statement
            
        Returns:
            INSERT SQL statements
        """
        sanitized_table = self.sanitize_column_name(table_name)
        sanitized_columns = [self.sanitize_column_name(col) for col in columns]
        
        sql = f"-- Insert data into {sanitized_table}\n"
        
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            
            sql += f"INSERT INTO {sanitized_table} ({', '.join(sanitized_columns)}) VALUES\n"
            
            value_rows = []
            for row in batch:
                values = []
                for col in columns:
                    value = row.get(col, '')
                    if value == '' or value is None:
                        values.append('NULL')
                    else:
                        values.append(self.escape_sql_string(value))
                value_rows.append(f"    ({', '.join(values)})")
            
            sql += ",\n".join(value_rows)
            sql += ";\n\n"
        
        return sql
    
    def migrate_csv_to_sql(self, csv_path: Path, force_update: bool = False) -> Optional[Path]:
        """
        Migrate a single CSV file to SQL format.
        
        Args:
            csv_path: Path to the CSV file
            force_update: If True, update existing SQL file
            
        Returns:
            Path to the generated SQL file, or None if skipped
        """
        sql_path = csv_path.with_suffix('.sql')
        
        # Check if SQL file already exists
        if sql_path.exists() and not force_update:
            print(f"SQL file already exists: {sql_path.name}")
            return None
        
        print(f"Processing: {csv_path.name}")
        
        # Read CSV file
        rows = []
        columns = []
        column_values = {}
        
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            columns = reader.fieldnames
            
            # Initialize column value collectors
            for col in columns:
                column_values[col] = []
            
            # Read all rows and collect sample values
            for idx, row in enumerate(reader):
                rows.append(row)
                
                # Collect sample values for type inference (first 1000 rows)
                if idx < 1000:
                    for col in columns:
                        column_values[col].append(row.get(col, ''))
        
        print(f"  - Found {len(rows)} rows with {len(columns)} columns")
        
        # Infer column types
        column_types = {}
        for col in columns:
            column_types[col] = self.infer_sql_type(column_values[col])
        
        # Generate SQL
        table_name = csv_path.stem
        sql_content = "-- Generated SQL from CSV\n"
        sql_content += f"-- Source: {csv_path.name}\n"
        sql_content += f"-- Generated on: {os.popen('date').read().strip()}\n\n"
        
        sql_content += self.create_table_statement(table_name, columns, column_types)
        sql_content += self.create_insert_statements(table_name, columns, rows)
        
        # Write SQL file
        with open(sql_path, 'w', encoding='utf-8') as sqlfile:
            sqlfile.write(sql_content)
        
        print(f"  - Generated: {sql_path.name}")
        return sql_path
    
    def migrate_all(self, force_update: bool = False) -> List[Path]:
        """
        Migrate all CSV files in the dataset directory.
        
        Args:
            force_update: If True, update existing SQL files
            
        Returns:
            List of paths to generated SQL files
        """
        csv_files = self.get_csv_files()
        
        if not csv_files:
            print("No CSV files found in dataset directory")
            return []
        
        print(f"Found {len(csv_files)} CSV file(s) to migrate")
        print(f"Update mode: {'ON' if force_update else 'OFF'}\n")
        
        sql_files = []
        for csv_file in csv_files:
            try:
                sql_file = self.migrate_csv_to_sql(csv_file, force_update)
                if sql_file:
                    sql_files.append(sql_file)
            except Exception as e:
                print(f"Error processing {csv_file.name}: {str(e)}")
        
        print(f"\nMigration complete! Generated {len(sql_files)} SQL file(s)")
        return sql_files

