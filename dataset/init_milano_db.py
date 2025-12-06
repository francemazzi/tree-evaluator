#!/usr/bin/env python
"""
Initialize SQLite database for Milan Trees dataset.

This script converts the dataset_milano.csv file into a SQLite database
optimized for querying by the Tree Evaluator agent.
"""

import pandas as pd
import sqlite3
from pathlib import Path


def create_milan_database():
    """Create SQLite database from Milan CSV dataset."""
    
    csv_path = Path(__file__).parent / "dataset_milano.csv"
    db_path = Path(__file__).parent / "dataset_milano.db"
    
    print(f"📂 Reading CSV: {csv_path}")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    print(f"📊 Rows: {len(df):,}")
    print(f"📋 Columns: {list(df.columns)}")
    
    # Clean column names (lowercase, no special chars)
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    
    # Extract year from data_ini for easier querying
    df['plant_year'] = pd.to_datetime(df['data_ini'], errors='coerce').dt.year
    
    # Create genus_species combined column for compatibility with Vienna dataset
    df['genus_species'] = df['genere'] + ' ' + df['specie']
    
    # Rename columns for consistency
    df = df.rename(columns={
        'municipio': 'district',
        'diam_tronc': 'trunk_diameter_cm',
        'diam_chiom': 'crown_diameter_m',
        'h_m': 'height_m',
        'localita': 'street',
        'long_x_4326': 'longitude',
        'lat_y_4326': 'latitude'
    })
    
    print(f"\n🔄 Processed columns: {list(df.columns)}")
    
    # Create database
    print(f"\n💾 Creating database: {db_path}")
    
    conn = sqlite3.connect(db_path)
    df.to_sql('milano_trees', conn, if_exists='replace', index=False)
    
    # Create indexes for faster queries
    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_district ON milano_trees(district)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_genus ON milano_trees(genere)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_species ON milano_trees(specie)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_plant_year ON milano_trees(plant_year)")
    
    conn.commit()
    
    # Verify
    cursor.execute("SELECT COUNT(*) FROM milano_trees")
    count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT district) FROM milano_trees")
    districts = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT genus_species) FROM milano_trees")
    species = cursor.fetchone()[0]
    
    conn.close()
    
    print(f"\n✅ Database created successfully!")
    print(f"   - Total trees: {count:,}")
    print(f"   - Districts (Municipi): {districts}")
    print(f"   - Unique species: {species}")
    print(f"   - File: {db_path}")
    
    return db_path


if __name__ == "__main__":
    create_milan_database()

