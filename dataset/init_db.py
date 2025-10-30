#!/usr/bin/env python3
"""
Script per inizializzare il database SQLite dai file SQL nella cartella dataset.
Crea il database e gli indici necessari per query veloci.
"""

import sqlite3
import sys
from pathlib import Path


def init_database(sql_file: Path, db_file: Path) -> None:
    """Inizializza il database SQLite da un file SQL.
    
    Args:
        sql_file: Path al file .sql
        db_file: Path dove creare il database .db
    """
    print(f"📊 Inizializzazione database da {sql_file.name}")
    
    # Controlla se il database esiste già
    if db_file.exists():
        print(f"⚠️  Database {db_file.name} già esistente")
        response = input("Vuoi sovrascriverlo? (y/n): ")
        if response.lower() != 'y':
            print("❌ Operazione annullata")
            return
        db_file.unlink()
        print("🗑️  Database precedente eliminato")
    
    # Leggi il file SQL
    print(f"📖 Lettura file SQL ({sql_file.stat().st_size / 1024 / 1024:.1f} MB)...")
    with open(sql_file, 'r', encoding='utf-8') as f:
        sql_script = f.read()
    
    # Crea database e esegui script
    print("💾 Creazione database SQLite...")
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    try:
        # Esegui lo script SQL
        cursor.executescript(sql_script)
        conn.commit()
        
        # Conta righe inserite
        cursor.execute("SELECT COUNT(*) FROM baumkatogd")
        count = cursor.fetchone()[0]
        print(f"✅ {count:,} righe inserite nella tabella 'baumkatogd'")
        
        # Crea indici per performance
        print("\n🔍 Creazione indici per ottimizzare le query...")
        
        indices = [
            ("idx_district", "district"),
            ("idx_genus_species", "genus_species"),
            ("idx_plant_year", "plant_year"),
            ("idx_tree_id", "tree_id"),
            ("idx_area_group", "area_group"),
        ]
        
        for idx_name, column in indices:
            print(f"  - Creazione indice su '{column}'...")
            cursor.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON baumkatogd({column})")
        
        conn.commit()
        print("✅ Indici creati con successo")
        
        # Statistiche database
        print("\n📊 Statistiche database:")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"  - Tabelle: {len(tables)}")
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = cursor.fetchall()
        print(f"  - Indici: {len(indexes)}")
        
        # Dimensione file
        db_size_mb = db_file.stat().st_size / 1024 / 1024
        print(f"  - Dimensione DB: {db_size_mb:.1f} MB")
        
        # Esempi di query veloci
        print("\n⚡ Test velocità query:")
        
        import time
        
        # Query 1: Count totale
        start = time.time()
        cursor.execute("SELECT COUNT(*) FROM baumkatogd")
        total = cursor.fetchone()[0]
        elapsed = (time.time() - start) * 1000
        print(f"  - Count totale: {total:,} righe in {elapsed:.2f}ms")
        
        # Query 2: Filtro per distretto
        start = time.time()
        cursor.execute("SELECT COUNT(*) FROM baumkatogd WHERE district = 19")
        district_count = cursor.fetchone()[0]
        elapsed = (time.time() - start) * 1000
        print(f"  - Count distretto 19: {district_count:,} alberi in {elapsed:.2f}ms")
        
        # Query 3: Group by specie
        start = time.time()
        cursor.execute("""
            SELECT genus_species, COUNT(*) as count 
            FROM baumkatogd 
            WHERE genus_species IS NOT NULL
            GROUP BY genus_species 
            ORDER BY count DESC 
            LIMIT 5
        """)
        top_species = cursor.fetchall()
        elapsed = (time.time() - start) * 1000
        print(f"  - Top 5 specie in {elapsed:.2f}ms:")
        for species, count in top_species:
            print(f"    • {species}: {count:,}")
        
        print(f"\n✅ Database {db_file.name} creato con successo!")
        
    except sqlite3.Error as e:
        print(f"❌ Errore durante la creazione del database: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def main():
    """Main function."""
    dataset_dir = Path(__file__).parent
    
    print("=" * 60)
    print("🌳 Tree Evaluator - Database Initialization")
    print("=" * 60)
    
    # Trova tutti i file .sql nella cartella dataset
    sql_files = list(dataset_dir.glob("*.sql"))
    
    if not sql_files:
        print("❌ Nessun file .sql trovato nella cartella dataset/")
        sys.exit(1)
    
    print(f"\n📁 Trovati {len(sql_files)} file SQL:")
    for sql_file in sql_files:
        print(f"  - {sql_file.name}")
    
    # Processa ogni file SQL
    for sql_file in sql_files:
        db_file = sql_file.with_suffix('.db')
        print(f"\n{'=' * 60}")
        
        try:
            init_database(sql_file, db_file)
        except Exception as e:
            print(f"❌ Errore: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'=' * 60}")
    print("✅ Inizializzazione completata!")
    print("\nPer usare il database nell'app:")
    print("  1. Riavvia l'applicazione Streamlit")
    print("  2. Le query useranno automaticamente il database SQLite")
    print("=" * 60)


if __name__ == "__main__":
    main()

