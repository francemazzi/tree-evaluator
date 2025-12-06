#!/usr/bin/env python
"""
Manual test script for CSV upload functionality.

This script performs a complete manual test of the CSV upload feature
without requiring pytest or Streamlit.

Usage:
    python test_csv_manual.py
"""

import sys
from pathlib import Path
from streamlit_app.services.data_manager import DynamicDataManager
from streamlit_app.tools.dataset_tool import DatasetQueryTool
import sqlite3


class FakeUploadedFile:
    """Mock Streamlit uploaded file for testing."""
    
    def __init__(self, name: str, content: str):
        self.name = name
        self._content = content
    
    def getbuffer(self):
        return self._content.encode('utf-8')


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def print_success(text: str):
    """Print success message."""
    print(f"✅ {text}")


def print_error(text: str):
    """Print error message."""
    print(f"❌ {text}")


def print_info(text: str):
    """Print info message."""
    print(f"ℹ️  {text}")


def test_step_1_load_csv():
    """Test Step 1: Load CSV file."""
    print_header("Test 1: Caricamento CSV")
    
    # Check if test CSV exists
    csv_path = Path("test_data/esempio_vendite.csv")
    if not csv_path.exists():
        print_error(f"File CSV non trovato: {csv_path}")
        print_info("Creazione file di test...")
        
        # Create test CSV
        csv_path.parent.mkdir(exist_ok=True)
        csv_content = """Regione,Mese,Anno,Vendite,Prodotto
Lombardia,Gennaio,2023,15000,Laptop
Lazio,Gennaio,2023,12000,Laptop
Toscana,Gennaio,2023,8500,Laptop
Lombardia,Febbraio,2023,16500,Laptop
Lazio,Febbraio,2023,13200,Laptop
Toscana,Febbraio,2023,9100,Laptop
Lombardia,Marzo,2023,18000,Laptop
Lazio,Marzo,2023,14500,Laptop
Toscana,Marzo,2023,10200,Laptop"""
        
        with open(csv_path, 'w') as f:
            f.write(csv_content)
        print_success("File CSV di test creato")
    
    # Read CSV
    with open(csv_path, 'r') as f:
        csv_content = f.read()
    
    print_info(f"CSV caricato: {csv_path}")
    print_info(f"Dimensione: {len(csv_content)} bytes")
    
    # Count lines
    lines = csv_content.strip().split('\n')
    print_info(f"Righe totali: {len(lines)} (header + {len(lines)-1} dati)")
    
    print_success("Step 1 completato: CSV caricato correttamente")
    return csv_content


def test_step_2_convert_to_sql(csv_content):
    """Test Step 2: Convert CSV to SQLite."""
    print_header("Test 2: Conversione CSV → SQLite")
    
    try:
        # Initialize manager
        manager = DynamicDataManager(Path("temp_data"))
        print_success("DynamicDataManager inizializzato")
        
        # Create fake uploaded file
        fake_file = FakeUploadedFile("esempio_vendite.csv", csv_content)
        
        # Process file
        print_info("Conversione CSV in database SQLite in corso...")
        db_path, table_name, metadata = manager.process_uploaded_file(fake_file)
        
        print_success(f"Database creato: {db_path}")
        print_success(f"Tabella: {table_name}")
        
        # Print metadata
        print(f"\n📊 Metadata:")
        print(f"   - File originale: {metadata['original_filename']}")
        print(f"   - Righe: {metadata['row_count']}")
        print(f"   - Colonne: {metadata['column_count']}")
        print(f"   - Nomi colonne: {', '.join(metadata['columns'])}")
        
        print(f"\n🔄 Mapping colonne (originale → SQL):")
        for orig, sql in metadata['column_mapping'].items():
            print(f"   - {orig} → {sql}")
        
        print(f"\n📝 Tipi di dati:")
        for col, dtype in metadata['dtypes'].items():
            print(f"   - {col}: {dtype}")
        
        print_success("Step 2 completato: Database SQLite creato")
        return db_path, table_name, metadata
        
    except Exception as e:
        print_error(f"Errore durante la conversione: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def test_step_3_query_database(db_path, table_name):
    """Test Step 3: Query the database."""
    print_header("Test 3: Query del Database")
    
    if not db_path or not db_path.exists():
        print_error("Database non trovato, impossibile eseguire query")
        return False
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        print_success("Connessione al database stabilita")
        
        # Query 1: Count rows
        print("\n🔍 Query 1: Conteggio righe totali")
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"   Risultato: {count} righe")
        print_success(f"Query 1 eseguita: {count} righe trovate")
        
        # Query 2: Total sales
        print("\n🔍 Query 2: Totale vendite")
        cursor.execute(f"SELECT SUM(vendite) FROM {table_name}")
        total = cursor.fetchone()[0]
        print(f"   Risultato: €{total:,}")
        print_success(f"Query 2 eseguita: Total vendite = €{total:,}")
        
        # Query 3: Sales by region
        print("\n🔍 Query 3: Vendite per regione")
        cursor.execute(f"""
            SELECT regione, SUM(vendite) as totale 
            FROM {table_name} 
            GROUP BY regione 
            ORDER BY totale DESC
        """)
        results = cursor.fetchall()
        print("   Risultati:")
        for regione, totale in results:
            print(f"      - {regione}: €{totale:,}")
        print_success(f"Query 3 eseguita: {len(results)} regioni trovate")
        
        # Query 4: Sales by product
        print("\n🔍 Query 4: Vendite per prodotto")
        cursor.execute(f"""
            SELECT prodotto, SUM(vendite) as totale 
            FROM {table_name} 
            GROUP BY prodotto 
            ORDER BY totale DESC
        """)
        results = cursor.fetchall()
        print("   Risultati:")
        for prodotto, totale in results:
            print(f"      - {prodotto}: €{totale:,}")
        print_success(f"Query 4 eseguita: {len(results)} prodotti trovati")
        
        # Query 5: Average sales by month
        print("\n🔍 Query 5: Media vendite per mese")
        cursor.execute(f"""
            SELECT mese, AVG(vendite) as media 
            FROM {table_name} 
            GROUP BY mese
        """)
        results = cursor.fetchall()
        print("   Risultati:")
        for mese, media in results:
            print(f"      - {mese}: €{media:,.2f}")
        print_success(f"Query 5 eseguita: {len(results)} mesi trovati")
        
        conn.close()
        print_success("Step 3 completato: Tutte le query eseguite correttamente")
        return True
        
    except Exception as e:
        print_error(f"Errore durante l'esecuzione delle query: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_step_4_tool_integration(db_path, table_name):
    """Test Step 4: Integration with DatasetQueryTool."""
    print_header("Test 4: Integrazione con DatasetQueryTool")
    
    if not db_path:
        print_error("Database non disponibile")
        return False
    
    try:
        # Create tool
        print_info("Inizializzazione DatasetQueryTool...")
        tool = DatasetQueryTool(
            db_path=db_path,
            table_name=table_name,
            user_description="Dataset di vendite mensili per regione e prodotto in Italia nel 2023"
        )
        print_success("DatasetQueryTool creato")
        
        # Verify configuration
        print(f"\n⚙️  Configurazione Tool:")
        print(f"   - Database: {tool._db_path}")
        print(f"   - Tabella: {tool._table_name}")
        print(f"   - Descrizione: {tool._user_description}")
        
        # Test connection
        print_info("\nTest connessione al database...")
        conn = tool._get_connection()
        print_success("Connessione stabilita")
        
        # Test schema retrieval
        print_info("Recupero schema database...")
        schema = tool._get_schema_info(conn)
        print("\n📋 Schema:")
        print(f"   {schema[:200]}...")
        print_success("Schema recuperato")
        
        # Test SQL execution
        print_info("\nTest esecuzione SQL diretta...")
        result = tool._execute_sql(conn, f"SELECT COUNT(*) as total FROM {table_name}", "")
        print(f"   Risultato: {result}")
        print_success(f"SQL eseguito: {result.get('result', 'N/A')} righe")
        
        conn.close()
        
        print_success("Step 4 completato: Tool integrato correttamente")
        return True
        
    except Exception as e:
        print_error(f"Errore durante l'integrazione del tool: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("\n" + "="*60)
    print("  TEST MANUALE - Funzionalità Caricamento CSV")
    print("="*60)
    
    # Step 1: Load CSV
    csv_content = test_step_1_load_csv()
    if not csv_content:
        print_error("Test fallito allo Step 1")
        sys.exit(1)
    
    # Step 2: Convert to SQL
    db_path, table_name, metadata = test_step_2_convert_to_sql(csv_content)
    if not db_path:
        print_error("Test fallito allo Step 2")
        sys.exit(1)
    
    # Step 3: Query database
    if not test_step_3_query_database(db_path, table_name):
        print_error("Test fallito allo Step 3")
        sys.exit(1)
    
    # Step 4: Tool integration
    if not test_step_4_tool_integration(db_path, table_name):
        print_error("Test fallito allo Step 4")
        sys.exit(1)
    
    # Final summary
    print_header("🎉 TUTTI I TEST COMPLETATI CON SUCCESSO!")
    
    print("📋 Riepilogo:")
    print(f"   ✅ Step 1: CSV caricato")
    print(f"   ✅ Step 2: Database SQLite creato")
    print(f"   ✅ Step 3: Query eseguite correttamente")
    print(f"   ✅ Step 4: Tool integrato e funzionante")
    
    print(f"\n📁 File generati:")
    print(f"   - Database: {db_path}")
    print(f"   - Tabella: {table_name}")
    print(f"   - Righe: {metadata['row_count'] if metadata else 'N/A'}")
    
    print("\n💡 Prossimi passi:")
    print("   1. Avvia Streamlit: streamlit run streamlit_app/app.py")
    print("   2. Vai su 'Gestione Dataset' nella sidebar")
    print("   3. Carica il file: test_data/esempio_vendite.csv")
    print("   4. Fai domande sui dati nella chat!")
    
    print("\n✨ La funzionalità di caricamento CSV è operativa!\n")


if __name__ == "__main__":
    main()

