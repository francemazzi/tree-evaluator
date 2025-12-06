# 🧪 Test per Funzionalità Caricamento CSV

## Panoramica

Questa suite di test verifica completamente la funzionalità di caricamento CSV dinamico implementata nel sistema Tree Evaluator.

## Test Disponibili

### 1. Test Automatici (pytest)

**File**: `tests/test_csv_upload.py`

Suite completa di 12 test automatici che verificano:

#### TestDynamicDataManager
- ✅ `test_initialization`: Verifica inizializzazione del manager
- ✅ `test_sanitize_column_name`: Testa pulizia nomi colonne
- ✅ `test_process_uploaded_file`: Verifica pipeline completa CSV → SQLite
- ✅ `test_database_content`: Controlla dati nel database
- ✅ `test_get_schema_info`: Verifica recupero schema

#### TestDatasetQueryToolWithCustomDB
- ✅ `test_tool_with_custom_database`: Testa configurazione tool con DB personalizzato
- ✅ `test_direct_sql_execution`: Verifica esecuzione SQL diretta
- ✅ `test_schema_retrieval`: Testa recupero schema tramite tool

#### TestIntegration
- ✅ `test_complete_flow`: Verifica flusso completo end-to-end
- ✅ `test_multiple_uploads`: Testa caricamento multiplo CSV

#### Test Speciali
- ✅ `test_csv_with_special_characters`: Testa caratteri speciali e accenti
- ✅ `test_csv_with_numeric_types`: Verifica inferenza tipi numerici

**Esecuzione**:
```bash
# Tutti i test
pytest tests/test_csv_upload.py -v

# Test specifico
pytest tests/test_csv_upload.py::TestDynamicDataManager::test_initialization -v

# Con output dettagliato
pytest tests/test_csv_upload.py -vv

# Con coverage
pytest tests/test_csv_upload.py --cov=streamlit_app.services.data_manager --cov-report=html
```

### 2. Test Manuale Interattivo

**File**: `test_csv_manual.py`

Script Python standalone che esegue un test completo con output formattato.

**Cosa testa**:
1. ✅ Caricamento CSV
2. ✅ Conversione CSV → SQLite
3. ✅ Esecuzione query SQL (5 query diverse)
4. ✅ Integrazione con DatasetQueryTool

**Esecuzione**:
```bash
python test_csv_manual.py
```

**Output**:
```
============================================================
  TEST MANUALE - Funzionalità Caricamento CSV
============================================================

✅ Step 1: CSV caricato correttamente
✅ Step 2: Database SQLite creato
✅ Step 3: Tutte le query eseguite correttamente
✅ Step 4: Tool integrato e funzionante

🎉 TUTTI I TEST COMPLETATI CON SUCCESSO!
```

### 3. File CSV di Test

**File**: `test_data/esempio_vendite.csv`

Dataset di esempio per test manuali in Streamlit:
- 18 righe di dati
- 5 colonne (Regione, Mese, Anno, Vendite, Prodotto)
- Dati di vendite mensili per 3 regioni italiane

**Struttura**:
```csv
Regione,Mese,Anno,Vendite,Prodotto
Lombardia,Gennaio,2023,15000,Laptop
Lazio,Gennaio,2023,12000,Laptop
...
```

## Esecuzione Rapida

### Test Automatici
```bash
# Install pytest se necessario
pip install pytest pytest-cov

# Esegui i test
pytest tests/test_csv_upload.py -v
```

### Test Manuale
```bash
# Esegui lo script
python test_csv_manual.py
```

### Test in Streamlit
```bash
# 1. Avvia Streamlit
streamlit run streamlit_app/app.py

# 2. Nella UI:
#    - Vai su "Gestione Dataset" nella sidebar
#    - Seleziona "Carica CSV Personalizzato"
#    - Carica: test_data/esempio_vendite.csv
#    - Descrizione: "Dataset vendite mensili Italia 2023"

# 3. Fai domande nella chat:
#    - "Quante righe ci sono nel dataset?"
#    - "Qual è il totale delle vendite?"
#    - "Mostrami le vendite per regione"
#    - "Crea un grafico delle vendite per prodotto"
```

## Risultati Attesi

### Test Automatici (pytest)

```
tests/test_csv_upload.py::TestDynamicDataManager::test_initialization PASSED
tests/test_csv_upload.py::TestDynamicDataManager::test_sanitize_column_name PASSED
tests/test_csv_upload.py::TestDynamicDataManager::test_process_uploaded_file PASSED
tests/test_csv_upload.py::TestDynamicDataManager::test_database_content PASSED
tests/test_csv_upload.py::TestDynamicDataManager::test_get_schema_info PASSED
tests/test_csv_upload.py::TestDatasetQueryToolWithCustomDB::test_tool_with_custom_database PASSED
tests/test_csv_upload.py::TestDatasetQueryToolWithCustomDB::test_direct_sql_execution PASSED
tests/test_csv_upload.py::TestDatasetQueryToolWithCustomDB::test_schema_retrieval PASSED
tests/test_csv_upload.py::TestIntegration::test_complete_flow PASSED
tests/test_csv_upload.py::TestIntegration::test_multiple_uploads PASSED
tests/test_csv_upload.py::test_csv_with_special_characters PASSED
tests/test_csv_upload.py::test_csv_with_numeric_types PASSED

============================== 12 passed in 1.00s ==============================
```

### Test Manuale

```
✅ Step 1 completato: CSV caricato correttamente
✅ Step 2 completato: Database SQLite creato

📊 Metadata:
   - Righe: 18
   - Colonne: 5

🔍 Query 1: Conteggio righe totali → 18 righe
🔍 Query 2: Totale vendite → €177,800
🔍 Query 3: Vendite per regione → 3 regioni
🔍 Query 4: Vendite per prodotto → 2 prodotti
🔍 Query 5: Media vendite per mese → 3 mesi

✅ Step 3 completato: Tutte le query eseguite correttamente
✅ Step 4 completato: Tool integrato e funzionante

🎉 TUTTI I TEST COMPLETATI CON SUCCESSO!
```

## Cosa Viene Testato

### 1. DynamicDataManager
- [x] Creazione directory upload
- [x] Sanitizzazione nomi colonne
- [x] Lettura file CSV
- [x] Inferenza tipi di dati
- [x] Creazione database SQLite
- [x] Generazione metadata
- [x] Gestione caratteri speciali
- [x] Gestione accenti
- [x] Mapping colonne originali → SQL

### 2. Conversione CSV → SQLite
- [x] Struttura database corretta
- [x] Dati inseriti correttamente
- [x] Tipi di dati appropriati (INTEGER, REAL, TEXT)
- [x] Numero righe corretto
- [x] Valori specifici corrispondono

### 3. DatasetQueryTool
- [x] Configurazione con DB personalizzato
- [x] Connessione al database
- [x] Recupero schema
- [x] Esecuzione SQL diretta
- [x] Formattazione risultati

### 4. Integrazione Completa
- [x] Pipeline end-to-end
- [x] Query multiple su stesso DB
- [x] Caricamenti multipli CSV
- [x] Isolamento tra database diversi

## Struttura File di Test

```
tree-evaluator/
├── tests/
│   └── test_csv_upload.py          # Test suite pytest (12 test)
├── test_data/
│   └── esempio_vendite.csv         # CSV di esempio per test manuali
├── test_csv_manual.py              # Script test manuale standalone
└── temp_data/                      # Directory database temporanei
    └── *.db                        # Database SQLite generati dai test
```

## Debug e Troubleshooting

### Problema: Test falliscono con "FileNotFoundError"

**Soluzione**: Verifica che le dipendenze siano installate:
```bash
pip install -r requirements.txt
```

### Problema: "No module named 'pytest'"

**Soluzione**: Installa pytest:
```bash
pip install pytest
```

### Problema: Database non viene creato

**Soluzione**: Verifica permessi sulla directory `temp_data`:
```bash
mkdir -p temp_data
chmod 755 temp_data
```

### Problema: Test charset/encoding

**Soluzione**: Verifica che il CSV sia in UTF-8:
```bash
file -I test_data/esempio_vendite.csv
# Output dovrebbe essere: charset=utf-8
```

## Continuous Integration (CI)

Per integrare questi test in una pipeline CI/CD:

```yaml
# .github/workflows/test.yml
name: Test CSV Upload

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: |
          pytest tests/test_csv_upload.py -v --cov=streamlit_app.services.data_manager
```

## Metriche

### Coverage

Per generare un report di coverage:
```bash
pytest tests/test_csv_upload.py --cov=streamlit_app.services.data_manager --cov-report=html
open htmlcov/index.html
```

### Performance

I test sono progettati per essere veloci:
- Tempo medio esecuzione: ~1 secondo per tutti i 12 test
- Usa directory temporanee che vengono pulite automaticamente
- Nessuna dipendenza da servizi esterni

## Best Practices

1. **Esegui i test prima di fare commit**:
   ```bash
   pytest tests/test_csv_upload.py -v
   ```

2. **Aggiungi nuovi test quando modifichi il codice**:
   - Se aggiungi funzionalità → aggiungi test
   - Se trovi bug → aggiungi test di regressione

3. **Usa fixture per dati comuni**:
   ```python
   @pytest.fixture
   def sample_csv():
       return "Col1,Col2\nval1,val2"
   ```

4. **Testa casi edge**:
   - CSV vuoti
   - CSV con solo header
   - Colonne con nomi duplicati
   - Valori NULL/NaN

## Prossimi Test da Aggiungere

- [ ] Test con CSV molto grandi (>1GB)
- [ ] Test con encoding diversi (latin-1, windows-1252)
- [ ] Test con delimitatori diversi (;, \t)
- [ ] Test con quote characters
- [ ] Test performance caricamento
- [ ] Test cleanup automatico file vecchi
- [ ] Test limite memoria
- [ ] Test concorrenza (upload multipli simultanei)

---

**Versione**: 1.0  
**Data**: Dicembre 2024  
**Autore**: Tree Evaluator Team

