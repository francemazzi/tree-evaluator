# 🔧 Troubleshooting - Tree Evaluator

## ❌ "L'agent non risponde / Usa solo risposte demo"

### Sintomo

Streamlit mostra:

```
Echo (00:05:54): ciao [demo]
```

### Cause possibili e soluzioni:

#### 1. ⚠️ API Key non inserita o non valida

**Verifica:**

- Apri Streamlit → Sidebar → Settings
- Campo "OpenAI API Key" è vuoto o non inizia con `sk-`?

**Soluzione:**

1. Vai su https://platform.openai.com/api-keys
2. Genera nuova chiave (o copia esistente)
3. Incolla nel campo "OpenAI API Key"
4. Dovrebbe apparire: "✅ Chiave API aggiornata!"
5. Crea nuova conversazione
6. Invia messaggio

#### 2. ❌ Errore import dipendenze

**Verifica:**
Streamlit mostra: "❌ Errore import dipendenze: No module named 'langchain'"

**Soluzione:**

```bash
# Attiva virtual environment
source .venv/bin/activate  # macOS/Linux
# oppure .\.venv\Scripts\Activate.ps1  # Windows

# Installa dipendenze
pip install -r requirements.txt

# Verifica installazione
pip list | grep langchain
pip list | grep pandas
pip list | grep openai

# Riavvia Streamlit
streamlit run streamlit_app/app.py
```

#### 3. 🔑 API key non valida o scaduta

**Verifica:**
Streamlit mostra: "❌ Chiave API non valida: ..."

**Soluzione:**

1. Controlla su https://platform.openai.com/api-keys
2. Verifica che la chiave sia attiva
3. Se necessario, genera nuova chiave
4. Aggiorna in Streamlit Settings

#### 4. 💳 Credito OpenAI esaurito

**Verifica:**
Errore: "insufficient_quota" o "rate_limit_exceeded"

**Soluzione:**

1. Vai su https://platform.openai.com/account/billing
2. Controlla credito disponibile
3. Aggiungi metodo di pagamento se necessario
4. Free tier: $5 di credito gratuito per nuovi account

#### 5. 📁 Dataset non trovato

**Verifica:**
Quando chiedi "Quanti alberi?", risponde "Dataset not found"

**Soluzione:**

```bash
# Verifica presenza file
ls -lh dataset/BAUMKATOGD.csv

# Se manca, controlla che sia nella posizione corretta
# Il file dovrebbe essere in:
# /Users/francesco/Sviluppo/frasma_studio/tree-evaluator/dataset/BAUMKATOGD.csv
```

#### 6. 🐛 Errore Python generico

**Verifica:**
Console/terminale mostra traceback Python

**Soluzione:**

```bash
# Test agent standalone (più facile per debug)
python test_agent.py

# O con chiave esplicita:
python test_agent.py sk-your-key-here

# Questo testerà:
# - Dipendenze installate?
# - Dataset accessibile?
# - Agent si inizializza?
# - Query funzionano?
# - Streaming funziona?
```

## 🧪 Test rapido completo

```bash
# 1. Verifica environment
python --version  # Deve essere 3.11+

# 2. Verifica virtual env attivo
which python  # Deve puntare a .venv/bin/python

# 3. Test agent
python test_agent.py sk-your-actual-key

# 4. Se test OK, prova Streamlit
streamlit run streamlit_app/app.py
```

## 📋 Checklist debug

- [ ] Virtual environment attivato?
- [ ] Dipendenze installate? (`pip list | grep langchain`)
- [ ] File `.env` esiste? (`cat .env | grep OPENAI`)
- [ ] Chiave API inserita in UI Settings?
- [ ] Dataset presente? (`ls dataset/BAUMKATOGD.csv`)
- [ ] Port 8501 libera? (`lsof -i :8501`)
- [ ] Test standalone passa? (`python test_agent.py`)

## 🔍 Debug avanzato

### Logs dettagliati

Streamlit nasconde alcuni log. Per vederli tutti:

```bash
# Avvia con logs verbose
streamlit run streamlit_app/app.py --logger.level=debug

# O controlla console dove hai avviato streamlit
# Gli errori vengono stampati lì con traceback completo
```

### Test tool singoli

```python
# Test CO2 tool
from streamlit_app.tools.co2_tool import CO2CalculationTool
tool = CO2CalculationTool()
result = tool._run(dbh_cm=30, height_m=15, wood_density_g_cm3=0.6)
print(result)

# Test Dataset tool
from streamlit_app.tools.dataset_tool import DatasetQueryTool
tool = DatasetQueryTool()
result = tool._run(query_type="summary")
print(result)

# Test Environment tool
from streamlit_app.tools.environment_tool import EnvironmentEstimationTool
tool = EnvironmentEstimationTool()
result = tool._run(diameter_cm=30, height_m=15)
print(result)
```

### Verifica connessione OpenAI

```python
from openai import OpenAI
client = OpenAI(api_key="sk-your-key")
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Test"}]
)
print(response.choices[0].message.content)
```

## 🐳 Docker troubleshooting

### Agent non funziona in container

```bash
# Verifica env vars
docker compose exec streamlit env | grep OPENAI

# Controlla logs
docker compose logs streamlit | tail -50

# Shell interattiva
docker compose exec streamlit /bin/bash

# Dentro container, testa:
python test_agent.py $OPENAI_API_KEY

# Verifica dataset montato
ls -la /app/dataset/
```

### Rebuild immagine

```bash
# Se modifiche a requirements.txt o Dockerfile
docker compose build --no-cache streamlit
docker compose up streamlit
```

## 💡 Problemi comuni e fix veloci

### "Agent si inizializza ma non risponde"

→ Probabilmente l'agent NON si inizializza, controlla sidebar per messaggi di errore

### "Risponde in inglese invece che italiano"

→ L'agent risponde in italiano. Se risponde in inglese, sta usando il fallback demo (agent non inizializzato)

### "Streaming non funziona, risposta appare tutta insieme"

→ Questo è normale se agent non inizializzato (fallback è non-streaming). Con agent vero, vedrai cursore `▌`

### "Errore 'NoneType' object has no attribute 'chat'"

→ Agent è None, inizializzazione fallita. Controlla console per errore specifico

### "st.spinner non appare"

→ Normale, spinner appare solo primo init. Se vuoi sempre vederlo, resetta agent:

```python
# In Streamlit, change API key to force reinit
```

## 📞 Ancora problemi?

1. **Controlla console** dove hai avviato `streamlit run`
2. **Leggi traceback completo** - indica esattamente quale riga fallisce
3. **Run test_agent.py** - isola il problema dall'UI
4. **Verifica requirements.txt** - versioni compatibili?
5. **Reinstalla da zero**:
   ```bash
   rm -rf .venv
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## ✅ Verifica funzionante

Quando tutto funziona, vedrai:

1. **Inserisci chiave in Settings** → "✅ Chiave API aggiornata!"
2. **Primo messaggio** → "🤖 Inizializzazione agent LangGraph..." → "✅ Agent inizializzato correttamente!"
3. **Risposte** → Testo con cursore `▌` che si aggiorna in tempo reale
4. **Contenuto** → Risposte intelligenti in italiano, non "Echo (timestamp)"
5. **Tools** → Quando chiedi del dataset, vedi dati reali dal CSV

## 📝 Report bug

Se nulla funziona, raccogli queste info:

```bash
# Sistema
uname -a
python --version

# Dipendenze
pip freeze > installed_packages.txt

# Test
python test_agent.py &> test_output.txt

# Logs Streamlit
streamlit run streamlit_app/app.py &> streamlit_logs.txt
```

Allega questi file quando riporti il problema.
