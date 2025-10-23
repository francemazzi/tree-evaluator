# Quickstart: Chatbot con LangChain/LangGraph

## Setup veloce (5 minuti)

### 1. Installa dipendenze

```bash
# Attiva ambiente virtuale
source .venv/bin/activate  # macOS/Linux
# oppure: .\.venv\Scripts\Activate.ps1  # Windows

# Installa pacchetti
pip install -r requirements.txt
```

### 2. Configura OpenAI API Key

```bash
# Copia template
cp .env.example .env

# Modifica .env e inserisci la tua chiave:
# OPENAI_API_KEY=sk-your-actual-key-here
```

### 3. Avvia il chatbot

```bash
streamlit run streamlit_app/app.py
```

Apri il browser su `http://localhost:8501`

## Come usare il chatbot

### 1. Crea una nuova conversazione

Clicca su "➕ Nuova Chat" nella sidebar.

### 2. Fai domande sul dataset

**Esempi:**

- "Quanti alberi ci sono nel dataset?"
- "Mostrami le statistiche per distretto"
- "Quali sono le 5 specie più comuni?"
- "Quanti alberi Acer ci sono nel distretto 19?"
- "Dammi 3 alberi casuali piantati dopo il 2010"

### 3. Calcola CO2 per alberi specifici

**Esempi:**

- "Calcola il CO2 sequestrato da un albero con diametro 30 cm e altezza 15 metri"
- "Quanto carbonio stocca un tiglio con circonferenza tronco 94 cm (quindi diametro 30 cm) e altezza 12 m?"
- "Stima la biomassa di un acero con DBH 25 cm e altezza 10 m"

### 4. Stime ambientali

**Esempi:**

- "Calcola il volume di un albero con diametro 35 cm"
- "Stima carbonio e biomassa per diametro 40 cm e altezza 18 m"

## Funzionalità chiave

### Tool disponibili per l'agent

1. **calculate_co2_sequestration**: Calcola CO2, biomassa (AGB, BGB), carbonio
2. **calculate_environmental_estimates**: Volume, biomassa, carbonio con formule alternative
3. **query_tree_dataset**: Query sul dataset Vienna (filtra, aggrega, campiona)

### Dataset

Il chatbot ha accesso al file `dataset/BAUMKATOGD.csv` con ~230K alberi di Vienna:

- Distretto, specie, anno di piantagione
- Circonferenza tronco, altezza, diametro chioma
- Coordinate geografiche (lon/lat)

### Conversazioni multiple

- Sidebar mostra tutte le conversazioni
- Rinomina (✏️) o elimina (🗑️) conversazioni
- Storia persistente su SQLite (`data/chat_index.db`)

## Architettura

```
User → Streamlit UI → ChatService → TreeEvaluatorAgent (LangGraph)
                                          ↓
                        ┌─────────────────┼─────────────────┐
                        ↓                 ↓                 ↓
                   CO2Tool         EnvironmentTool    DatasetTool
                        ↓                 ↓                 ↓
              CO2CalculationService  EnvService    pandas/CSV
```

### Flusso conversazione

1. User invia messaggio
2. ChatService chiama TreeEvaluatorAgent
3. Agent (GPT-4o-mini) analizza richiesta e decide quale tool usare
4. Tool esegue calcolo/query
5. Agent sintetizza risposta in italiano
6. Risposta salvata su SQLite e mostrata in UI

## Troubleshooting

### "OpenAI API key not found"

→ Verifica che `.env` esista e contenga `OPENAI_API_KEY=sk-...`

### "Dataset not found"

→ Verifica che `dataset/BAUMKATOGD.csv` esista nella root del progetto

### Agent fallback to demo

→ Se l'agent fallisce, il chatbot usa una risposta di fallback. Controlla i log in console per l'errore specifico (es. rate limit OpenAI, API key invalida)

### Pandas/import errors

→ Reinstalla dipendenze: `pip install -r requirements.txt`

## Sviluppo

### Aggiungere un nuovo tool

1. Crea file in `streamlit_app/tools/my_tool.py`:

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class MyToolInput(BaseModel):
    param: str = Field(description="...")

class MyTool(BaseTool):
    name = "my_tool_name"
    description = "..."
    args_schema = MyToolInput

    def _run(self, param: str) -> dict:
        # Logica del tool
        return {"result": "..."}
```

2. Registra in `streamlit_app/agent.py`:

```python
from streamlit_app.tools.my_tool import MyTool

# Nel __init__ di TreeEvaluatorAgent:
self._tools = [
    CO2CalculationTool(),
    EnvironmentEstimationTool(),
    DatasetQueryTool(),
    MyTool(),  # ← aggiungi qui
]
```

3. (Opzionale) Aggiorna system prompt in `agent.py` per descrivere il tool

### Testare localmente

```bash
# Test interattivo Python
python
>>> from streamlit_app.agent import TreeEvaluatorAgent
>>> agent = TreeEvaluatorAgent()
>>> response = agent.chat("Quanti alberi ci sono?")
>>> print(response)
```

### Docker

```bash
# Build e run
docker compose up --build streamlit

# Visita http://localhost:8501
```

## Prossimi passi

- [ ] Aggiungere streaming delle risposte (LangGraph streaming)
- [ ] Tool per esportare CSV/GeoJSON con CO2 calcolato
- [ ] Visualizzazioni (grafici statistiche, mappe)
- [ ] Supporto upload CSV custom dall'UI
- [ ] RAG con documentazione scientifica (Chave et al., ecc.)
- [ ] Multi-language support (EN, IT, DE)

## Link utili

- [LangChain docs](https://python.langchain.com/)
- [LangGraph docs](https://langchain-ai.github.io/langgraph/)
- [Streamlit docs](https://docs.streamlit.io/)
- [OpenAI API](https://platform.openai.com/docs/)
