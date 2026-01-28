## Tree Evaluator API

Minimal FastAPI-based API (with OOP design) to estimate CO2 sequestered by trees from dendrometric parameters. It provides an endpoint that computes above-ground biomass (AGB), below-ground biomass (BGB), total biomass, carbon, and CO2, with optional estimation of annual CO2 flux.

### How it works (calculation model)

- AGB: general allometric equation (Chave et al., 2014) AGB = a × (WD × DBH² × H)^b, with a=0.0673, b=0.976
- BGB: BGB = RSR × AGB (RSR default 0.24)
- Carbon: C = Total_biomass × CF (CF default 0.47)
- CO2 (stock): CO2 = C × 44/12 ≈ C × 3.667
- Annual CO2 (flux): ΔBiomass × CF × 3.667 (if annual increment is provided)

The implementation is encapsulated in the OOP service `CO2CalculationService`.

### Local (Python)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://localhost:8000` and the docs at `http://localhost:8000/docs`.

### Docker

Build and run all services (API + Streamlit):

```bash
# Development mode (con hot-reload)
docker compose up --build

# Solo Streamlit chatbot
docker compose up streamlit --build

# Produzione
docker compose -f docker-compose.prod.yml up -d
```

Then visit:

- **API**: http://localhost:8000 (docs: /docs)
- **Streamlit Chat**: http://localhost:8501

**Configurazione OpenAI API Key per Docker:**

```bash
# Opzione 1: File .env (raccomandato)
cp .env.example .env
# Modifica .env e inserisci: OPENAI_API_KEY=sk-...

# Opzione 2: Environment variable
OPENAI_API_KEY=sk-xxx docker compose up

# Opzione 3: Dall'UI Streamlit (funziona sempre)
# Settings → "OpenAI API Key" → inserisci chiave
```

Vedi [DOCKER.md](DOCKER.md) per configurazione avanzata.

**Uso Ollama in locale (senza API key):**

```bash
# Abilita Ollama come provider LLM
cp .env.example .env
echo "LLM_PROVIDER=ollama" >> .env

# (Opzionale) Modelli
echo "OLLAMA_CHAT_MODEL=qwen2.5:7b-instruct" >> .env
echo "OLLAMA_EMBEDDING_MODEL=nomic-embed-text" >> .env

# Avvio (di default usa Ollama installato sul tuo host)
docker compose up --build

# Prima esecuzione: scarica i modelli (sul tuo host)
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text

# (Opzionale) Se vuoi Ollama dentro Docker invece che sull'host:
# docker compose --profile with-ollama up --build
# e imposta: OLLAMA_BASE_URL=http://ollama:11434
```

### One-line install and run

macOS/Linux:

```bash
bash install.sh --run
```

Windows:

```bat
install.bat --run
```

Swagger UI will be available at `http://localhost:8000/docs`.

### Manual installation (alternative)

macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Tip: use the local `.venv` (avoid system environments like Anaconda for tests).

### Main endpoints

- Health check: `GET /api/v1/health/`
- CO2 calculation: `POST /api/v1/co2/calc`

Request body (JSON):

```json
{
  "dbh_cm": 30.0,
  "height_m": 15.0,
  "wood_density_g_cm3": 0.6,
  "carbon_fraction": 0.47,
  "root_shoot_ratio": 0.24,
  "annual_biomass_increment_t": 0.03
}
```

Response (JSON):

```json
{
  "agb_t": 0.44,
  "bgb_t": 0.106,
  "total_biomass_t": 0.546,
  "carbon_t": 0.256,
  "co2_stock_t": 0.94,
  "co2_annual_t": 0.052
}
```

cURL example:

```bash
curl -X POST "http://localhost:8000/api/v1/co2/calc" \
  -H "Content-Type: application/json" \
  -d '{
    "dbh_cm": 30.0,
    "height_m": 15.0,
    "wood_density_g_cm3": 0.6,
    "carbon_fraction": 0.47,
    "root_shoot_ratio": 0.24,
    "annual_biomass_increment_t": 0.03
  }'
```

### Data glossary (inputs and outputs)

Inputs:

- `dbh_cm`: diameter at breast height (DBH) in centimeters; float > 0.
- `height_m`: total tree height in meters; float > 0.
- `wood_density_g_cm3`: wood density in g/cm³ (species-specific, typically 0.3–1.0); float > 0.
- `carbon_fraction`: fraction of carbon on dry mass (default 0.47); float within (0,1).
- `root_shoot_ratio`: root-to-shoot ratio R:S to estimate below-ground biomass (default 0.24); float > 0.
- `annual_biomass_increment_t`: annual increment of dry biomass in tonnes per tree per year (optional); float ≥ 0.

Outputs:

- `agb_t`: Above-Ground Biomass in tonnes/tree.
- `bgb_t`: Below-Ground Biomass in tonnes/tree.
- `total_biomass_t`: total biomass (AGB + BGB) in tonnes/tree.
- `carbon_t`: carbon stock in tonnes of C/tree.
- `co2_stock_t`: CO2-equivalent stock in tonnes of CO2e/tree.
- `co2_annual_t`: annual CO2 uptake in tonnes of CO2e/tree/year (present only if `annual_biomass_increment_t` is provided).

Notes:

- Inputs refer to a single tree; for per-hectare values multiply by the number of trees/ha.
- If species is unknown, use an average wood density for the biome or local context.

### Project architecture

- `app/main.py`: FastAPI app creation and configuration
- `app/core/config.py`: application configuration (`APP_NAME`, `APP_VERSION`, `APP_ENV`)
- `app/api/v1/router.py`: v1 routers registration
- `app/api/v1/endpoints/`: endpoint groups (`health.py`, `co2.py`, `environment.py`)
- `app/services/`: OOP application logic (`health_service.py`, `co2_service.py`, `environment_service.py`)
- `app/models/`: Pydantic request/response models (`response.py`, `co2.py`, `environment.py`)
- `tests/`: integration tests

### Streamlit Chat App with LangChain/LangGraph

The project includes an intelligent chatbot interface built with:

- **Streamlit**: Interactive web UI
- **LangChain/LangGraph**: Agent orchestration
- **OpenAI GPT-4**: Language model
- **SQLite**: Conversation persistence

#### Agent Capabilities (Tools)

The chatbot agent has access to **20+ specialized tools** organized in the following categories:

##### 1. CO2 & Carbon Calculations

| Tool | Description | Example Questions |
|------|-------------|-------------------|
| **CO2 Calculation** | Calculate CO2 sequestration for single trees using Chave et al. (2014) allometric equations | "Calcola CO2 per un albero con DBH 30cm e altezza 15m" |
| **CO2 Aggregate** | Calculate total carbon stock for groups of trees from the dataset | "Quanto carbonio stoccano tutti i Platanus del distretto 5?" |
| **Carbon Sequestration Lookup** | Annual carbon sequestration rates per species (Paoletti et al.) | "Quanto carbonio sequestra un Acer platanoides all'anno?" |
| **Carbon Projection** | Project future carbon sequestration trends over time | "Proiezione a 30 anni per 10 tigli di 20 anni" |
| **Carbon Content Lookup** | Species-specific carbon fraction (Martin et al., 2018) | "Qual è la frazione di carbonio per la quercia?" |

##### 2. Biomass Calculations

| Tool | Description |
|------|-------------|
| **Total Biomass** | Total biomass = e^(-4.2) × D^1.36 × H^0.57 × age^1.67 × (R/S)^(-0.3) × 1.23 |
| **Stem Biomass** | Stem biomass with interaction term (D × age) |
| **Leaf Biomass** | Leaf biomass estimation |
| **Root Biomass** | Below-ground biomass estimation |
| **Ipogeo/Epigeo Ratios** | Root-to-shoot ratios for hardwood/softwood from dataset |

##### 3. Volume Calculations

| Tool | Description |
|------|-------------|
| **General Volume** | V = a × D² × H (classic allometric model) |
| **Heyer Volume** | Heyer formula for volume estimation |
| **Simplified Volume** | Simplified volume calculation |

##### 4. Allometric Relations

| Tool | Description |
|------|-------------|
| **General Allometric** | Y = a × X^b (fundamental allometric equation) |
| **Log Allometric** | Logarithmic allometric relationships |
| **Log Fuel Biomass** | Fuel biomass estimation |

##### 5. Dataset Queries

| Tool | Description | Example Questions |
|------|-------------|-------------------|
| **Tree Dataset Query** | Natural language to SQL for Vienna/Milano datasets (~230K trees) | "Quanti alberi ci sono nel distretto 19?", "Top 5 specie più comuni" |
| **Species List Query** | Taxonomy and traits lookup (family, order, growth form, leaf type) | "Dimmi la famiglia dell'Acer platanoides", "Specie del genere Abies" |

##### 6. Visualizations

| Tool | Description | Example Questions |
|------|-------------|-------------------|
| **Chart Generation** | 6 chart types: bar, pie, line, scatter, histogram, box plot | "Grafico a barre degli alberi per distretto", "Istogramma dell'età" |
| **Map Generation** | Interactive maps: markers, clusters, heatmaps (requires GPS coordinates) | "Mappa dei tigli a Milano", "Heatmap della distribuzione degli alberi" |

##### 7. Research & Export

| Tool | Description | Example Questions |
|------|-------------|-------------------|
| **Scientific Paper Search** | Search arXiv and PubMed for scientific papers | "Cerca paper su carbon sequestration in urban trees" |
| **Data Export** | Export query results to CSV or Excel | "Esporta i risultati in CSV" |

##### 8. Environmental Estimates

| Tool | Description |
|------|-------------|
| **Environment Estimation** | Volume, biomass, carbon stock with confidence metrics |

#### Scientific References

All calculations are based on peer-reviewed scientific literature:

- **Chave et al. (2014)**: AGB allometric equations - [DOI:10.1111/gcb.12629](https://doi.org/10.1111/gcb.12629)
- **Martin et al. (2018)**: Carbon content of tree tissues - [DOI:10.1007/s10021-017-0198-4](https://doi.org/10.1007/s10021-017-0198-4)
- **Paoletti et al.**: Annual carbon sequestration rates per species
- **Cairns et al. (1997)**: Root biomass allocation - [DOI:10.1007/s004420050128](https://doi.org/10.1007/s004420050128)

#### Setup

1. **Copy environment template:**

   ```bash
   cp .env.example .env
   ```

2. **Add your OpenAI API key** to `.env`:

   ```
   OPENAI_API_KEY=sk-your-key-here
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app:**

   ```bash
   streamlit run streamlit_app/app.py
   ```

   Or with Docker:

   ```bash
   docker compose up streamlit
   ```

   Visit `http://localhost:8501`

#### Usage Examples

Ask the chatbot questions like:

**Dataset queries:**

- "Quanti alberi ci sono nel distretto 19?"
- "Mostrami gli alberi Acer piantati dopo il 2000"
- "Statistiche per distretto"
- "Qual è l'albero più vecchio del dataset?"
- "Top 10 specie più comuni"

**CO2 calculations (single tree):**

- "Calcola il CO2 sequestrato da un albero con diametro 30 cm e altezza 15 metri"
- "Quanta biomassa ha un Acer con circonferenza tronco 94 cm e altezza 12 m?"

**CO2 aggregate (groups of trees):**

- "Quanto carbonio stoccano tutti i Platanus del distretto 5?"
- "Stock di CO2 totale per tutti gli alberi del distretto 19"

**Carbon sequestration rates:**

- "Quanto carbonio sequestra un Acer platanoides all'anno?"
- "Confronta il sequestro annuale di carbonio tra Tilia e Quercus"
- "Stoccaggio annuale di carbonio per 100 tigli"

**Future projections:**

- "Proiezione a 30 anni per un tiglio di 20 anni"
- "Quanto carbonio avrà un acero tra 50 anni?"

**Species lookup:**

- "Qual è la frazione di carbonio per la quercia?"
- "Dimmi la famiglia e l'ordine di Acer platanoides"
- "Quali specie sono della famiglia Pinaceae?"

**Chart generation:**

- "Crea un grafico a barre dei distretti con più alberi"
- "Mostra un grafico a torta delle 5 specie più comuni"
- "Fai un istogramma dell'età degli alberi"
- "Crea un grafico a linee delle piantumazioni per anno dal 1950"
- "Mostra un box plot della circonferenza per le specie principali"

**Map generation (Milano dataset only):**

- "Mostra una mappa con tutti i tigli"
- "Crea una heatmap della distribuzione degli alberi a Milano"
- "Visualizza su mappa gli alberi del municipio 3"

**Scientific papers:**

- "Cerca paper su carbon sequestration in urban trees"
- "Trova articoli scientifici su allometric equations for biomass"

**Data export:**

- "Esporta i risultati in CSV"
- "Scarica i dati in Excel"

#### Architecture

```text
streamlit_app/
├── app.py              # Main entry point
├── ui.py               # Streamlit UI components (with chart/map visualization)
├── service.py          # Chat service with agent integration
├── repository.py       # SQLite persistence layer
├── models.py           # Domain models (Conversation, ChatMessage)
├── agent/              # LangGraph agent modules
│   ├── core.py         # Main agent orchestrator
│   ├── state.py        # Agent state management
│   ├── prompts.py      # System prompts and templates
│   └── ...
└── tools/              # 20+ LangChain tools
    ├── co2_tool.py                 # CO2 calculation (single tree)
    ├── co2_aggregate_tool.py       # CO2 aggregate (groups of trees)
    ├── carbon_sequestration_tool.py # Annual sequestration rates
    ├── carbon_projection_tool.py   # Future carbon projections
    ├── carbon_content_tool.py      # Species carbon fractions
    ├── total_biomass_tool.py       # Total biomass calculation
    ├── stem_biomass_tool.py        # Stem biomass
    ├── leaf_biomass_tool.py        # Leaf biomass
    ├── root_biomass_tool.py        # Root biomass
    ├── ipogeo_epigeo_tool.py       # Root/shoot ratios
    ├── general_volume_tool.py      # Volume equations
    ├── heyer_volume_tool.py        # Heyer volume formula
    ├── allometric_relation_tool.py # General allometric Y = a × X^b
    ├── dataset_tool.py             # Vienna/Milano dataset queries
    ├── species_list_tool.py        # Taxonomy and traits lookup
    ├── chart_tool.py               # Interactive Plotly charts
    ├── map_tool.py                 # Interactive Folium maps
    ├── paper_search_tool.py        # arXiv/PubMed search
    ├── export_tool.py              # CSV/Excel export
    ├── environment_tool.py         # Environmental estimates
    └── ...
```

The agent uses LangGraph to orchestrate tool calls:

1. User sends message
2. Agent (GPT-4) analyzes query and selects appropriate tool(s)
3. Tools execute (call FastAPI services, query datasets, generate visualizations)
4. Agent synthesizes response in Italian with scientific references
5. Response stored in SQLite and shown to user

#### Key Tool Features

**Natural Language to SQL**: Both `DatasetQueryTool` and `SpeciesListQueryTool` translate natural language questions into optimized SQL queries with automatic vector search for large result sets.

**Scientific References**: All calculation tools return the formulas used and their scientific sources (DOI links).

**Species-Specific Parameters**: Tools like `CO2AggregateTool` automatically look up species-specific carbon fractions and root-to-shoot ratios from the included datasets (`carbon_content.csv`, `ipogeo_epigeo.csv`, `c_sequestration.csv`).

**Interactive Visualizations**: Charts (Plotly) and maps (Folium) are interactive with zoom, pan, hover tooltips, and export capabilities.

See [CHART_TOOL_GUIDE.md](CHART_TOOL_GUIDE.md) for detailed chart documentation.

#### Configuration

Environment variables (`.env`):

```bash
# Required
OPENAI_API_KEY=your_key_here

# Optional
CHAT_DB_PATH=data/chat_index.db
APP_ENV=development
```

### Testing

Run integration tests:

```bash
pytest tests/
```

#### Ground truth evaluation commands

L'agente LangGraph può essere validato contro il dataset di ground truth (`dataset/ground_truth.csv`).

**Come funziona:**

Il comando `python tests/ground_truth_runner.py` esegue le seguenti operazioni:

1. **Carica il dataset di ground truth** dal file CSV (`dataset/ground_truth.csv`)
2. **Per ogni domanda nel dataset:**
   - Invia la domanda all'agente `TreeEvaluatorAgent` (via `TreeAgentClient`)
   - Riceve la risposta dell'LLM
   - Estrae il valore numerico dalla risposta (se presente)
   - Confronta la risposta numerica con quella attesa (con tolleranza configurabile)
   - Calcola la similarità testuale tra risposta LLM e risposta attesa (usando SequenceMatcher)
3. **Genera un report** con:
   - Accuratezza numerica (% di risposte numeriche corrette)
   - Similarità testuale media
   - Lista dei record che hanno fallito con i motivi

**Uso:**

```bash
# Assicurati di avere OPENAI_API_KEY impostata
export OPENAI_API_KEY=sk-...

# Esegui tutte le domande del ground truth
python tests/ground_truth_runner.py

# Limita a 5 domande per test rapidi
python tests/ground_truth_runner.py --limit 5

# Personalizza tolleranza numerica (default: 1% relativo)
python tests/ground_truth_runner.py --tolerance 0.05

# Personalizza soglia di similarità testuale (default: 0.65)
python tests/ground_truth_runner.py --text-threshold 0.70

# Combina più opzioni
python tests/ground_truth_runner.py --limit 10 --tolerance 0.02 --text-threshold 0.75
```

**Output esempio:**

```
=== Ground Truth Accuracy Report ===
Records evaluated: 10
Numeric accuracy: 80.0%
Average text similarity: 72.5%

Failures:
- ID 3: Numeric mismatch (expected 21363.0, got 21000.0)
- ID 5: Low text similarity (0.58)
```

**Test automatizzato Pytest:**

Per integrare la valutazione nei test automatizzati:

```bash
pytest tests/test_ground_truth_agent.py -v
```

Il test è marcato come `@pytest.mark.slow` e viene saltato se `OPENAI_API_KEY` non è impostata.

### Dataset

Place your tree dataset CSV/Excel files in the `dataset/` folder. The chatbot will automatically load and query them.

Current dataset: **BAUMKATOGD.csv** (Vienna trees cadastre)

- ~230K trees
- Columns: DISTRICT, GENUS_SPECIES, PLANT_YEAR, TRUNK_CIRCUMFERENCE, TREE_HEIGHT, CROWN_DIAMETER, coordinates, etc.

### License

MIT
