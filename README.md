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

#### Features

The chatbot can:

1. **Calculate CO2 sequestration** for individual trees given measurements (DBH, height, wood density)
2. **Query the Vienna trees dataset** (BAUMKATOGD.csv) with:
   - Filtering by district, species, plant year
   - Aggregations and statistics
   - Random sampling
   - Count queries
3. **Compute environmental estimates** (volume, biomass, carbon stock)
4. **Maintain conversation history** with multi-session support

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
- "Dammi 5 alberi casuali"

**CO2 calculations:**

- "Calcola il CO2 sequestrato da un albero con diametro 30 cm e altezza 15 metri"
- "Quanta biomassa ha un Acer con circonferenza tronco 94 cm e altezza 12 m?"

**Environmental estimates:**

- "Stima il volume di un albero con diametro 25 cm"
- "Calcola carbonio stoccato per diametro 40 cm e altezza 18 m"

#### Architecture

```
streamlit_app/
├── app.py              # Main entry point
├── ui.py               # Streamlit UI components
├── service.py          # Chat service with agent integration
├── repository.py       # SQLite persistence layer
├── models.py           # Domain models (Conversation, ChatMessage)
├── agent.py            # LangGraph agent orchestrator
└── tools/              # LangChain tools
    ├── co2_tool.py           # CO2 calculation tool
    ├── environment_tool.py   # Environmental estimates tool
    └── dataset_tool.py       # Dataset query tool
```

The agent uses LangGraph to orchestrate tool calls:

1. User sends message
2. Agent (GPT-4) decides which tool(s) to call
3. Tools execute (call existing FastAPI services or query dataset)
4. Agent synthesizes response in Italian
5. Response stored in SQLite and shown to user

#### Tool Details

**CO2CalculationTool**: Wraps `CO2CalculationService` from FastAPI

- Calculates AGB, BGB, total biomass, carbon, CO2 stock
- Uses Chave et al. (2014) allometric equation
- Supports custom wood density per species

**EnvironmentEstimationTool**: Wraps `EnvironmentalEstimationService`

- Computes volume, biomass, carbon with alternative formulas
- Works with/without height data
- Provides confidence metrics

**DatasetQueryTool**: Direct pandas queries on BAUMKATOGD.csv

- Summary statistics (total trees, species count, districts)
- Filtering (district, species, plant year range)
- Aggregations (group by district/species with medians)
- Random sampling for data exploration
- Count queries with filters

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

### Dataset

Place your tree dataset CSV/Excel files in the `dataset/` folder. The chatbot will automatically load and query them.

Current dataset: **BAUMKATOGD.csv** (Vienna trees cadastre)

- ~230K trees
- Columns: DISTRICT, GENUS_SPECIES, PLANT_YEAR, TRUNK_CIRCUMFERENCE, TREE_HEIGHT, CROWN_DIAMETER, coordinates, etc.

### License

MIT
