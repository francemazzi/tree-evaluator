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

Build and run:

```bash
docker compose up --build
```

Then visit `http://localhost:8000`.

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
- `app/api/v1/endpoints/`: endpoint groups (`health.py`, `co2.py`)
- `app/services/`: OOP application logic (`health_service.py`, `co2_service.py`)
- `app/models/`: Pydantic request/response models (`response.py`, `co2.py`)
- `tests/`: integration tests (`test_co2_integration.py`)

### Testing

Run tests (recommended inside `.venv`):

```bash
pytest -q
```

### Environment variables (optional)

- `APP_NAME` (default: "Tree Evaluator API")
- `APP_VERSION` (default: "0.1.0")
- `APP_ENV` (default: "development")

### Project layout

```
app/
  core/config.py
  api/v1/router.py
  api/v1/endpoints/health.py
  api/v1/endpoints/co2.py
  services/health_service.py
  services/co2_service.py
  models/response.py
  models/co2.py
  main.py
streamlit_app/
  __init__.py
  app.py
  models.py
  repository.py
  service.py
  ui.py
```

### Streamlit chat demo

This project includes a lightweight Streamlit demo app providing a fake chat with per-user history persisted in a local SQLite database `chat_index.db`.

Run locally (inside your virtualenv after installing `requirements.txt`):

```bash
streamlit run streamlit_app/app.py
```

Notes:

- The default user id is `guest` and can be changed from the sidebar.
- Click "Clear history" in the sidebar to wipe the history for the current user.
