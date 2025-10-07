## Tree Evaluator API

API minimale basata su FastAPI (con design OOP) per stimare la CO2 assorbita dagli alberi a partire da parametri dendrometrici. Fornisce un endpoint che calcola biomassa epigea (AGB), biomassa ipogea (BGB), biomassa totale, carbonio e CO2, con possibilità di stima del flusso annuo di CO2.

### Come funziona (modello di calcolo)

- AGB: equazione allometrica generale (Chave et al., 2014) AGB = a × (WD × DBH² × H)^b, con a=0.0673, b=0.976
- BGB: BGB = RSR × AGB (RSR default 0.24)
- Carbonio: C = Biomassa_tot × CF (CF default 0.47)
- CO2 (stock): CO2 = C × 44/12 ≈ C × 3.667
- CO2 annua (flussso): ΔBiomassa × CF × 3.667 (se fornito incremento annuo)

L’implementazione è incapsulata nel servizio OOP `CO2CalculationService`.

### Local (Python)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://localhost:8000` and docs at `http://localhost:8000/docs`.

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

Swagger UI sarà disponibile su `http://localhost:8000/docs`.

### Installazione manuale (alternativa)

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

Consiglio: usa il venv locale `.venv` (evita ambienti di sistema tipo Anaconda per i test).

### Endpoint principali

- Health check: `GET /api/v1/health/`
- Calcolo CO2: `POST /api/v1/co2/calc`

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

Esempio cURL:

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

### Legenda dati (input e output)

Input:

- `dbh_cm`: diametro a petto d’uomo (DBH) in centimetri; float > 0.
- `height_m`: altezza totale dell’albero in metri; float > 0.
- `wood_density_g_cm3`: densità del legno in g/cm³ (specie-specifica, tipico 0.3–1.0); float > 0.
- `carbon_fraction`: frazione di carbonio sulla massa secca (default 0.47); float nell’intervallo (0,1).
- `root_shoot_ratio`: rapporto radici/fusto R:S per stimare la biomassa ipogea (default 0.24); float > 0.
- `annual_biomass_increment_t`: incremento annuo di biomassa secca in tonnellate per albero per anno (opzionale); float ≥ 0.

Output:

- `agb_t`: Above-Ground Biomass (biomassa epigea) in tonnellate/albero.
- `bgb_t`: Below-Ground Biomass (biomassa ipogea) in tonnellate/albero.
- `total_biomass_t`: biomassa totale (AGB + BGB) in tonnellate/albero.
- `carbon_t`: stock di carbonio in tonnellate di C/albero.
- `co2_stock_t`: stock equivalente di CO2 in tonnellate di CO2e/albero.
- `co2_annual_t`: assorbimento annuo di CO2 in tonnellate di CO2e/albero/anno (presente solo se fornito `annual_biomass_increment_t`).

Note:

- Gli input si riferiscono a un singolo albero; per ettaro moltiplica per il numero di alberi/ha.
- Se la specie non è nota, usa una densità del legno media del bioma o locale.

### Architettura del progetto

- `app/main.py`: creazione e configurazione dell’app FastAPI
- `app/core/config.py`: configurazione applicativa (`APP_NAME`, `APP_VERSION`, `APP_ENV`)
- `app/api/v1/router.py`: registrazione dei router v1
- `app/api/v1/endpoints/`: gruppi di endpoint (`health.py`, `co2.py`)
- `app/services/`: logica OOP applicativa (`health_service.py`, `co2_service.py`)
- `app/models/`: modelli Pydantic di request/response (`response.py`, `co2.py`)
- `tests/`: test di integrazione (`test_co2_integration.py`)

### Testing

Esegui i test (consigliato dentro `.venv`):

```bash
pytest -q
```

### Variabili d’ambiente (opzionali)

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
```
