## Tree Evaluator API

Minimal FastAPI stack with OOP structure and Docker.

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

### Project layout

```
app/
  core/config.py
  api/v1/router.py
  api/v1/endpoints/health.py
  services/health_service.py
  models/response.py
  main.py
```
