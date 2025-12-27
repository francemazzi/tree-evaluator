# Docker Setup per Tree Evaluator

## 🚀 Quick Start

### Sviluppo (con hot-reload)

```bash
# Build e avvio
docker compose up --build

# Solo Streamlit
docker compose up streamlit

# Solo API
docker compose up api
```

Accedi a:

- **Streamlit Chat**: http://localhost:8501
- **FastAPI**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### Produzione

```bash
docker compose -f docker-compose.prod.yml up --build -d
```

## 🔨 Script di Build

Per semplificare la gestione della build dei container, sono disponibili script dedicati:

### Linux/macOS

```bash
# Build tutti i servizi
./build.sh

# Build solo un servizio
./build.sh api
./build.sh streamlit

# Rebuild completo senza cache
./build.sh --no-cache

# Build per produzione
./build.sh --prod

# Pulizia e rebuild completo
./build.sh --clean --no-cache

# Build con pull immagini base aggiornate
./build.sh --pull

# Mostra help
./build.sh --help
```

### Windows

```batch
# Build tutti i servizi
build.bat

# Build solo un servizio
build.bat api
build.bat streamlit

# Rebuild completo senza cache
build.bat --no-cache

# Build per produzione
build.bat --prod

# Pulizia e rebuild completo
build.bat --clean --no-cache

# Build con pull immagini base aggiornate
build.bat --pull

# Mostra help
build.bat --help
```

### Comandi Docker Compose Diretti

Se preferisci usare direttamente Docker Compose:

```bash
# Build tutti i servizi (sviluppo)
docker compose build

# Build senza cache
docker compose build --no-cache

# Build solo un servizio
docker compose build api
docker compose build streamlit

# Build per produzione
docker compose -f docker-compose.prod.yml build

# Build e avvio in un comando
docker compose up --build

# Build e avvio in background
docker compose up --build -d
```

## 📦 Immagini Docker

### 1. API (FastAPI)

**Dockerfile**: `Dockerfile`

- Base: `python:3.11-slim`
- Porta: `8000`
- Include: `app/`, `requirements.txt`

### 2. Streamlit Chat

**Dockerfile**: `Dockerfile.streamlit`

- Base: `python:3.11-slim`
- Porta: `8501`
- Include: `streamlit_app/`, `app/` (per services), `dataset/`
- Dipendenze extra: `gcc`, `g++`, `build-essential` (per pandas/numpy)

## 🔑 Gestione OpenAI API Key

### Opzione 1: File .env (Raccomandato per sviluppo)

```bash
# Crea file .env
cp .env.example .env

# Modifica .env
OPENAI_API_KEY=sk-your-key-here
```

Il `docker-compose.yml` carica automaticamente `.env` con `env_file`.

### Opzione 2: Inserimento dall'UI (Funziona sempre)

1. Avvia Streamlit: `docker compose up streamlit`
2. Apri http://localhost:8501
3. Sidebar → Settings → "OpenAI API Key"
4. Incolla la chiave

### Opzione 3: Environment variable (Produzione)

```bash
# Passa al runtime
OPENAI_API_KEY=sk-xxx docker compose up

# O con -e
docker compose up -e OPENAI_API_KEY=sk-xxx
```

### Opzione 4: Docker secrets (Produzione avanzata)

```bash
# Crea secret
echo "sk-your-key" | docker secret create openai_api_key -

# Usa in docker-compose con swarm mode
docker stack deploy -c docker-compose.prod.yml tree-evaluator
```

## 📂 Volumi e persistenza

### Volume `chat_data`

```yaml
volumes:
  chat_data:
    driver: local
```

**Cosa contiene:**

- `chat_index.db`: SQLite database con conversazioni
- Persistente tra riavvii container

**Dove si trova:**

```bash
# Ispeziona volume
docker volume inspect tree-evaluator_chat_data

# Backup database
docker run --rm \
  -v tree-evaluator_chat_data:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/chat_backup.tar.gz -C /data .

# Restore database
docker run --rm \
  -v tree-evaluator_chat_data:/data \
  -v $(pwd):/backup \
  alpine tar xzf /backup/chat_backup.tar.gz -C /data
```

### Mount bind (sviluppo)

```yaml
volumes:
  - ./streamlit_app:/app/streamlit_app:ro # read-only hot-reload
  - ./app:/app/app:ro # API services
  - ./dataset:/app/dataset:ro # CSV dataset
```

**Hot-reload attivo** per modifiche codice (eccetto dataset che è RO).

## 🏗️ Build ottimizzato

### Layer caching

I Dockerfile sono ottimizzati per cache:

1. System dependencies (cambia raramente)
2. `requirements.txt` (cambia occasionalmente)
3. Codice applicazione (cambia spesso)

### Build da zero

```bash
# Senza cache
docker compose build --no-cache

# Rimuovi immagini vecchie
docker compose down --rmi all
docker system prune -a
```

## 🔍 Debug e logs

### Logs in tempo reale

```bash
# Tutti i servizi
docker compose logs -f

# Solo Streamlit
docker compose logs -f streamlit

# Ultimi 100 righe
docker compose logs --tail=100 streamlit
```

### Shell interattiva

```bash
# Accedi al container Streamlit
docker compose exec streamlit /bin/bash

# Verifica installazioni
docker compose exec streamlit pip list

# Test connettività
docker compose exec streamlit curl http://api:8000/api/v1/health

# Verifica dataset
docker compose exec streamlit ls -lh /app/dataset/
```

### Health checks

```bash
# Status servizi
docker compose ps

# Health API
curl http://localhost:8000/api/v1/health

# Health Streamlit
curl http://localhost:8501/_stcore/health
```

## 🚨 Troubleshooting

### "ModuleNotFoundError: No module named 'pandas'"

→ Rebuild con `--no-cache`:

```bash
docker compose build --no-cache streamlit
docker compose up streamlit
```

### "Dataset not found"

→ Verifica che `dataset/` sia montato:

```bash
docker compose exec streamlit ls -la /app/dataset/
```

Se vuoto, aggiungi volume in `docker-compose.yml`:

```yaml
volumes:
  - ./dataset:/app/dataset:ro
```

### "OpenAI API key not found"

→ Tre opzioni:

1. Inserisci dall'UI (Settings)
2. Aggiungi a `.env`: `OPENAI_API_KEY=sk-...`
3. Passa come env var: `OPENAI_API_KEY=sk-... docker compose up`

### Streamlit non si avvia

→ Controlla logs:

```bash
docker compose logs streamlit
```

Verifica porte libere:

```bash
# macOS/Linux
lsof -i :8501

# Windows
netstat -ano | findstr :8501
```

### Container esce subito

→ Controlla errori Python:

```bash
docker compose logs streamlit

# O run manuale per debug
docker compose run --rm streamlit python -c "import streamlit; print('OK')"
```

### Permessi SQLite database

→ Il volume `chat_data` deve essere scrivibile:

```bash
# Se usi bind mount locale
mkdir -p ./chat_data
chmod 777 ./chat_data
```

## 📊 Performance

### Ottimizzazioni produzione

1. **Multi-stage build** (opzionale):

```dockerfile
# Builder stage
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.11-slim
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
WORKDIR /app
COPY streamlit_app ./streamlit_app
...
```

2. **Resource limits**:

```yaml
services:
  streamlit:
    deploy:
      resources:
        limits:
          cpus: "1.0"
          memory: 1G
        reservations:
          cpus: "0.5"
          memory: 512M
```

3. **Replica scale**:

```bash
docker compose up --scale streamlit=3
```

(Richiede load balancer per multiple istanze Streamlit)

## 🔐 Sicurezza produzione

### 1. Non esporre API pubblicamente

```yaml
services:
  api:
    # ports:
    #   - "8000:8000"  # ❌ Commentato
    expose:
      - "8000" # ✅ Solo rete interna
```

### 2. Secrets per API key

```yaml
services:
  streamlit:
    secrets:
      - openai_api_key

secrets:
  openai_api_key:
    external: true
```

### 3. User non-root

```dockerfile
RUN useradd -m -u 1000 appuser
USER appuser
```

### 4. Scan vulnerabilità

```bash
# Con Trivy
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image tree-evaluator-streamlit:latest
```

## 🌐 Deploy cloud

### Docker Swarm

```bash
docker swarm init
docker stack deploy -c docker-compose.prod.yml tree-evaluator
```

### Kubernetes (Helm chart consigliato)

```bash
# Converti docker-compose in k8s (con kompose)
kompose convert -f docker-compose.prod.yml

# Deploy
kubectl apply -f .
```

### Cloud providers

- **AWS**: ECS/Fargate o EKS
- **Azure**: Container Instances o AKS
- **GCP**: Cloud Run o GKE
- **DigitalOcean**: App Platform
- **Fly.io**: `fly launch`

## 📋 Comandi utili

```bash
# Stop tutto
docker compose down

# Stop e rimuovi volumi (⚠️ cancella database!)
docker compose down -v

# Rebuild singolo servizio
docker compose build streamlit

# Restart senza rebuild
docker compose restart streamlit

# Esegui comando one-off
docker compose run --rm streamlit python -c "print('test')"

# Export logs
docker compose logs > logs.txt

# Stats risorse
docker stats

# Cleanup totale
docker system prune -a --volumes
```

## 📚 Riferimenti

- [Docker Compose docs](https://docs.docker.com/compose/)
- [Streamlit Docker guide](https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker)
- [FastAPI Docker guide](https://fastapi.tiangolo.com/deployment/docker/)
