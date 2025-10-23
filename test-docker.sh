#!/bin/bash
# Test script per verificare setup Docker

set -e

echo "🧪 Tree Evaluator - Docker Test Script"
echo "========================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  File .env non trovato${NC}"
    echo "Copia .env.example a .env e configura OPENAI_API_KEY"
    echo "  cp .env.example .env"
    exit 1
fi

# Check if OPENAI_API_KEY is set
source .env
if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "sk-your-openai-api-key-here" ]; then
    echo -e "${YELLOW}⚠️  OPENAI_API_KEY non configurata in .env${NC}"
    echo "Puoi comunque avviare Streamlit e inserire la chiave dall'UI"
fi

echo ""
echo "1️⃣  Building Docker images..."
docker compose build --no-cache

echo ""
echo "2️⃣  Starting services..."
docker compose up -d

echo ""
echo "3️⃣  Waiting for services to be ready..."
sleep 5

echo ""
echo "4️⃣  Testing API health..."
if curl -f http://localhost:8000/api/v1/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ API is healthy${NC}"
else
    echo -e "${RED}❌ API health check failed${NC}"
    docker compose logs api
    exit 1
fi

echo ""
echo "5️⃣  Testing Streamlit..."
if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Streamlit is running${NC}"
else
    echo -e "${YELLOW}⚠️  Streamlit health check failed (potrebbe essere normale)${NC}"
fi

echo ""
echo "6️⃣  Checking dataset availability..."
docker compose exec -T streamlit ls -lh /app/dataset/BAUMKATOGD.csv > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Dataset accessible${NC}"
else
    echo -e "${RED}❌ Dataset not found in container${NC}"
fi

echo ""
echo "7️⃣  Checking volumes..."
docker volume ls | grep chat_data > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Chat data volume created${NC}"
else
    echo -e "${RED}❌ Chat data volume not found${NC}"
fi

echo ""
echo "========================================"
echo -e "${GREEN}🎉 Setup completato!${NC}"
echo ""
echo "Accedi ai servizi:"
echo "  - API:        http://localhost:8000"
echo "  - API Docs:   http://localhost:8000/docs"
echo "  - Streamlit:  http://localhost:8501"
echo ""
echo "Comandi utili:"
echo "  docker compose logs -f           # Logs in tempo reale"
echo "  docker compose logs -f streamlit # Solo Streamlit"
echo "  docker compose down              # Stop servizi"
echo "  docker compose restart streamlit # Restart Streamlit"
echo ""
echo "Per testare il chatbot:"
echo "  1. Apri http://localhost:8501"
echo "  2. Inserisci OpenAI API Key in Settings (se non in .env)"
echo "  3. Crea nuova conversazione"
echo "  4. Chiedi: 'Quanti alberi ci sono nel dataset?'"

