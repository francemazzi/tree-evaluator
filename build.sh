#!/usr/bin/env bash
# Script per gestire la build dei container Docker
# Ottimizzato per Mac ARM64 con build sequenziale

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Funzione di help
show_help() {
    echo "Usage: ./build.sh [OPTIONS] [SERVICE]"
    echo ""
    echo "Build Docker containers per Tree Evaluator"
    echo ""
    echo "Options:"
    echo "  --no-cache          Build senza cache (rebuild completo)"
    echo "  --prod              Build per produzione (usa docker-compose.prod.yml)"
    echo "  --clean             Rimuovi immagini vecchie prima della build"
    echo "  --pull              Pull immagini base aggiornate"
    echo "  --parallel          Build in parallelo (default: sequenziale)"
    echo "  --help              Mostra questo messaggio"
    echo ""
    echo "Services:"
    echo "  api                 Build solo il servizio API"
    echo "  streamlit           Build solo il servizio Streamlit"
    echo "  all                 Build tutti i servizi (default)"
    echo ""
    echo "Examples:"
    echo "  ./build.sh                          # Build tutti i servizi (sequenziale)"
    echo "  ./build.sh api                      # Build solo API"
    echo "  ./build.sh --no-cache               # Rebuild completo senza cache"
    echo "  ./build.sh --prod                   # Build per produzione"
    echo "  ./build.sh --clean --no-cache       # Pulizia e rebuild completo"
    echo "  ./build.sh --parallel               # Build parallela (più veloce ma può fallire)"
    echo ""
}

# Funzione per build con retry
build_with_retry() {
    local service=$1
    local max_retries=3
    local retry=0
    
    while [ $retry -lt $max_retries ]; do
        echo -e "${CYAN}🔧 Building $service (tentativo $((retry + 1))/$max_retries)...${NC}"
        
        if docker compose -f "$COMPOSE_FILE" build $NO_CACHE "$service"; then
            echo -e "${GREEN}✅ $service build completata!${NC}"
            return 0
        else
            retry=$((retry + 1))
            if [ $retry -lt $max_retries ]; then
                echo -e "${YELLOW}⚠️  Build fallita, riprovo tra 3 secondi...${NC}"
                sleep 3
            fi
        fi
    done
    
    echo -e "${RED}❌ Build di $service fallita dopo $max_retries tentativi${NC}"
    return 1
}

# Variabili
NO_CACHE=""
PROD_MODE=false
CLEAN_MODE=false
PULL_IMAGES=false
PARALLEL_MODE=false
SERVICE="all"
COMPOSE_FILE="docker-compose.yml"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --prod)
            PROD_MODE=true
            COMPOSE_FILE="docker-compose.prod.yml"
            shift
            ;;
        --clean)
            CLEAN_MODE=true
            shift
            ;;
        --pull)
            PULL_IMAGES=true
            shift
            ;;
        --parallel)
            PARALLEL_MODE=true
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        api|streamlit|all)
            SERVICE=$1
            shift
            ;;
        *)
            echo -e "${RED}❌ Opzione sconosciuta: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

echo -e "${BLUE}🐳 Tree Evaluator - Docker Build${NC}"
echo "========================================"
echo ""

# Clean mode
if [ "$CLEAN_MODE" = true ]; then
    echo -e "${YELLOW}🧹 Pulizia immagini e cache...${NC}"
    docker compose -f "$COMPOSE_FILE" down --rmi local 2>/dev/null || true
    docker builder prune -f 2>/dev/null || true
    echo -e "${GREEN}✅ Pulizia completata${NC}"
    echo ""
fi

# Pull images
if [ "$PULL_IMAGES" = true ]; then
    echo -e "${YELLOW}📥 Pull immagini base...${NC}"
    docker compose -f "$COMPOSE_FILE" pull || true
    echo ""
fi

# Build
echo -e "${YELLOW}🔨 Building containers...${NC}"
echo "Compose file: $COMPOSE_FILE"
echo "Service: $SERVICE"
echo "No cache: $([ -n "$NO_CACHE" ] && echo "Yes" || echo "No")"
echo "Mode: $([ "$PARALLEL_MODE" = true ] && echo "Parallelo" || echo "Sequenziale")"
echo ""

if [ "$SERVICE" = "all" ]; then
    if [ "$PARALLEL_MODE" = true ]; then
        # Build parallela (può fallire su Mac ARM64)
        docker compose -f "$COMPOSE_FILE" build $NO_CACHE
    else
        # Build sequenziale (più affidabile)
        echo -e "${CYAN}📦 Build sequenziale per evitare conflitti...${NC}"
        echo ""
        
        build_with_retry "api"
        echo ""
        build_with_retry "streamlit"
    fi
else
    build_with_retry "$SERVICE"
fi

echo ""
echo -e "${GREEN}✅ Build completata!${NC}"
echo ""
echo "Comandi utili:"
echo "  docker compose -f $COMPOSE_FILE up -d        # Avvia servizi"
echo "  docker compose -f $COMPOSE_FILE logs -f      # Visualizza logs"
echo "  docker compose -f $COMPOSE_FILE ps           # Stato servizi"
echo "  docker compose -f $COMPOSE_FILE down         # Stop servizi"
echo ""
