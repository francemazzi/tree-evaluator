#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash install.sh [--run]
#
# Options:
#   --run  After installation, start the API server and keep it running.

RUN_AFTER_INSTALL=0
if [[ "${1:-}" == "--run" ]]; then
    RUN_AFTER_INSTALL=1
fi

PYTHON_BIN=${PYTHON_BIN:-python3.11}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN=python3
fi

echo "[install] Using Python: $PYTHON_BIN"

"$PYTHON_BIN" -m venv .venv
echo "[install] Created virtual environment at .venv"

".venv/bin/python" -m pip install --upgrade pip
".venv/bin/python" -m pip install -r requirements.txt

echo "[install] Installation complete."
echo "[install] To run the server: .venv/bin/uvicorn app.main:app --reload"
echo "[install] Swagger UI: http://localhost:8000/docs"

if [[ "$RUN_AFTER_INSTALL" -eq 1 ]]; then
    echo "[install] Starting API server..."
    exec .venv/bin/uvicorn app.main:app --reload
fi


