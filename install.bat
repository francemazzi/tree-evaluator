@echo off
setlocal enabledelayedexpansion

REM Usage:
REM   install.bat [--run]

set RUN_AFTER_INSTALL=0
if "%1"=="--run" set RUN_AFTER_INSTALL=1

REM Prefer python 3.11 if available
where python >nul 2>nul
if errorlevel 1 (
  echo [install] Python not found in PATH. Please install Python 3.11+
  exit /b 1
)

echo [install] Creating virtual environment .venv
python -m venv .venv
if errorlevel 1 (
  echo [install] Failed to create virtual environment
  exit /b 1
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo [install] Installation complete.
echo [install] To run the server: .venv\Scripts\uvicorn app.main:app --reload
echo [install] Swagger UI: http://localhost:8000/docs

if "%RUN_AFTER_INSTALL%"=="1" (
  echo [install] Starting API server...
  .venv\Scripts\uvicorn app.main:app --reload
)

endlocal

