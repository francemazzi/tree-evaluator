@echo off
REM Script per gestire la build dei container Docker (Windows)

setlocal enabledelayedexpansion

set "NO_CACHE="
set "PROD_MODE=0"
set "CLEAN_MODE=0"
set "PULL_IMAGES=0"
set "SERVICE=all"
set "COMPOSE_FILE=docker-compose.yml"

REM Parse arguments
:parse_args
if "%~1"=="" goto :build
if "%~1"=="--no-cache" (
    set "NO_CACHE=--no-cache"
    shift
    goto :parse_args
)
if "%~1"=="--prod" (
    set "PROD_MODE=1"
    set "COMPOSE_FILE=docker-compose.prod.yml"
    shift
    goto :parse_args
)
if "%~1"=="--clean" (
    set "CLEAN_MODE=1"
    shift
    goto :parse_args
)
if "%~1"=="--pull" (
    set "PULL_IMAGES=1"
    shift
    goto :parse_args
)
if "%~1"=="--help" goto :help
if "%~1"=="-h" goto :help
if "%~1"=="api" (
    set "SERVICE=api"
    shift
    goto :parse_args
)
if "%~1"=="streamlit" (
    set "SERVICE=streamlit"
    shift
    goto :parse_args
)
if "%~1"=="all" (
    set "SERVICE=all"
    shift
    goto :parse_args
)
echo Unknown option: %~1
goto :help

:help
echo Usage: build.bat [OPTIONS] [SERVICE]
echo.
echo Build Docker containers per Tree Evaluator
echo.
echo Options:
echo   --no-cache          Build senza cache (rebuild completo)
echo   --prod              Build per produzione (usa docker-compose.prod.yml)
echo   --clean             Rimuovi immagini vecchie prima della build
echo   --pull              Pull immagini base aggiornate
echo   --help              Mostra questo messaggio
echo.
echo Services:
echo   api                 Build solo il servizio API
echo   streamlit           Build solo il servizio Streamlit
echo   all                 Build tutti i servizi (default)
echo.
echo Examples:
echo   build.bat                          # Build tutti i servizi
echo   build.bat api                      # Build solo API
echo   build.bat --no-cache               # Rebuild completo senza cache
echo   build.bat --prod                   # Build per produzione
echo   build.bat --clean --no-cache       # Pulizia e rebuild completo
echo.
exit /b 0

:build
echo 🐳 Tree Evaluator - Docker Build
echo ========================================
echo.

REM Clean mode
if "%CLEAN_MODE%"=="1" (
    echo 🧹 Pulizia immagini vecchie...
    docker compose -f "%COMPOSE_FILE%" down --rmi local 2>nul || echo Cleanup completed
    echo ✅ Pulizia completata
    echo.
)

REM Pull images
if "%PULL_IMAGES%"=="1" (
    echo 📥 Pull immagini base...
    docker compose -f "%COMPOSE_FILE%" pull || echo Pull completed
    echo.
)

REM Build
echo 🔨 Building containers...
echo Compose file: %COMPOSE_FILE%
echo Service: %SERVICE%
if defined NO_CACHE (
    echo No cache: Yes
) else (
    echo No cache: No
)
echo.

if "%SERVICE%"=="all" (
    docker compose -f "%COMPOSE_FILE%" build %NO_CACHE%
) else (
    docker compose -f "%COMPOSE_FILE%" build %NO_CACHE% %SERVICE%
)

if errorlevel 1 (
    echo.
    echo ❌ Build fallita!
    exit /b 1
)

echo.
echo ✅ Build completata!
echo.
echo Comandi utili:
echo   docker compose -f %COMPOSE_FILE% up -d        # Avvia servizi
echo   docker compose -f %COMPOSE_FILE% logs -f      # Visualizza logs
echo   docker compose -f %COMPOSE_FILE% ps           # Stato servizi
echo   docker compose -f %COMPOSE_FILE% down          # Stop servizi
echo.

endlocal

