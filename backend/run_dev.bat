@echo off
REM Run backend and frontend servers for development
REM This script starts:
REM  - Terminal 1: Backend FastAPI server (port 8000)
REM  - Terminal 2: Frontend HTTP server (port 3000)

echo Starting Sign Language Web Development Servers...
echo.

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0
cd /d %SCRIPT_DIR%

REM Terminal 1: Backend server
echo [Terminal 1] Starting Backend Server...
start cmd /k "cd %SCRIPT_DIR% && sign\Scripts\activate.bat && set SL_DEPLOYMENT_MODE=wlasl2000 && uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000 --reload"

REM Wait a moment for backend to start
timeout /t 3 /nobreak

REM Terminal 2: Frontend server
echo [Terminal 2] Starting Frontend Server...
cd /d %SCRIPT_DIR%..\frontend
start cmd /k "cd %SCRIPT_DIR%..\frontend && python -m http.server 3000"

echo.
echo Servers started:
echo  - Backend: http://localhost:8000 (with reload on changes)
echo  - Frontend: http://localhost:3000
echo.
echo Press Ctrl+C in each terminal to stop the servers.
