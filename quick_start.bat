@echo off
title SoloTab - Quick Start
echo =======================================
echo  SoloTab - Quick Start
echo =======================================
echo.

:: ========================================
:: Step 0: Clean up any existing processes
:: ========================================
echo [0/2] Cleaning up old processes...

:: Kill any process listening on port 8001 (backend)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8001" ^| findstr "LISTENING" 2^>nul') do (
    taskkill /F /PID %%a /T >nul 2>&1
)
:: Kill any process listening on port 5174 (frontend)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5174" ^| findstr "LISTENING" 2^>nul') do (
    taskkill /F /PID %%a /T >nul 2>&1
)
:: Small wait for ports to be released
timeout /t 2 /nobreak >nul

:: Verify port 8001 is free
netstat -ano | findstr ":8001" | findstr "LISTENING" >nul 2>&1
if %errorlevel%==0 (
    echo [WARN] Port 8001 is still in use. Retrying cleanup...
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8001" ^| findstr "LISTENING" 2^>nul') do (
        taskkill /F /PID %%a /T >nul 2>&1
    )
    timeout /t 2 /nobreak >nul
)

:: ========================================
:: Step 1: Start Backend
:: ========================================
echo [1/2] Starting Backend...
cd /d "D:\Music\nextchord-solotab\backend"
set PYTHONIOENCODING=utf-8
set TF_ENABLE_ONEDNN_OPTS=0
set TF_CPP_MIN_LOG_LEVEL=3
start "Backend (Port 8001)" cmd /c "chcp 65001 >nul & "D:\Music\nextchord\venv312\Scripts\python.exe" -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload --reload-dir . --reload-exclude uploads --reload-exclude __pycache__ --reload-exclude logs --reload-exclude ground_truth --reload-exclude benchmark --reload-exclude train"

:: ========================================
:: Step 2: Start Frontend
:: ========================================
echo [2/2] Starting Frontend...
cd /d "D:\Music\nextchord-solotab\frontend"
start "Frontend (localhost:5174)" cmd /c "npm run dev -- --port 5174 --strictPort"

:: ========================================
:: Open browser
:: ========================================
echo.
echo Waiting for servers to start...
timeout /t 4 /nobreak >nul
start http://localhost:5174/

echo.
echo =======================================
echo  SoloTab is running!
echo  Backend:  http://localhost:8001
echo  Frontend: http://localhost:5174
echo =======================================
echo Close this window at any time.
