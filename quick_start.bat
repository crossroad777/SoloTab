@echo off
title SoloTab - Quick Start
echo =======================================
echo  SoloTab - Quick Start (Lightning Fast)
echo =======================================

:: Stop existing uvicorn processes on port 8001 to prevent conflicts
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8001" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)

:: Start Backend (Python)
echo [1/2] Starting Backend...
cd /d "D:\Music\nextchord-solotab\backend"
start "Backend (Port 8001)" "D:\Music\nextchord\venv312\Scripts\python.exe" -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload

:: Start Frontend (NPM)
echo [2/2] Starting Frontend...
cd /d "D:\Music\nextchord-solotab\frontend"
start "Frontend (localhost:5174)" cmd /c "npm run dev"

echo.
echo Launching browser...
timeout /t 2 >nul
start http://localhost:5174/

echo Done! The app is starting up in separate windows.
echo Close this window at any time.
