@echo off
title SoloTab - Quick Start
echo =======================================
echo  SoloTab - Local Startup Script
echo =======================================
echo.

cd /d "%~dp0"
echo [1/2] Starting server...
start "SoloTab Server" cmd /k "python start_servers.py"

echo [2/2] Waiting for connections...
timeout /t 5 /nobreak >nul

echo Starting browser...
start http://localhost:5174/

echo.
echo =======================================
echo  SoloTab started locally!
echo =======================================
echo  Backend:  http://localhost:8002
echo  Frontend: http://localhost:5174
echo.
echo  You can safely close this window.
echo  To stop the server, press [Ctrl+C] in the black window.
exit
