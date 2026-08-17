@echo off
chcp 65001 >nul
title SoloTab - 強力リフレッシュ起動
echo ===================================================
echo   SoloTab - Powerful Refresh ^& Local Startup
echo ===================================================
echo.

cd /d "%~dp0"

echo [1/3] 古い残存プロセスを強制クリーンアップ中...
for %%P in (8000 8001 8002 5173 5174 5175) do (
    for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":%%P" ^| findstr "LISTENING"') do (
        taskkill /F /PID %%a >nul 2>&1
    )
)

echo [2/3] サーバー（バックエンド ^& フロントエンド）を起動中...
start "SoloTab Server [Port 8000 / 5174]" cmd /k "python start_servers.py"

echo [3/3] サーバーの待受準備を確認中...
set count=0
:WAIT_LOOP
timeout /t 1 /nobreak >nul
set /a count+=1

powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri 'http://localhost:5174/' -UseBasicParsing -TimeoutSec 1; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
if %ERRORLEVEL% equ 0 goto READY

if %count% geq 15 goto READY
goto WAIT_LOOP

:READY
echo.
echo ===================================================
echo   SoloTab が正常に起動しました！ブラウザを開きます...
echo ===================================================
echo   フロントエンド: http://localhost:5174
echo   バックエンド:   http://localhost:8000
echo.
echo   このウィンドウは閉じても問題ありません。
echo   サーバー停止は黒いコンソール画面で [Ctrl+C] を押してください。
echo.

start http://localhost:5174/
timeout /t 3 /nobreak >nul
exit
