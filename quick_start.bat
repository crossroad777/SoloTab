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

echo [3/3] サーバーの待受準備を待機中...
timeout /t 3 /nobreak >nul

echo ブラウザを起動します...
start http://localhost:5174/

echo.
echo ===================================================
echo   SoloTab を最新状態で起動しました！
echo ===================================================
echo   フロントエンド: http://localhost:5174
echo   バックエンド:   http://localhost:8000
echo.
echo   このウィンドウは閉じても問題ありません。
echo   サーバー停止は黒いコンソール画面で [Ctrl+C] を押してください。
exit
