@echo off
title SoloTab - Quick Start
echo =======================================
echo  SoloTab - ローカル起動スクリプト
echo =======================================
echo.

cd /d "%~dp0"
echo [1/2] サーバーを起動しています...
:: バックグラウンドで python start_servers.py を起動し、標準出力を見せる
start "SoloTab Server" cmd /c "python start_servers.py"

echo [2/2] 接続を待っています...
timeout /t 5 /nobreak >nul

echo ブラウザを起動します...
start http://localhost:5174/

echo.
echo =======================================
echo  SoloTab はローカル環境で起動しました！
echo =======================================
echo  バックエンド: http://localhost:8002
echo  フロントエンド: http://localhost:5174
echo.
echo  このウィンドウは閉じて構いません。
echo  サーバーを終了するときは、新しく開いた黒い画面で [Ctrl+C] を押してください。
exit
