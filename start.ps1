# SoloTab 起動スクリプト
# バックエンドとフロントエンドを一括起動します

$ErrorActionPreference = "Continue"
$PYTHON = "D:\Music\nextchord\venv312\Scripts\python.exe"
$BACKEND_DIR = "D:\Music\nextchord-solotab\backend"
$FRONTEND_DIR = "D:\Music\nextchord-solotab\frontend"
$BACKEND_PORT = 8001

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  SoloTab - Starting..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# --- 1. 古いプロセスの停止 ---
Write-Host "[1/3] Checking for existing processes..." -ForegroundColor Yellow

$existing = netstat -ano 2>$null | Select-String ":$BACKEND_PORT.*LISTENING"
if ($existing) {
    $pids = $existing | ForEach-Object {
        ($_ -replace '.*LISTENING\s+', '').Trim()
    } | Sort-Object -Unique
    foreach ($procId in $pids) {
        Write-Host "  Stopping old backend process (PID: $procId)..." -ForegroundColor Red
        try { taskkill /F /PID $procId 2>&1 | Out-Null } catch { }
    }
    Start-Sleep -Seconds 1
    Write-Host "  Old processes stopped." -ForegroundColor Green
} else {
    Write-Host "  No existing processes on port $BACKEND_PORT." -ForegroundColor Green
}

# --- 2. バックエンド起動 ---
Write-Host ""
Write-Host "[2/3] Starting backend on port $BACKEND_PORT..." -ForegroundColor Yellow

$backendJob = Start-Process -FilePath $PYTHON `
    -ArgumentList "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$BACKEND_PORT", "--reload", "--reload-dir", ".", "--reload-exclude", "uploads", "--reload-exclude", "__pycache__", "--reload-exclude", "logs", "--reload-exclude", "ground_truth", "--reload-exclude", "benchmark", "--reload-exclude", "train" `
    -WorkingDirectory $BACKEND_DIR `
    -PassThru -NoNewWindow

Write-Host "  Backend started (PID: $($backendJob.Id))" -ForegroundColor Green

# バックエンドが起動するまで待機
Write-Host "  Waiting for backend to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 5
Write-Host "  Backend is assumed ready!" -ForegroundColor Green

# --- 3. フロントエンド起動 ---
Write-Host ""
Write-Host "[3/3] Starting frontend..." -ForegroundColor Yellow

$npmPath = "npm.cmd"

$frontendJob = Start-Process -FilePath $npmPath `
    -ArgumentList "run", "dev" `
    -WorkingDirectory $FRONTEND_DIR `
    -PassThru -NoNewWindow

Start-Sleep -Seconds 3

Write-Host "  Frontend started (PID: $($frontendJob.Id))" -ForegroundColor Green

# --- 完了 ---
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  SoloTab is running!" -ForegroundColor Green
Write-Host "  Frontend: http://localhost:5174/" -ForegroundColor Cyan
Write-Host "  Backend:  http://localhost:$BACKEND_PORT/" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop all processes." -ForegroundColor DarkGray

# プロセス終了待機
try {
    while ($true) {
        if ($backendJob.HasExited -and $frontendJob.HasExited) {
            Write-Host "Both processes have exited." -ForegroundColor Red
            break
        }
        Start-Sleep -Seconds 2
    }
} finally {
    Write-Host "Stopping processes..." -ForegroundColor Yellow
    if (-not $backendJob.HasExited) { Stop-Process -Id $backendJob.Id -Force -ErrorAction SilentlyContinue }
    if (-not $frontendJob.HasExited) { Stop-Process -Id $frontendJob.Id -Force -ErrorAction SilentlyContinue }
    Write-Host "All processes stopped." -ForegroundColor Green
}
