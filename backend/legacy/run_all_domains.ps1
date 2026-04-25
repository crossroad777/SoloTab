$PYTHON = "D:\Music\nextchord\venv312\Scripts\python.exe"

$datasets = @(
    "luthier_finger",
    "martin_pick",
    "taylor_pick",
    "luthier_pick"
)

foreach ($ds in $datasets) {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host " Starting Automated Training for: $ds" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    
    $process = Start-Process -FilePath $PYTHON -ArgumentList "train_domain.py", "--dataset", "$ds" -NoNewWindow -PassThru -Wait
    
    if ($process.ExitCode -ne 0) {
        Write-Host "Warning: Training for $ds exited with code $($process.ExitCode). However, moving to next to prevent blockage..." -ForegroundColor Yellow
    } else {
        Write-Host "Successfully completed training pipeline for $ds." -ForegroundColor Green
    }
}

Write-Host "All specified domain trainings have concluded." -ForegroundColor Green
