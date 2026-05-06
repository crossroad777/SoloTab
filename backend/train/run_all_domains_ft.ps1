# run_all_domains_ft.ps1
# Train all 7 domains sequentially with extended epochs.
# martin_finger is placed last (already running in another process).
# The parse_log function auto-skips completed domains.

$ErrorActionPreference = "Continue"
Set-Location "d:\Music\nextchord-solotab"

$domains = @(
    "taylor_finger",
    "luthier_finger",
    "martin_pick",
    "taylor_pick",
    "luthier_pick",
    "gibson_thumb",
    "martin_finger"
)

$epochs = 9999
$patience = 10

Write-Host "=========================================="
Write-Host "  All-Domain Extended FT (epochs=$epochs, patience=$patience)"
Write-Host "  Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "=========================================="

foreach ($d in $domains) {
    Write-Host ""
    Write-Host "--- Starting: $d ---"
    python backend\train\train_guitarset_ft_all.py --domain $d --epochs $epochs --patience $patience
    Write-Host "--- Finished: $d ---"
    Write-Host ""
}

Write-Host "=========================================="
Write-Host "  All domains complete!"
Write-Host "  Finished: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "=========================================="
