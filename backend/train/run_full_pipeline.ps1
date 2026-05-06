# run_full_pipeline.ps1
# ============================================================
# SoloTab 統合学習パイプライン — Step 1〜8 完全自動実行
# ============================================================
# 統一ルール: 25エポック / patience 7 / 毎Best保存
#
# 使い方:
#   powershell -ExecutionPolicy Bypass -File backend\train\run_full_pipeline.ps1
#
# 中断しても再実行すれば、各スクリプトのログresumeにより
# 完了済みステップは自動スキップされる。
#
# 特定ステップから再開:
#   powershell -ExecutionPolicy Bypass -File backend\train\run_full_pipeline.ps1 -StartStep 3
# ============================================================

param(
    [int]$StartStep = 1
)

$ErrorActionPreference = "Continue"
Set-Location "d:\Music\nextchord-solotab"

$EPOCHS = 9999
$PATIENCE = 10
$LOG_FILE = "d:\Music\nextchord-solotab\pipeline_progress.log"

$domains = @(
    "martin_finger",
    "taylor_finger",
    "gibson_thumb",
    "luthier_finger",
    "luthier_pick",
    "martin_pick",
    "taylor_pick"
)

function Log-Msg {
    param([string]$msg)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] $msg"
    Write-Host $line
    Add-Content -Path $LOG_FILE -Value $line -Encoding UTF8
}

function Run-Python {
    param([string]$cmd, [string]$label)
    Log-Msg "  RUN: $label"
    try {
        Invoke-Expression "python $cmd"
        $ec = $LASTEXITCODE
        if ($ec -eq 0) {
            Log-Msg "  OK: $label"
        } else {
            Log-Msg "  WARN: $label — exit code $ec"
        }
    } catch {
        Log-Msg "  ERROR: $label — $_"
    }
    # GPU VRAM解放
    Start-Sleep -Seconds 5
}

Log-Msg "========== PIPELINE START (from Step $StartStep) =========="

# ============================================================
# Step 1: martin_finger GAPS Multi-task 完走
# ============================================================
if ($StartStep -le 1) {
    Log-Msg "===== Step 1/8: martin_finger Multi-task (GAPS) ====="
    Run-Python "backend\train\train_gaps_multitask.py --domain martin_finger --epochs $EPOCHS --patience $PATIENCE" "martin_finger Multi-task"
    Log-Msg "Step 1 DONE"
}

# ============================================================
# Step 2: 全7ドメイン GuitarSet FT 完走
# ============================================================
if ($StartStep -le 2) {
    Log-Msg "===== Step 2/8: All-Domain GuitarSet FT ====="
    foreach ($d in $domains) {
        Run-Python "backend\train\train_guitarset_ft_all.py --domain $d --epochs $EPOCHS --patience $PATIENCE" "GuitarSet FT: $d"
    }
    Log-Msg "Step 2 DONE"
}

# ============================================================
# Step 3: 全7ドメイン GAPS Multi-task
# ============================================================
if ($StartStep -le 3) {
    Log-Msg "===== Step 3/8: All-Domain GAPS Multi-task ====="
    foreach ($d in $domains) {
        Run-Python "backend\train\train_gaps_multitask.py --domain $d --epochs $EPOCHS --patience $PATIENCE" "GAPS Multi-task: $d"
    }
    Log-Msg "Step 3 DONE"
}

# ============================================================
# Step 4: MoE統合評価（推論テスト）
# ============================================================
if ($StartStep -le 4) {
    Log-Msg "===== Step 4/8: MoE E2E Benchmark ====="
    Run-Python "backend\benchmark_e2e.py --max 10" "MoE E2E Benchmark (10 tracks)"
    Log-Msg "Step 4 DONE"
}

# ============================================================
# Step 5: AG-PT-set前処理 + 3データセット統合（martin_fingerで検証）
# ============================================================
if ($StartStep -le 5) {
    Log-Msg "===== Step 5/8: 3-Dataset Integration (martin_finger) ====="

    # AG-PT-set前処理（既に完了済みならスキップ）
    $agptProcessed = (Get-ChildItem "D:\Music\datasets\AG-PT-set\aGPTset\_processed\*_features.pt" -ErrorAction SilentlyContinue | Measure-Object).Count
    if ($agptProcessed -lt 100) {
        Log-Msg "  AG-PT-set preprocessing needed ($agptProcessed files found)"
        Run-Python "backend\train\preprocess_agpt.py" "AG-PT-set Preprocess"
    } else {
        Log-Msg "  AG-PT-set already preprocessed ($agptProcessed files)"
    }

    # 3データセット統合学習 (martin_finger)
    # --include-agpt フラグを使用（train_gaps_multitask.py に追加する）
    Run-Python "backend\train\train_gaps_multitask.py --domain martin_finger --epochs $EPOCHS --patience $PATIENCE --include-agpt" "3-Dataset: martin_finger"
    Log-Msg "Step 5 DONE"
}

# ============================================================
# Step 6: 全7ドメイン 3データセット統合
# ============================================================
if ($StartStep -le 6) {
    Log-Msg "===== Step 6/8: All-Domain 3-Dataset Integration ====="
    foreach ($d in $domains) {
        Run-Python "backend\train\train_gaps_multitask.py --domain $d --epochs $EPOCHS --patience $PATIENCE --include-agpt" "3-Dataset: $d"
    }
    Log-Msg "Step 6 DONE"
}

# ============================================================
# Step 7: 弦分類CNN学習
# ============================================================
if ($StartStep -le 7) {
    Log-Msg "===== Step 7/8: String Classifier CNN ====="
    Run-Python "backend\string_classifier.py" "String Classifier CNN"
    Log-Msg "Step 7 DONE"
}

# ============================================================
# Step 8: 運指LSTM学習（CNN弦確率統合版）
# ============================================================
if ($StartStep -le 8) {
    Log-Msg "===== Step 8/8: Fingering LSTM (with CNN probs) ====="
    Run-Python "backend\train_fingering_with_cnn.py" "Fingering LSTM"
    Log-Msg "Step 8 DONE"
}

# ============================================================
# 完了サマリー
# ============================================================
Log-Msg "========== PIPELINE FINISHED =========="

$outputDir = "d:\Music\nextchord-solotab\music-transcription\python\_processed_guitarset_data\training_output"

Write-Host ""
Write-Host "===== FINAL RESULTS ====="
Log-Msg "===== FINAL RESULTS ====="

foreach ($d in $domains) {
    $ftF1 = "N/A"; $mtF1 = "N/A"; $triF1 = "N/A"

    # GuitarSet FT
    $ftLog = Join-Path $outputDir "finetuned_${d}_guitarset_ft\training_log.txt"
    if (Test-Path $ftLog) {
        foreach ($line in (Get-Content $ftLog -Encoding UTF8)) {
            if ($line -match "Best F1:\s*([\d.]+)") { $ftF1 = $Matches[1] }
        }
    }

    # GAPS Multi-task
    $mtLog = Join-Path $outputDir "finetuned_${d}_multitask\training_log.txt"
    if (Test-Path $mtLog) {
        foreach ($line in (Get-Content $mtLog -Encoding UTF8)) {
            if ($line -match "Best F1:\s*([\d.]+)") { $mtF1 = $Matches[1] }
        }
    }

    # 3DS Multi-task
    $triLog = Join-Path $outputDir "finetuned_${d}_multitask_3ds\training_log.txt"
    if (Test-Path $triLog) {
        foreach ($line in (Get-Content $triLog -Encoding UTF8)) {
            if ($line -match "Best F1:\s*([\d.]+)") { $triF1 = $Matches[1] }
        }
    }

    $result = "$d | GS-FT: $ftF1 | Multi-task: $mtF1 | 3DS: $triF1"
    Write-Host $result
    Log-Msg "RESULT: $result"
}

# 弦分類器
$scPath = "d:\Music\nextchord-solotab\backend\string_classifier.pth"
if (Test-Path $scPath) {
    Write-Host "String Classifier: TRAINED"
    Log-Msg "RESULT: String Classifier: TRAINED"
} else {
    Write-Host "String Classifier: NOT YET"
    Log-Msg "RESULT: String Classifier: NOT YET"
}

# 運指LSTM
$flPath = "d:\Music\nextchord-solotab\backend\fingering_lstm.pth"
if (Test-Path $flPath) {
    $flSize = [math]::Round((Get-Item $flPath).Length / 1KB, 1)
    Write-Host "Fingering LSTM: TRAINED (${flSize}KB)"
    Log-Msg "RESULT: Fingering LSTM: TRAINED (${flSize}KB)"
} else {
    Write-Host "Fingering LSTM: NOT YET"
    Log-Msg "RESULT: Fingering LSTM: NOT YET"
}

Write-Host ""
Write-Host "Full log: $LOG_FILE"
