"""
overnight_training.py - 夜間一括学習スクリプト
================================================================
3つのタスクを順番に実行:
  1. 2段階テクニックCNN V4 (IDMT → aGPTset fine-tune)
  2. テクニックCNN V2 Extended (200 epochs, IDMT単独)
  3. Optuna Viterbi DP 重み最適化
全て別ファイルに保存。既存モデルは一切変更しない。
================================================================
"""
import subprocess, sys, os, time, datetime

BACKEND = r"d:\Music\nextchord-solotab\backend"

def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def run_task(name, cmd, cwd, timeout_hours=3):
    log(f"{'='*60}")
    log(f"START: {name}")
    log(f"{'='*60}")
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, cwd=cwd, shell=True,
            timeout=timeout_hours * 3600,
        )
        elapsed = (time.time() - t0) / 60
        status = "SUCCESS" if result.returncode == 0 else f"FAILED (exit={result.returncode})"
        log(f"DONE: {name} - {status} ({elapsed:.0f}min)")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        elapsed = (time.time() - t0) / 60
        log(f"TIMEOUT: {name} ({elapsed:.0f}min)")
        return False
    except Exception as e:
        log(f"ERROR: {name}: {e}")
        return False

results = []

# ──────────────────────────────────────────────
# Task 1: テクニックCNN V4 Stage 1 (IDMT-only, 100 epochs)
# ──────────────────────────────────────────────
ok = run_task(
    "V4-Stage1: IDMT 100ep → technique_cnn_v4_stage1.pth",
    'python -u train_technique_cnn.py --epochs 100 --output models/technique_cnn_v4_stage1.pth',
    BACKEND, timeout_hours=2
)
results.append(("V4-Stage1", ok))

# ──────────────────────────────────────────────
# Task 2: テクニックCNN V4 Stage 2 (aGPTset fine-tune)
# ──────────────────────────────────────────────
if ok:
    ok2 = run_task(
        "V4-Stage2: aGPTset fine-tune → technique_cnn_v4.pth",
        'python -u train_technique_cnn_v4_finetune.py',
        BACKEND, timeout_hours=2.5
    )
    results.append(("V4-Stage2", ok2))
else:
    log("SKIP V4-Stage2 (Stage1 failed)")
    results.append(("V4-Stage2", False))

# ──────────────────────────────────────────────
# Task 3: テクニックCNN V2 Extended (200 epochs, IDMT-only)
# ──────────────────────────────────────────────
ok3 = run_task(
    "V2-Extended: IDMT 200ep → technique_cnn_v2_extended.pth",
    'python -u train_technique_cnn.py --epochs 200 --output models/technique_cnn_v2_extended.pth',
    BACKEND, timeout_hours=2
)
results.append(("V2-Extended", ok3))

# ──────────────────────────────────────────────
# Task 4: Optuna Viterbi Optimization (500 trials)
# ──────────────────────────────────────────────
optuna_script = os.path.join(BACKEND, "gp5_training", "optuna_v3_pure_viterbi.py")
if os.path.exists(optuna_script):
    ok4 = run_task(
        "Optuna Viterbi 500 trials → optimized_weights_v3.json",
        'python -u gp5_training/optuna_v3_pure_viterbi.py',
        BACKEND, timeout_hours=2.5
    )
    results.append(("Optuna", ok4))
else:
    log("Optuna script not found, skipping")
    results.append(("Optuna", False))

# ──────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────
log("")
log("=" * 60)
log("OVERNIGHT TRAINING COMPLETE")
log("=" * 60)
for name, ok in results:
    status = "✓ SUCCESS" if ok else "✗ FAILED"
    log(f"  {name:20s} {status}")
log("")
log("Output files:")
log("  models/technique_cnn_v4_stage1.pth   (IDMT Stage 1)")
log("  models/technique_cnn_v4.pth          (2-stage fine-tuned)")
log("  models/technique_cnn_v2_extended.pth  (200ep IDMT)")
log("  optimized_weights_v3.json            (Optuna best weights)")
