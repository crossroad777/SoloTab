"""
run_all_7domains.py — 7ドメイン全てをSynth V2混合で再学習
================================================================
GuitarSet + GAPS + Synth V2 で各ドメインを100エポック（patience=15）学習する。
中断・再開対応：ログにearly-stop記録があるドメインは自動スキップ。

Usage:
    python run_all_7domains.py
    python run_all_7domains.py --fresh   # ログをクリアして全ドメイン再実行
"""
import subprocess
import sys
import os
import shutil
import time

DOMAINS = [
    "martin_finger",
    "taylor_finger",
    "luthier_finger",
    "martin_pick",
    "taylor_pick",
    "luthier_pick",
    "gibson_thumb",
]

EPOCHS = 3
PATIENCE = 3
CWD = r"d:\Music\nextchord-solotab"
TRAINING_OUTPUT = os.path.join(
    CWD, "music-transcription", "python",
    "_processed_guitarset_data", "training_output"
)

def reset_domain(domain):
    """ログクリア + バックアップモデルから復元"""
    ga_dir = os.path.join(TRAINING_OUTPUT, f"finetuned_{domain}_multitask_3ds_ga")
    backup_dir = os.path.join(ga_dir, "_backup_pre_synth_experiment")
    backup_model = os.path.join(backup_dir, "best_model.pth")
    log_path = os.path.join(ga_dir, "training_log.txt")
    best_path = os.path.join(ga_dir, "best_model.pth")
    
    # バックアップがなければ3DSモデルからコピー
    if not os.path.exists(backup_model):
        src_3ds = os.path.join(TRAINING_OUTPUT, f"finetuned_{domain}_multitask_3ds", "best_model.pth")
        if os.path.exists(src_3ds):
            os.makedirs(backup_dir, exist_ok=True)
            shutil.copy2(src_3ds, backup_model)
            print(f"    Created backup from 3DS model")
        else:
            print(f"    [WARN] No source model found for {domain}")
            return False
    
    # ログクリア
    if os.path.exists(log_path):
        os.remove(log_path)
    
    # バックアップから復元
    os.makedirs(ga_dir, exist_ok=True)
    shutil.copy2(backup_model, best_path)
    print(f"    Reset: {domain} (backup restored)")
    return True


def main():
    fresh = "--fresh" in sys.argv
    
    print(f"\n{'='*60}")
    print(f"  7-Domain Synth V2 Mixed Training")
    print(f"  Epochs: {EPOCHS}, Patience: {PATIENCE}")
    print(f"  Datasets: GuitarSet + GAPS + Synth V2")
    print(f"  Fresh: {fresh}")
    print(f"{'='*60}\n")

    if fresh:
        print("  Resetting all domains...")
        for domain in DOMAINS:
            reset_domain(domain)
        print()

    results = []
    t_total = time.time()

    for i, domain in enumerate(DOMAINS, 1):
        print(f"\n{'='*60}")
        print(f"  [{i}/7] {domain}")
        print(f"{'='*60}")
        t0 = time.time()

        cmd = [
            sys.executable,
            "backend/train/train_gaps_multitask.py",
            "--domain", domain,
            "--epochs", str(EPOCHS),
            "--patience", str(PATIENCE),
            "--include-synth",
            "--ga-retrain",
        ]

        result = subprocess.run(cmd, cwd=CWD)
        elapsed = (time.time() - t0) / 60

        status = "OK" if result.returncode == 0 else f"FAIL(code={result.returncode})"
        results.append((domain, status, f"{elapsed:.1f} min"))
        print(f"\n  [{status}] {domain} - {elapsed:.1f} min")

    total_min = (time.time() - t_total) / 60
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS  (Total: {total_min:.1f} min)")
    print(f"{'='*60}")
    for domain, status, elapsed in results:
        print(f"  {domain:25s} {status:10s} {elapsed}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
