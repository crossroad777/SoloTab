"""
run_ga_retrain.py - 全7ドメインのGradient Accumulation再学習
batch_size=1, accumulation_steps=4, AMP無し

3DS F1を超えないドメインは自動でおかわり（最大3回）
"""
import subprocess, sys, time, re, os

DOMAINS = [
    "martin_finger",
    "taylor_finger",
    "luthier_finger",
    "martin_pick",
    "taylor_pick",
    "luthier_pick",
    "gibson_thumb",
]

# 3DS (Step 6) の各ドメインBest F1 — これを超えることが目標
TARGET_F1 = {
    "martin_finger": 0.7813,
    "taylor_finger": 0.7743,
    "luthier_finger": 0.7817,
    "martin_pick": 0.7929,
    "taylor_pick": 0.7906,
    "luthier_pick": 0.7967,
    "gibson_thumb": 0.7891,
}

MAX_ROUNDS = 3  # おかわり最大回数

LOG_DIR = r"d:\Music\nextchord-solotab\music-transcription\python\_processed_guitarset_data\training_output"


def get_best_f1(domain):
    """ログファイルからGA学習のBest F1を取得"""
    log_path = os.path.join(LOG_DIR, f"finetuned_{domain}_multitask_3ds_ga", "training_log.txt")
    if not os.path.exists(log_path):
        return 0.0
    best = 0.0
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = re.search(r"F1:\s*([\d.]+)", line)
            if m:
                val = float(m.group(1))
                if val > best:
                    best = val
    return best


def run_domain(domain, round_num, idx, total):
    """1ドメインの学習を実行"""
    tag = f"R{round_num}" if round_num > 1 else ""
    print(f"\n{'='*60}")
    print(f"  [{idx}/{total}] {domain} {tag}")
    print(f"  Target: F1 > {TARGET_F1.get(domain, '?')}")
    print(f"{'='*60}")

    start_t = time.time()
    cmd = [
        sys.executable,
        "backend/train/train_gaps_multitask.py",
        "--domain", domain,
        "--include-agpt",
        "--ga-retrain",
        "--patience", "15",
    ]

    result = subprocess.run(cmd, cwd=r"d:\Music\nextchord-solotab")
    elapsed = (time.time() - start_t) / 60

    best_f1 = get_best_f1(domain)
    target = TARGET_F1.get(domain, 0)
    passed = best_f1 > target

    status = "OK" if result.returncode == 0 else "FAIL"
    mark = "[OK]" if passed else "[NG]"
    print(f"  {domain}: {status} ({elapsed:.1f} min) | Best F1={best_f1:.4f} vs 3DS={target:.4f} [{mark}]")

    return domain, status, elapsed, best_f1, passed


def main():
    print("=" * 60)
    print("  GA Retrain - All 7 Domains")
    print("  batch_size=1, accumulation_steps=4, AMP=off")
    print("  3DS F1を超えないドメインは自動おかわり（最大3回）")
    print("=" * 60)

    all_results = []
    total_start = time.time()

    # --- Round 1: 全7ドメイン ---
    pending = list(DOMAINS)
    round_num = 1

    while pending and round_num <= MAX_ROUNDS:
        if round_num > 1:
            print(f"\n{'#'*60}")
            print(f"  おかわり Round {round_num}: {len(pending)} ドメイン")
            print(f"  {', '.join(pending)}")
            print(f"{'#'*60}")

        failed_domains = []
        for i, domain in enumerate(pending):
            # おかわり: 前回のログ/モデルをリセットして最初からやり直す
            if round_num > 1:
                ga_dir = os.path.join(LOG_DIR, f"finetuned_{domain}_multitask_3ds_ga")
                for fname in ["training_log.txt", "best_model.pth"]:
                    fpath = os.path.join(ga_dir, fname)
                    if os.path.exists(fpath):
                        os.remove(fpath)
                print(f"  [Reset] {domain}: cleared log/model for retry")

            domain, status, elapsed, best_f1, passed = run_domain(
                domain, round_num, i + 1, len(pending)
            )
            all_results.append((domain, round_num, status, elapsed, best_f1, passed))

            if not passed:
                failed_domains.append(domain)

        pending = failed_domains
        round_num += 1

    # --- Summary ---
    total_elapsed = (time.time() - total_start) / 60

    print(f"\n{'='*60}")
    print("  GA Retrain Final Summary")
    print(f"{'='*60}")
    for domain, rnd, status, elapsed, best_f1, passed in all_results:
        target = TARGET_F1.get(domain, 0)
        mark = "[OK]" if passed else "[NG]"
        print(f"  {domain} R{rnd}: {status} ({elapsed:.1f}min) F1={best_f1:.4f} vs 3DS={target:.4f} [{mark}]")

    if pending:
        print(f"\n  [WARN] {MAX_ROUNDS}回おかわりしても超えなかったドメイン: {', '.join(pending)}")
    else:
        print(f"\n  [OK] 全ドメイン3DS超え達成！")

    print(f"\n  Total: {total_elapsed:.1f} min ({total_elapsed/60:.1f} hrs)")


if __name__ == "__main__":
    main()
