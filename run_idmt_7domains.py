"""
run_idmt_7domains.py — 7ドメイン IDMT混合学習
================================================================
GuitarSet + GAPS + Synth V2 + IDMT-SMT-V2 で各ドメインを3エポック学習。
出力suffix: multitask_4ds（既存モデル上書きなし）

Usage:
    python run_idmt_7domains.py
"""
import subprocess, sys, os, time

DOMAINS = [
    "martin_finger", "taylor_finger", "luthier_finger",
    "martin_pick", "taylor_pick", "luthier_pick",
    "gibson_thumb",
]

EPOCHS = 3
PATIENCE = 3
CWD = r"d:\Music\nextchord-solotab"


def main():
    print(f"\n{'='*60}")
    print(f"  7-Domain IDMT Mixed Training")
    print(f"  Epochs: {EPOCHS}, Patience: {PATIENCE}")
    print(f"  Datasets: GuitarSet + GAPS + Synth V2 + IDMT-SMT-V2")
    print(f"  Output suffix: multitask_4ds")
    print(f"{'='*60}\n")

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
            "--include-idmt",
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
