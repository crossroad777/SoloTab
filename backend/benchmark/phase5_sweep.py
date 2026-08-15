import os
import sys
import json
import time
import subprocess
import re

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def run_bench(vt):
    print(f"\n==================================================")
    print(f" EVALUATING VOTE_THRESHOLD = {vt} / 7 (Fast MoE)")
    print(f"==================================================")
    env = os.environ.copy()
    env["SOLOTAB_MOE_VOTE_THRESHOLD"] = str(vt)
    
    t0 = time.time()
    # 1. Run E2E benchmark
    proc = subprocess.run(
        [sys.executable, "e2e_pipeline_benchmark.py"],
        cwd=".",
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env
    )
    e2e_out = proc.stdout
    
    pitch_f1 = 0.0
    pitch_p = 0.0
    pitch_r = 0.0
    string_acc = 0.0
    
    m_f1 = re.search(r"Overall Pitch F1 \(E2E Pipeline\)\s*:\s*([0-9\.]+)\s*\(P:\s*([0-9\.]+),\s*R:\s*([0-9\.]+)\)", e2e_out)
    if m_f1:
        pitch_f1 = float(m_f1.group(1))
        pitch_p = float(m_f1.group(2))
        pitch_r = float(m_f1.group(3))
        
    m_str = re.search(r"Overall String Accuracy\s*:\s*([0-9\.]+)", e2e_out)
    if m_str:
        string_acc = float(m_str.group(1))
        
    # 2. Run pitch error analysis (Category A)
    proc_a = subprocess.run(
        [sys.executable, "pitch_error_analysis.py"],
        cwd=".",
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env
    )
    a_out = proc_a.stdout
    
    a1, a2, a3 = 0, 0, 0
    m_a1 = re.search(r"Total A1 \(False Negative\)\s*:\s*(\d+)", a_out)
    if m_a1: a1 = int(m_a1.group(1))
    m_a2 = re.search(r"Total A2 \(False Positive\)\s*:\s*(\d+)", a_out)
    if m_a2: a2 = int(m_a2.group(1))
    m_a3 = re.search(r"Total A3 \(Pitch Mismatch\)\s*:\s*(\d+)", a_out)
    if m_a3: a3 = int(m_a3.group(1))
    
    elapsed = time.time() - t0
    
    res = {
        "vote_threshold": vt,
        "pitch_f1": pitch_f1,
        "pitch_precision": pitch_p,
        "pitch_recall": pitch_r,
        "string_accuracy": string_acc,
        "a1_false_negative": a1,
        "a2_false_positive": a2,
        "a3_pitch_mismatch": a3,
        "elapsed_sec": elapsed
    }
    
    print(f"[Results for vote_threshold={vt}]")
    print(f"  Pitch F1       : {pitch_f1:.4f} (P: {pitch_p:.4f}, R: {pitch_r:.4f})")
    print(f"  String Accuracy: {string_acc:.4f} ({string_acc*100:.2f}%)")
    print(f"  A1 (Missed)    : {a1}")
    print(f"  A2 (Spurious)  : {a2}")
    print(f"  A3 (Mismatch)  : {a3}")
    print(f"  Time           : {elapsed:.1f}s")
    
    return res

if __name__ == "__main__":
    thresholds = [4, 5, 6]
    all_results = []
    
    for vt in thresholds:
        res = run_bench(vt)
        all_results.append(res)
        
    with open("phase5_sweep_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
        
    print("\n==================================================")
    print(" PHASE 5 SWEEP SUMMARY ")
    print("==================================================")
    print(f"{'VoteThres':<10} | {'Pitch F1':<10} | {'String Acc':<12} | {'A1 (FN)':<8} | {'A2 (FP)':<8} | {'A3 (Mis)':<8}")
    print("-" * 65)
    for r in all_results:
        print(f"{r['vote_threshold']:<10} | {r['pitch_f1']:<10.4f} | {r['string_accuracy']*100:<11.2f}% | {r['a1_false_negative']:<8} | {r['a2_false_positive']:<8} | {r['a3_pitch_mismatch']:<8}")
