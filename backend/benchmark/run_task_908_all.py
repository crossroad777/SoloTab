"""
backend/benchmark/run_task_908_all.py
=====================================
TASK-908: WAV不要の大規模統計的検証（3アプローチ統合実行）

Step 1: 大規模エルゴノミクス A/B テスト (N=1000)
Step 2: ラウンドトリップ・テスト (GP5 -> MIDI -> SynthWAV -> AMT -> SoloTab, N=50)
Step 3: ホールドアウト交叉検証 (Transformer V3, N=1000)
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from benchmark.run_task_908_step1 import run_step1_evaluation
from benchmark.run_task_908_step2 import run_step2_roundtrip
from benchmark.run_task_908_step3 import run_step3_evaluation

def main():
    print("=" * 70)
    print("【TASK-908: WAV不要の大規模統計的検証（3アプローチ） 開始】")
    print("=" * 70)
    
    t_start = time.time()
    
    # --- Step 1 ---
    print("\n>>> Running Step 1: 大規模エルゴノミクス A/B テスト (N=1000)...")
    res1 = run_step1_evaluation(sample_size=1000, seed=42)
    
    # --- Step 2 ---
    print("\n>>> Running Step 2: ラウンドトリップ・テスト (N=50)...")
    res2 = run_step2_roundtrip(sample_size=50, seed=42)
    
    # --- Step 3 ---
    print("\n>>> Running Step 3: ホールドアウト交叉検証 (Transformer V3, N=1000)...")
    res3 = run_step3_evaluation(sample_size=1000, seed=42)
    
    total_elapsed = round(time.time() - t_start, 2)
    
    final_output = {
        "task_id": "TASK-908",
        "title": "WAV不要の大規模統計的検証（3アプローチ）",
        "total_elapsed_seconds": total_elapsed,
        "step1_ergonomics_ab_test": res1,
        "step2_roundtrip_test": res2,
        "step3_holdout_cross_validation": res3
    }
    
    out_file = "backend/benchmark/task_908_final_statistical_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
        
    print("\n" + "=" * 70)
    print("【TASK-908 全検証完了】")
    print("=" * 70)
    print(f"結果保存先: {out_file}")
    print(json.dumps(final_output, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
