import os
import sys
import subprocess
import json
import time

def sweep_vote_threshold():
    print("==================================================")
    print(" SWEEPING VOTE THRESHOLD (4 to 7)")
    print("==================================================")
    
    thresholds = [4, 5, 6, 7]
    results = {}
    
    for thresh in thresholds:
        print(f"\n>>> Running Benchmark with VOTE_THRESH = {thresh} ...")
        env = os.environ.copy()
        env["VOTE_THRESH"] = str(thresh)
        
        cmd = [sys.executable, "e2e_pipeline_benchmark.py"]
        
        start_t = time.time()
        try:
            # We capture stdout to parse String Accuracy if it's printed there
            result = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True, encoding='utf-8')
            elapsed = time.time() - start_t
            
            # Read the results json output
            results_path = "detailed_benchmark_results.json"
            pitch_f1 = 0.0
            pitch_p = 0.0
            pitch_r = 0.0
            string_f1 = 0.0
            
            if os.path.exists(results_path):
                with open(results_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                overall = data.get("overall", {})
                pitch_f1 = overall.get("f1", 0.0)
                pitch_p = overall.get("precision", 0.0)
                pitch_r = overall.get("recall", 0.0)
            
            # Try to parse string accuracy from stdout
            for line in result.stdout.split('\n'):
                if "Overall String Accuracy" in line:
                    try:
                        string_f1 = float(line.split(':')[-1].strip())
                    except:
                        pass

            results[thresh] = {
                "Pitch F1": pitch_f1,
                "Precision": pitch_p,
                "Recall": pitch_r,
                "String Accuracy": string_f1,
                "Time": elapsed
            }
            print(f"  Result: Pitch F1 = {pitch_f1:.4f} (P: {pitch_p:.4f}, R: {pitch_r:.4f}), String Acc = {string_f1:.4f}")
                
        except subprocess.CalledProcessError as e:
            print(f"  Error running benchmark for thresh {thresh}: {e}")
            
    print("\n==================================================")
    print(" SWEEP SUMMARY")
    print("==================================================")
    best_thresh = None
    best_f1 = -1
    for t, r in results.items():
        print(f" Vote Threshold {t}/7: Pitch F1 = {r['Pitch F1']:.4f}, String Acc = {r['String Accuracy']:.4f}")
        if r['Pitch F1'] > best_f1:
            best_f1 = r['Pitch F1']
            best_thresh = t
            
    print(f"\n>>> Optimal Vote Threshold: {best_thresh} (Pitch F1 = {best_f1:.4f})")
    
    with open("sweep_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    sweep_vote_threshold()
