import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
import subprocess
import re

DAY2_BASELINE = {
    "A1": 107,
    "A2": 195,
    "A3": 179
}

def recalc_category_a():
    print("==================================================")
    print(" RE-CALCULATING CATEGORY A (Pitch Error)")
    print("==================================================")
    
    cmd = [sys.executable, "pitch_error_analysis.py"]
    
    try:
        print("Running pitch_error_analysis.py (this runs the E2E pipeline under the hood)...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        
        # Parse the output
        a1, a2, a3 = None, None, None
        
        for line in result.stdout.split('\n'):
            if "Total A1 (False Negative):" in line:
                a1 = int(re.search(r':\s*(\d+)', line).group(1))
            elif "Total A2 (False Positive):" in line:
                a2 = int(re.search(r':\s*(\d+)', line).group(1))
            elif "Total A3 (Pitch Mismatch):" in line:
                a3 = int(re.search(r':\s*(\d+)', line).group(1))
                
        if a1 is not None and a2 is not None and a3 is not None:
            print("\n==================================================")
            print(" CATEGORY A RESULTS & DIFF (vs Day 2 Baseline)")
            print("==================================================")
            
            diff_a1 = a1 - DAY2_BASELINE["A1"]
            diff_a2 = a2 - DAY2_BASELINE["A2"]
            diff_a3 = a3 - DAY2_BASELINE["A3"]
            
            def sign(n):
                return f"+{n}" if n > 0 else str(n)
                
            print(f"Total A1 (False Negative) : {a1} ({sign(diff_a1)})")
            print(f"Total A2 (False Positive) : {a2} ({sign(diff_a2)})")
            print(f"Total A3 (Pitch Mismatch) : {a3} ({sign(diff_a3)})")
            
            # Print the rest of the detailed output
            detail_start = result.stdout.find("--- Distribution by Track ---")
            if detail_start != -1:
                print("\n" + result.stdout[detail_start:].strip())
        else:
            print("Error parsing the output. Make sure pitch_error_analysis.py ran successfully.")
            print("Raw Output:")
            print(result.stdout)
            
    except subprocess.CalledProcessError as e:
        print(f"Error running pitch_error_analysis.py: {e}")
        print(e.stderr)

if __name__ == "__main__":
    recalc_category_a()
