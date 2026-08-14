"""
GuitarSet Benchmark v2 — 論文§8 全手法統合評価
================================================
ファイルごとに処理してViterbi DPの爆発を防ぐ。
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import sys, json, os, time
import numpy as np

sys.path.insert(0, r'D:\Music\nextchord-solotab\backend')

import warnings
warnings.filterwarnings('ignore')

from string_assigner import assign_strings_dp, get_possible_positions
from solotab_utils import STANDARD_TUNING

GUITARSET_DIR = r"D:\Music\nextchord-solotab\datasets\GuitarSet"
NUM_STRINGS = 6


def load_guitarset_by_file():
    """Load JAMS files and return per-file note lists."""
    import jams
    
    jams_files = sorted([
        f for f in os.listdir(GUITARSET_DIR) if f.endswith('.jams')
    ])
    
    file_notes = {}
    total_notes = 0
    
    for jf in jams_files:
        path = os.path.join(GUITARSET_DIR, jf)
        try:
            jam = jams.load(path)
        except Exception:
            continue
        
        string_annotations = [a for a in jam.annotations if a.namespace == 'note_midi']
        if len(string_annotations) != 6:
            continue
        
        notes = []
        for str_idx, ann in enumerate(string_annotations):
            # GuitarSet: annotation 0 = lowest string (E2), 5 = highest (E4)
            # Our convention: String 1 = highest (E4), String 6 = lowest (E2)
            string_num = 6 - str_idx  # map: 0->6, 1->5, 2->4, 3->3, 4->2, 5->1
            for obs in ann.data:
                midi_pitch = int(round(obs.value))
                start_time = float(obs.time)
                duration = float(obs.duration)
                
                # STANDARD_TUNING = [40, 45, 50, 55, 59, 64] (6弦→1弦, 低→高)
                # String 1 = E4 = STANDARD_TUNING[5], String 6 = E2 = STANDARD_TUNING[0]
                open_pitch = STANDARD_TUNING[6 - string_num]
                fret = midi_pitch - open_pitch
                if fret < 0 or fret > 22:
                    continue
                
                notes.append({
                    'pitch': midi_pitch,
                    'start': start_time,
                    'duration': duration,
                    'gt_string': string_num,
                    'gt_fret': fret,
                })
        
        if notes:
            notes.sort(key=lambda n: n['start'])
            file_notes[jf] = notes
            total_notes += len(notes)
    
    print(f"  Loaded {total_notes} notes from {len(file_notes)} files")
    return file_notes


def main():
    print("=" * 60)
    print("  GuitarSet Benchmark v2 (論文§8 検証)")
    print("=" * 60)
    
    print("\n[1/3] Loading GuitarSet...")
    file_notes = load_guitarset_by_file()
    
    if not file_notes:
        print("  ERROR: No files loaded.")
        return
    
    print(f"\n[2/3] Running per-file pipeline...")
    t0 = time.time()
    
    total_correct = 0
    total_within_1 = 0
    total_notes = 0
    confusion = np.zeros((NUM_STRINGS, NUM_STRINGS), dtype=int)
    per_string_correct = [0] * NUM_STRINGS
    per_string_total = [0] * NUM_STRINGS
    
    processed = 0
    for filename, notes in file_notes.items():
        # Build input notes (without GT)
        input_notes = [
            {'pitch': n['pitch'], 'start': n['start'], 'duration': n['duration']}
            for n in notes
        ]
        
        # Run full pipeline
        result = assign_strings_dp(input_notes, tuning=STANDARD_TUNING)
        
        # Compare
        for i, note in enumerate(notes):
            if i >= len(result):
                break
            gt = note['gt_string']
            pred = result[i].get('string', 1)
            
            total_notes += 1
            if gt == pred:
                total_correct += 1
            if abs(gt - pred) <= 1:
                total_within_1 += 1
            
            if 1 <= gt <= 6 and 1 <= pred <= 6:
                confusion[gt - 1][pred - 1] += 1
                per_string_total[gt - 1] += 1
                if gt == pred:
                    per_string_correct[gt - 1] += 1
        
        processed += 1
        if processed % 60 == 0:
            acc_so_far = total_correct / total_notes * 100 if total_notes > 0 else 0
            print(f"    ...{processed}/{len(file_notes)} files, accuracy={acc_so_far:.1f}%")
    
    t1 = time.time()
    
    print(f"\n[3/3] Results")
    acc = total_correct / total_notes * 100 if total_notes > 0 else 0
    within_1_pct = total_within_1 / total_notes * 100 if total_notes > 0 else 0
    far_errors = total_notes - total_within_1
    
    print(f"  Accuracy: {acc:.1f}% ({total_correct}/{total_notes})")
    print(f"  +/-1 string: {within_1_pct:.1f}% ({total_within_1}/{total_notes})")
    print(f"  >1 string errors: {far_errors} ({far_errors/total_notes*100:.2f}%)")
    print(f"  Inference time: {t1-t0:.1f}s")
    
    print(f"\n  Per-string accuracy:")
    for s in range(NUM_STRINGS):
        if per_string_total[s] > 0:
            s_acc = per_string_correct[s] / per_string_total[s] * 100
            print(f"    String {s+1}: {s_acc:.1f}% ({per_string_correct[s]}/{per_string_total[s]})")
    
    print(f"\n  Confusion matrix (rows=GT, cols=Pred):")
    print(f"      {'':>6}", end="")
    for s in range(NUM_STRINGS):
        print(f"  P{s+1:d}", end="")
    print()
    for gt_s in range(NUM_STRINGS):
        print(f"      GT{gt_s+1}: ", end="")
        for pred_s in range(NUM_STRINGS):
            print(f"{confusion[gt_s][pred_s]:5d}", end="")
        print()
    
    print(f"\n  {'Metric':<30} {'Paper':>10} {'Current':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10}")
    print(f"  {'String accuracy':<30} {'95.2%':>10} {acc:>9.1f}%")
    print(f"  {'±1 string accuracy':<30} {'99.7%':>10} {within_1_pct:>9.1f}%")
    print(f"  {'>1 string errors':<30} {'0.27%':>10} {far_errors/total_notes*100:>9.2f}%")
    
    print(f"\n{'=' * 60}")
    print(f"  BENCHMARK COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
