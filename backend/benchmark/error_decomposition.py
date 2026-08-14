import os
import json
import glob
import sys
import copy
from collections import defaultdict
import tempfile
import shutil
from pathlib import Path
import time
import numpy as np

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    import mir_eval
except ImportError:
    print("mir_eval required: pip install mir_eval")
    sys.exit(1)

from e2e_pipeline_benchmark import TARGET_TRACKS, ANNOTATIONS_DIR, load_jams_notes_with_string, to_mireval
try:
    from pipeline import run_pipeline
except ImportError as e:
    print(f"Error importing pipeline: {e}")
    sys.exit(1)

# TUNING for standard EADGBE
TUNING_MIDI = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}

def main():
    print("==================================================")
    print(" ERROR DECOMPOSITION (GuitarSet Mini)")
    print("==================================================")
    
    total_gt = 0
    total_est = 0
    
    cat_A_fn = 0
    cat_A_fp = 0
    cat_A_pitch_mismatch = 0
    
    cat_B1 = 0
    cat_B2 = 0
    cat_B3 = 0
    
    cat_D = 0
    
    # Distributions
    pitch_bands = {'Low (E2-E3)': 0, 'Mid (E3-B3)': 0, 'High (C4+)': 0}
    gt_string_errors = {i: 0 for i in range(1, 7)}
    confusion = {i: {j: 0 for j in range(1, 7)} for i in range(1, 7)}
    
    def get_pitch_band(pitch):
        if pitch < 52: # E2 is 40, E3 is 52.
            return 'Low (E2-E3)'
        elif pitch < 60: # C4 is 60.
            return 'Mid (E3-B3)'
        else:
            return 'High (C4+)'

    start_time = time.time()
    for i, track in enumerate(TARGET_TRACKS):
        print(f"\n[{i+1}/{len(TARGET_TRACKS)}] Evaluating Track: {track}")
        
        temp_dir = tempfile.mkdtemp(prefix="solotab_e2e_")
        session_id = f"benchmark_{track}"
        wav_path = os.path.join(ANNOTATIONS_DIR.replace("annotation", "audio_mono-mic"), f"{track}_mic.wav")
        jams_path = os.path.join(ANNOTATIONS_DIR, f"{track}.jams")
        
        try:
            gt_notes = load_jams_notes_with_string(jams_path)
            
            run_pipeline(
                session_id=session_id,
                session_dir=Path(temp_dir),
                wav_path=Path(wav_path),
                tuning_name="standard",
                skip_demucs=True,
                fast_moe=True
            )
            
            json_path = os.path.join(temp_dir, "notes_assigned_original.json")
            if not os.path.exists(json_path):
                json_path = os.path.join(temp_dir, "notes_assigned.json")
                
            with open(json_path, 'r', encoding='utf-8') as f:
                est_notes = json.load(f)
                
            total_gt += len(gt_notes)
            total_est += len(est_notes)
            
            ref_intervals, ref_pitches = to_mireval(gt_notes)
            est_intervals, est_pitches = to_mireval(est_notes)
            
            # Match notes (50ms onset, 50 cents pitch)
            matched_pairs = mir_eval.transcription.match_notes(
                ref_intervals, ref_pitches, est_intervals, est_pitches,
                onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None
            )
            
            matched_ref_indices = set([p[0] for p in matched_pairs])
            matched_est_indices = set([p[1] for p in matched_pairs])
            
            # Match strictly by onset to find Pitch Mismatches
            matched_pairs_onset = mir_eval.transcription.match_notes(
                ref_intervals, ref_pitches, est_intervals, est_pitches,
                onset_tolerance=0.05, pitch_tolerance=10000.0, offset_ratio=None 
            )
            
            onset_matched_ref = set([p[0] for p in matched_pairs_onset])
            onset_matched_est = set([p[1] for p in matched_pairs_onset])
            
            # Category A: False Negatives
            for ref_idx in range(len(gt_notes)):
                if ref_idx not in matched_ref_indices:
                    # Either completely unmatched or matched by onset but pitch was wrong
                    if ref_idx in onset_matched_ref:
                        cat_A_pitch_mismatch += 1
                    else:
                        cat_A_fn += 1
                        
            # Category A: False Positives
            for est_idx in range(len(est_notes)):
                if est_idx not in matched_est_indices:
                    if est_idx not in onset_matched_est:
                        cat_A_fp += 1
                        
            # For correctly matched pitches, check string
            for ref_idx, est_idx in matched_pairs:
                ref = gt_notes[ref_idx]
                est = est_notes[est_idx]
                
                ref_string = ref['string']
                est_string = est.get('string', 0)
                
                if ref_string == est_string:
                    cat_D += 1
                else:
                    # Category B
                    cnn_probs = est.get('cnn_string_probs', {})
                    
                    if not cnn_probs:
                        # Fallback if no CNN prob found
                        cat_B3 += 1
                    else:
                        # Find max prob
                        max_s = None
                        max_p = -1
                        for s, p in cnn_probs.items():
                            s_int = int(s)
                            if float(p) > max_p:
                                max_p = float(p)
                                max_s = s_int
                        
                        if max_p < 0.5:
                            cat_B3 += 1
                        else:
                            if max_s == ref_string:
                                cat_B1 += 1
                            else:
                                cat_B2 += 1
                                
                    # Record distribution
                    band = get_pitch_band(ref['pitch'])
                    pitch_bands[band] += 1
                    if ref_string in gt_string_errors:
                        gt_string_errors[ref_string] += 1
                    
                    if ref_string in confusion and est_string in confusion[ref_string]:
                        confusion[ref_string][est_string] += 1
                        
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  Error running E2E pipeline for {track}: {e}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    print("\n=== Error Decomposition Summary ===")
    print(f"Total GT Notes: {total_gt}")
    print(f"Total Detected Notes: {total_est}")
    
    total_A = cat_A_fn + cat_A_fp + cat_A_pitch_mismatch
    total_B = cat_B1 + cat_B2 + cat_B3
    
    def pct(num):
        return f"{num / total_gt * 100:.1f}%" if total_gt > 0 else "0.0%"
        
    print(f"\nCategory A (Pitch Detection Error): {total_A} notes ({pct(total_A)})")
    print(f"  - False Negative (検出漏れ): {cat_A_fn}")
    print(f"  - False Positive (誤検出): {cat_A_fp}")
    print(f"  - Pitch Mismatch (ピッチずれ): {cat_A_pitch_mismatch}")
    
    print(f"\nCategory B (String Assignment Error): {total_B} notes ({pct(total_B)})")
    print(f"  - B1 (Viterbi override): {cat_B1}")
    print(f"  - B2 (CNN misclassification): {cat_B2}")
    print(f"  - B3 (Low confidence fallback/No CNN): {cat_B3}")
    
    print(f"\nCategory D (Correct): {cat_D} notes ({pct(cat_D)})")
    
    print("\n--- Pitch Band Distribution (Category B) ---")
    for b, c in pitch_bands.items():
        print(f"  {b}: {c}")
        
    print("\n--- GT String Error Distribution (Category B) ---")
    for s in range(1, 7):
        print(f"  String {s}: {gt_string_errors[s]}")
        
    print("\n--- Confusion Matrix (GT vs Est, Category B) ---")
    print("   " + "".join([f" E{i} " for i in range(1, 7)]))
    for gt_s in range(1, 7):
        row = f"G{gt_s} "
        for est_s in range(1, 7):
            row += f"{confusion[gt_s][est_s]:3d} "
        print(row)

if __name__ == "__main__":
    main()
