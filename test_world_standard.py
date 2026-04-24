import os
import json
import sys
import time
from typing import List, Dict
from collections import Counter

sys.path.insert(0, r'd:\Music\nextchord-solotab\backend')
try:
    from ensemble_transcriber import transcribe_ensemble
    from note_filter import apply_all_filters
    from string_assigner import assign_strings_dp
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def evaluate_metrics(predicted_notes: List[Dict], gt_path: str):
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt = json.load(f)

    bpm = gt['metadata'].get('bpm', 88)
    max_beat = 72.0 # Evaluate exactly 24 measures (24 * 3 = 72 beats)

    gt_pitches = Counter()
    gt_total = 0
    for measure in gt['measures_detailed']:
        measure_base_beat = (measure['measure'] - 1) * (gt['metadata'].get('time_signature_num', 3))
        for note in measure['notes']:
            note_abs_beat = measure_base_beat + note['beat']
            if note_abs_beat <= max_beat:
                gt_pitches[note.get('pitch', 0)] += 1
                gt_total += 1

    # Extract predicted notes within 50.8s (covering exactly the 72 beats)
    pred_notes = [n for n in predicted_notes if n.get('start', 0) < 50.8]
    pred_pitches = Counter(n.get('pitch', 0) for n in pred_notes)
    
    all_p = set(gt_pitches.keys()).union(pred_pitches.keys())
    
    correct = sum(min(gt_pitches.get(p, 0), pred_pitches.get(p, 0)) for p in all_p)
    tot_pred = len(pred_notes)

    p = correct / tot_pred if tot_pred > 0 else 0.0
    r = correct / gt_total if gt_total > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    return f1, p, r, tot_pred, gt_total

def search_best_params(raw_notes, tuning, gt_path):
    print("\n--- Starting Parameter Grid Search ---")
    best_f1 = 0.0
    best_params = {}
    best_metrics = {}

    vel_thresholds = [0.55, 0.60, 0.63, 0.65, 0.68, 0.72]
    min_durations = [0.03, 0.05, 0.08, 0.10]
    
    for vt in vel_thresholds:
        for md in min_durations:
            import copy
            notes_copy = copy.deepcopy(raw_notes)
            filtered = apply_all_filters(
                notes_copy,
                velocity_threshold=vt,
                min_duration_sec=md,
                max_notes_per_beat=6
            )
            final_notes = filtered['notes']
            
            # Simple string assigner overhead
            final_notes = assign_strings_dp(final_notes, tuning=tuning)
            
            f1, p, r, tot, gt = evaluate_metrics(final_notes, gt_path)
            
            # Only print promising results
            if f1 > 0.65:
                # print(f"VT={vt:.2f}, MD={md:.2f} => F1: {f1:.3f} (P: {p:.3f}, R: {r:.3f})")
                pass
                
            if f1 > best_f1:
                best_f1 = f1
                best_params = {'velocity_threshold': vt, 'min_duration_sec': md}
                best_metrics = {'f1': f1, 'p': p, 'r': r, 'tot': tot, 'gt': gt}
    
    print("\n=== BEST CONFIGURATION FOUND ===")
    print(f"velocity_threshold: {best_params['velocity_threshold']}")
    print(f"min_duration_sec:   {best_params['min_duration_sec']}")
    print(f"Val F1: {best_metrics['f1']:.3f} (World Target: >= 0.75)")
    print(f"P:      {best_metrics['p']:.3f}")
    print(f"R:      {best_metrics['r']:.3f}")
    print(f"Predicted Notes: {best_metrics['tot']} (GT: {best_metrics['gt']})")

def main():
    wav_path = r'd:\Music\nextchord-solotab\uploads\20260402-222304-yt-229aa6\converted.wav'
    gt_path = r'd:\Music\nextchord-solotab\backend\ground_truth\romance_forbidden_games.json'
    
    if not os.path.exists(wav_path):
        print(f"File not found: {wav_path}")
        return
        
    tuning = [40, 45, 50, 55, 59, 64]
    
    print("Running Ensemble Transcription...")
    t0 = time.time()
    try:
        res = transcribe_ensemble(wav_path, tuning_pitches=tuning)
        notes = res['notes']
        print(f"Raw ensemble notes: {len(notes)}")
        
        search_best_params(notes, tuning, gt_path)
        
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
