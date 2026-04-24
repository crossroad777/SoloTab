import json
from collections import Counter
from typing import List, Dict
import sys
sys.path.insert(0, r'd:\Music\nextchord-solotab\backend')
from ensemble_transcriber import transcribe_ensemble
from note_filter import apply_all_filters
from string_assigner import assign_strings_dp

def get_gt(gt_path):
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt = json.load(f)
    bpm = gt['metadata']['bpm']
    max_beat = 6.0 * (bpm / 60.0)
    gt_notes = []
    
    for measure in gt['measures_detailed']:
        measure_base_beat = (measure['measure'] - 1) * 3
        for note in measure['notes']:
            note_abs_beat = measure_base_beat + note['beat']
            if note_abs_beat <= max_beat:
                gt_notes.append({
                    "pitch": note.get('pitch', 0),
                    "beat": round(note_abs_beat, 2)
                })
    return gt_notes

def main():
    wav_path = r'd:\Music\nextchord-solotab\uploads\20260402-222304-yt-229aa6\converted.wav'
    gt_path = r'd:\Music\nextchord-solotab\backend\ground_truth\romance_forbidden_games.json'
    
    res = transcribe_ensemble(wav_path, tuning_pitches=[40, 45, 50, 55, 59, 64])
    f_res = apply_all_filters(res['notes'], velocity_threshold=0.55, min_duration_sec=0.05, max_notes_per_beat=6)
    pred_notes = [n for n in f_res['notes'] if n.get('start', 0) < 6.0]
    
    gt_notes = get_gt(gt_path)
    
    print("--- Ground Truth Notes (Pitch & Beat) ---")
    for n in sorted(gt_notes, key=lambda x: x['beat']):
        print(f"Beat: {n['beat']:.2f} | Pitch: {n['pitch']}")
        
    print("\n--- Predicted Notes (Pitch & Start) ---")
    for n in sorted(pred_notes, key=lambda x: x['start']):
        print(f"Start: {n['start']:.2f}s | Pitch: {n['pitch']} | Vel: {n.get('velocity', 0):.2f}")

if __name__ == '__main__':
    main()
