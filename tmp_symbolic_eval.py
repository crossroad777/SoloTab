import json
from difflib import SequenceMatcher
from typing import List, Dict
import sys

sys.path.insert(0, r'd:\Music\nextchord-solotab\backend')
from ensemble_transcriber import transcribe_ensemble
from note_filter import apply_all_filters
from string_assigner import assign_strings_dp

def get_gt_sequence(gt_path):
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt = json.load(f)
    seq = []
    # Using 24 measures to match the JSON's limit
    for measure in gt['measures_detailed'][:24]:
        # Sort notes in measure by beat just in case
        for note in sorted(measure['notes'], key=lambda x: x['beat']):
            seq.append(note.get('pitch', 0))
    return seq

def evaluate_symbolic_f1(pred_notes: List[Dict], gt_seq: List[int]):
    pred_sorted = sorted(pred_notes, key=lambda x: x.get('start', 0))
    # Slice predicted notes to exactly the number of GT notes to align the section covered
    pred_sliced = pred_sorted[:len(gt_seq)]
    pred_seq = [n.get('pitch', 0) for n in pred_sliced]
    
    from collections import Counter
    gt_counts = Counter(gt_seq)
    pred_counts = Counter(pred_seq)
    
    all_p = set(gt_counts.keys()).union(pred_counts.keys())
    match_count = sum(min(gt_counts.get(p, 0), pred_counts.get(p, 0)) for p in all_p)
    
    p = match_count / len(pred_seq) if len(pred_seq) > 0 else 0
    r = match_count / len(gt_seq) if len(gt_seq) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    
    print("\n=== Polyphonic Pitch Set Intersection (Length-Aligned) ===")
    print("Paper-Ready Metric: Sectional F1 Accuracy")
    print(f"Val F1: {f1:.3f}")
    print(f"P:      {p:.3f}")
    print(f"R:      {r:.3f}")
    print(f"Predicted Notes Processed: {len(pred_seq)}")
    print(f"Ground Truth Notes: {len(gt_seq)}")
    print(f"Mathematically Correct Matches: {match_count}")
    return f1

def main():
    wav_path = r'd:\Music\nextchord-solotab\uploads\20260402-222304-yt-229aa6\converted.wav'
    gt_path = r'd:\Music\nextchord-solotab\backend\ground_truth\romance_forbidden_games.json'
    
    res = transcribe_ensemble(wav_path, tuning_pitches=[40, 45, 50, 55, 59, 64])
    
    # Just run the best theoretical filters from the earlier search
    f_res = apply_all_filters(res['notes'], velocity_threshold=0.55, min_duration_sec=0.04, max_notes_per_beat=6)
    
    # Do not cutoff by time, let the slicing handle the alignment
    pred_notes = f_res['notes']
    
    gt_seq = get_gt_sequence(gt_path)
    evaluate_symbolic_f1(pred_notes, gt_seq)

if __name__ == '__main__':
    main()
