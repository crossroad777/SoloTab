import os
import json
import sys
import tempfile
import time
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    import mir_eval
except ImportError:
    print("mir_eval required: pip install mir_eval")
    sys.exit(1)

from benchmark.e2e_pipeline_benchmark import TARGET_TRACKS, ANNOTATIONS_DIR, load_jams_notes_with_string, to_mireval
from pipeline import run_pipeline

def evaluate_pipeline(params):
    stats = {"A1": 0, "A2": 0, "A3": 0}
    all_ref_intervals = []
    all_ref_pitches = []
    all_est_intervals = []
    all_est_pitches = []
    
    ss_a2 = 0
    
    temp_dir = tempfile.mkdtemp(prefix="solotab_sweep_")
    
    for track in TARGET_TRACKS:
        session_id = f"sweep_{track}"
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
                fast_moe=True,
                **params
            )

            json_path = os.path.join(temp_dir, "notes_assigned_original.json")
            if not os.path.exists(json_path):
                json_path = os.path.join(temp_dir, "notes_assigned.json")

            with open(json_path, 'r', encoding='utf-8') as f:
                est_notes = json.load(f)

            gt_intervals, gt_pitches = to_mireval(gt_notes)
            est_intervals, est_pitches = to_mireval(est_notes)
            
            all_ref_intervals.append(gt_intervals)
            all_ref_pitches.append(gt_pitches)
            all_est_intervals.append(est_intervals)
            all_est_pitches.append(est_pitches)
            
            matched_tp = mir_eval.transcription.match_notes(
                gt_intervals, gt_pitches, est_intervals, est_pitches,
                onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None
            )
            
            matched_time = mir_eval.transcription.match_notes(
                gt_intervals, np.ones_like(gt_pitches) * 440.0, est_intervals, np.ones_like(est_pitches) * 440.0,
                onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None
            )
            
            matched_gt_idx_tp = set([m[0] for m in matched_tp])
            matched_est_idx_tp = set([m[1] for m in matched_tp])
            matched_gt_idx_time = set([m[0] for m in matched_time])
            matched_est_idx_time = set([m[1] for m in matched_time])
            
            # A1
            a1 = len(set(range(len(gt_notes))) - matched_gt_idx_time)
            stats["A1"] += a1
            
            # A2
            a2 = len(set(range(len(est_notes))) - matched_est_idx_time)
            stats["A2"] += a2
            
            if track == "00_SS1-68-E_comp":
                ss_a2 = a2
                
            # A3
            a3 = 0
            for gt_idx, est_idx in matched_time:
                if gt_idx not in matched_gt_idx_tp or est_idx not in matched_est_idx_tp:
                    a3 += 1
            stats["A3"] += a3

        except Exception as e:
            print(f"Error on {track}: {e}")
            
    # Calculate F1

    # Actually mir_eval.multipitch.evaluate expects single tracks, so we have to manually compute global metrics
    # Or we can just calculate standard F1 for exact pitch matches.
    
    # We'll calculate Global Precision, Recall, F1 for Pitch
    total_tp = 0
    total_est_all = sum(len(est) for est in all_est_pitches)
    total_ref_all = sum(len(ref) for ref in all_ref_pitches)
    
    for ref_int, ref_p, est_int, est_p in zip(all_ref_intervals, all_ref_pitches, all_est_intervals, all_est_pitches):
        m = mir_eval.transcription.match_notes(
            ref_int, ref_p, est_int, est_p,
            onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None
        )
        total_tp += len(m)
        
    p = total_tp / total_est_all if total_est_all > 0 else 0
    r = total_tp / total_ref_all if total_ref_all > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

    return {
        "F1": f1, "P": p, "R": r,
        "A1": stats["A1"], "A2": stats["A2"], "A3": stats["A3"],
        "SS_A2": ss_a2
    }

def main():
    print("=== Step 1: Baseline ===")
    baseline_params = {
        "bp_onset_threshold": 0.5,
        "bp_minimum_note_length": 58.0,
        "moe_vote_threshold": -1, # default
        "moe_vote_prob_threshold": 0.5,
        "bp_only_threshold": 0.05
    }
    baseline_results = evaluate_pipeline(baseline_params)
    print(f"Baseline: F1={baseline_results['F1']:.4f} P={baseline_results['P']:.4f} R={baseline_results['R']:.4f} | A1={baseline_results['A1']} A2={baseline_results['A2']} (SS_A2={baseline_results['SS_A2']})")
    
    # Step 2: onset_threshold
    print("\n=== Step 2.1: Sweep onset_threshold ===")
    for v in [0.5, 0.6, 0.7, 0.8, 0.9]:
        params = baseline_params.copy()
        params["bp_onset_threshold"] = v
        res = evaluate_pipeline(params)
        print(f"onset_threshold={v}: F1={res['F1']:.4f} P={res['P']:.4f} R={res['R']:.4f} | A1={res['A1']} A2={res['A2']} (SS_A2={res['SS_A2']})")
        
    # Step 2: vote_threshold
    print("\n=== Step 2.2: Sweep vote_threshold ===")
    for v in [19, 20, 21, 22, 23, 24]:
        params = baseline_params.copy()
        params["moe_vote_threshold"] = v
        res = evaluate_pipeline(params)
        print(f"vote_threshold={v}: F1={res['F1']:.4f} P={res['P']:.4f} R={res['R']:.4f} | A1={res['A1']} A2={res['A2']} (SS_A2={res['SS_A2']})")

    # Step 2: minimum_note_length
    print("\n=== Step 2.3: Sweep minimum_note_length ===")
    for v in [58.0, 80.0, 100.0, 120.0]:
        params = baseline_params.copy()
        params["bp_minimum_note_length"] = v
        res = evaluate_pipeline(params)
        print(f"min_note_length={v}: F1={res['F1']:.4f} P={res['P']:.4f} R={res['R']:.4f} | A1={res['A1']} A2={res['A2']} (SS_A2={res['SS_A2']})")

if __name__ == "__main__":
    main()
