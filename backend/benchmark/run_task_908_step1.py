"""
backend/benchmark/run_task_908_step1.py
=======================================
TASK-908 Step 1: 大規模エルゴノミクス A/B テスト (N=1000)
"""

import os
import sys
import glob
import random
import json
import time
import io
import contextlib
from pathlib import Path
import numpy as np
import guitarpro

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from solotab_utils import STANDARD_TUNING
from string_assigner import assign_strings_dp
from refingering_engine import compute_ergonomic_cost

def process_single_gp5(file_path: str, max_notes: int = 150):
    try:
        song = guitarpro.parse(file_path)
        if not song.tracks:
            return None
            
        track = song.tracks[0]
        tuning_arr = [64, 59, 55, 50, 45, 40]
        if hasattr(track, 'strings') and track.strings:
            try:
                tuning_arr = [s.value for s in track.strings]
            except Exception:
                tuning_arr = [64, 59, 55, 50, 45, 40]
                
        raw_entries = []
        for m_idx, measure in enumerate(track.measures):
            if len(raw_entries) >= max_notes:
                break
            for v_idx, voice in enumerate(measure.voices):
                for b_idx, beat in enumerate(voice.beats):
                    for n_idx, note in enumerate(beat.notes):
                        original_string = note.string
                        original_fret = note.value
                        
                        if original_string - 1 < len(tuning_arr):
                            pitch = tuning_arr[original_string - 1] + original_fret
                        else:
                            pitch = 40 + original_fret
                            
                        start_t = float(measure.number - 1) * 3.0 + (float(beat.start) / 960.0)
                        raw_entries.append({
                            "start": start_t,
                            "end": start_t + 0.25,
                            "duration": 0.25,
                            "pitch": pitch,
                            "voice": v_idx,
                            "measure": measure.number,
                            "original_string": original_string,
                            "original_fret": original_fret,
                            "velocity": 0.8
                        })
                        if len(raw_entries) >= max_notes:
                            break
                    if len(raw_entries) >= max_notes:
                        break
                if len(raw_entries) >= max_notes:
                    break
                        
        if len(raw_entries) < 10:
            return None
            
        raw_entries.sort(key=lambda x: (x["start"], x["voice"]))
        
        # 運指最適化 (Refingering)
        with contextlib.redirect_stdout(io.StringIO()):
            assigned_notes = assign_strings_dp(
                raw_entries,
                tuning=STANDARD_TUNING,
                audio_path=None
            )
        
        orig_notes_meta = [{"string": n["original_string"], "fret": n["original_fret"]} for n in raw_entries]
        orig_cost = compute_ergonomic_cost(orig_notes_meta)
        opt_cost = compute_ergonomic_cost(assigned_notes)
        
        diff_movement = opt_cost["total_movement_frets"] - orig_cost["total_movement_frets"]
        diff_jumps = opt_cost["excessive_jumps_gt_4f"] - orig_cost["excessive_jumps_gt_4f"]
        
        exact_matches = sum(1 for o, a in zip(orig_notes_meta, assigned_notes) if o["string"] == a["string"] and o["fret"] == a["fret"])
        match_rate = exact_matches / len(raw_entries)
        
        return {
            "file": os.path.basename(file_path),
            "note_count": len(raw_entries),
            "orig_movement": orig_cost["total_movement_frets"],
            "opt_movement": opt_cost["total_movement_frets"],
            "diff_movement": diff_movement,
            "orig_jumps": orig_cost["excessive_jumps_gt_4f"],
            "opt_jumps": opt_cost["excessive_jumps_gt_4f"],
            "diff_jumps": diff_jumps,
            "match_rate": match_rate
        }
    except Exception as e:
        return None


def run_step1_evaluation(sample_size=1000, seed=42):
    print(f"=== TASK-908 Step 1: 大規模エルゴノミクス A/B テスト (N={sample_size}) ===", flush=True)
    random.seed(seed)
    
    dataset_dir = "datasets/non_standard"
    all_files = glob.glob(os.path.join(dataset_dir, "*.gp5"))
    print(f"Total available files in {dataset_dir}: {len(all_files)}", flush=True)
    
    if len(all_files) < sample_size:
        raise ValueError(f"Not enough files: requested {sample_size}, found {len(all_files)}")
        
    sampled_files = random.sample(all_files, sample_size * 2)
    print(f"Candidate files prepared. Starting evaluation...", flush=True)
    
    t0 = time.time()
    results = []
    
    for idx, f in enumerate(sampled_files):
        if len(results) >= sample_size:
            break
            
        res = process_single_gp5(f, max_notes=150)
        if res is not None:
            results.append(res)
            
        if len(results) % 100 == 0 and len(results) > 0 and (len(results) != getattr(run_step1_evaluation, '_last_log', 0)):
            run_step1_evaluation._last_log = len(results)
            print(f"Step 1 Progress: {len(results)}/{sample_size} songs evaluated (Elapsed: {time.time()-t0:.1f}s)", flush=True)
            
    elapsed = time.time() - t0
    print(f"Step 1 Completed in {elapsed:.2f}s. Valid evaluated songs: {len(results)}", flush=True)
    
    diff_movements = [r["diff_movement"] for r in results]
    diff_jumps = [r["diff_jumps"] for r in results]
    orig_movements = [r["orig_movement"] for r in results]
    opt_movements = [r["opt_movement"] for r in results]
    match_rates = [r["match_rate"] for r in results]
    
    improved = sum(1 for d in diff_movements if d < 0)
    worsened = sum(1 for d in diff_movements if d > 0)
    unchanged = sum(1 for d in diff_movements if d == 0)
    
    total_valid = len(results)
    improved_pct = (improved / total_valid) * 100
    worsened_pct = (worsened / total_valid) * 100
    unchanged_pct = (unchanged / total_valid) * 100
    
    jumps_decreased = sum(1 for j in diff_jumps if j < 0)
    jumps_increased = sum(1 for j in diff_jumps if j > 0)
    jumps_unchanged = sum(1 for j in diff_jumps if j == 0)
    
    p10_mov = np.percentile(diff_movements, 10)
    p50_mov = np.percentile(diff_movements, 50)
    p90_mov = np.percentile(diff_movements, 90)
    p95_mov = np.percentile(diff_movements, 95)
    p99_mov = np.percentile(diff_movements, 99)
    
    p10_jmp = np.percentile(diff_jumps, 10)
    p50_jmp = np.percentile(diff_jumps, 50)
    p90_jmp = np.percentile(diff_jumps, 90)
    p95_jmp = np.percentile(diff_jumps, 95)
    p99_jmp = np.percentile(diff_jumps, 99)
    
    summary = {
        "task": "TASK-908 Step 1: 大規模エルゴノミクス A/B テスト (N=1000)",
        "sample_size": sample_size,
        "valid_songs_evaluated": total_valid,
        "elapsed_seconds": round(elapsed, 2),
        "movement_difference_optimized_minus_original": {
            "mean": round(float(np.mean(diff_movements)), 2),
            "std": round(float(np.std(diff_movements)), 2),
            "median": round(float(np.median(diff_movements)), 2),
            "percentiles": {
                "P10": round(float(p10_mov), 2),
                "P50": round(float(p50_mov), 2),
                "P90": round(float(p90_mov), 2),
                "P95_worst_case": round(float(p95_mov), 2),
                "P99": round(float(p99_mov), 2)
            },
            "classification": {
                "improved_ratio_pct": round(improved_pct, 2),
                "improved_count": improved,
                "worsened_ratio_pct": round(worsened_pct, 2),
                "worsened_count": worsened,
                "unchanged_ratio_pct": round(unchanged_pct, 2),
                "unchanged_count": unchanged
            }
        },
        "excessive_jumps_difference_optimized_minus_original": {
            "mean": round(float(np.mean(diff_jumps)), 2),
            "std": round(float(np.std(diff_jumps)), 2),
            "median": round(float(np.median(diff_jumps)), 2),
            "percentiles": {
                "P10": round(float(p10_jmp), 2),
                "P50": round(float(p50_jmp), 2),
                "P90": round(float(p90_jmp), 2),
                "P95_worst_case": round(float(p95_jmp), 2),
                "P99": round(float(p99_jmp), 2)
            },
            "classification": {
                "decreased_jumps_pct": round((jumps_decreased / total_valid) * 100, 2),
                "increased_jumps_pct": round((jumps_increased / total_valid) * 100, 2),
                "unchanged_jumps_pct": round((jumps_unchanged / total_valid) * 100, 2)
            }
        },
        "overall_string_fret_match_rate_with_original": {
            "mean": round(float(np.mean(match_rates)), 4),
            "median": round(float(np.median(match_rates)), 4),
            "P10": round(float(np.percentile(match_rates, 10)), 4),
            "P90": round(float(np.percentile(match_rates, 90)), 4)
        }
    }
    
    out_path = "backend/benchmark/task_908_step1_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        
    print("\n--- Summary Results ---", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary

if __name__ == "__main__":
    run_step1_evaluation()
