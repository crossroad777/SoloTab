import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
import json
import tempfile
import shutil
from pathlib import Path
import numpy as np

# backend ディレクトリへのパスを追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from e2e_pipeline_benchmark import TARGET_TRACKS, ANNOTATIONS_DIR, AUDIO_DIR, load_jams_notes_with_string, to_mireval
from pipeline import run_pipeline
import mir_eval

def diagnose_mismatch():
    print("="*60)
    print(" DIAGNOSING VITERBI DP OVERWRITE BEHAVIOR")
    print("="*60)
    
    total_raw_matches = 0
    total_viterbi_matches = 0
    total_gt_notes = 0
    
    viterbi_overwrites_good_to_bad = 0
    viterbi_fixes_bad_to_good = 0
    viterbi_kept_good = 0
    viterbi_kept_bad = 0
    
    mismatch_details = []
    
    for i, track in enumerate(TARGET_TRACKS):
        jams_path = os.path.join(ANNOTATIONS_DIR, f"{track}.jams")
        wav_path = os.path.join(AUDIO_DIR, f"{track}_mic.wav")
        if not os.path.exists(jams_path) or not os.path.exists(wav_path):
            continue
            
        gt_notes = load_jams_notes_with_string(jams_path)
        if not gt_notes:
            continue
            
        temp_dir = tempfile.mkdtemp(prefix="solotab_diag_")
        session_id = f"diag_{track}"
        
        try:
            run_pipeline(
                session_id=session_id,
                session_dir=Path(temp_dir),
                wav_path=Path(wav_path),
                tuning_name="standard",
                skip_demucs=True,
                fast_moe=True,
                moe_vote_threshold=4
            )
            
            # 1. Viterbi 適用前の生ノート (notes.json)
            raw_notes = []
            raw_json_path = os.path.join(temp_dir, "notes.json")
            if os.path.exists(raw_json_path):
                with open(raw_json_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                    if isinstance(raw_data, dict) and "notes" in raw_data:
                        raw_notes = raw_data["notes"]
                    elif isinstance(raw_data, list):
                        raw_notes = raw_data
                        
            # 2. Viterbi 適用後のノート (notes_assigned_original.json)
            assigned_json_path = os.path.join(temp_dir, "notes_assigned_original.json")
            if not os.path.exists(assigned_json_path):
                assigned_json_path = os.path.join(temp_dir, "notes_assigned.json")
                
            if not os.path.exists(assigned_json_path):
                print(f"[{track}] Assigned notes not found.")
                continue
                
            with open(assigned_json_path, 'r', encoding='utf-8') as f:
                assigned_notes = json.load(f)
                    
            ref_intervals, ref_pitches = to_mireval(gt_notes)
            est_intervals, est_pitches = to_mireval(assigned_notes)
            
            matching = mir_eval.transcription.match_notes(
                ref_intervals, ref_pitches, est_intervals, est_pitches,
                onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None
            )
            
            track_g2b = 0
            track_b2g = 0
            track_good = 0
            
            for ref_idx, est_idx in matching:
                gt_n = gt_notes[ref_idx]
                ass_n = assigned_notes[est_idx]
                
                gt_s = gt_n.get('string')
                ass_s = ass_n.get('string')
                raw_s = ass_n.get('_raw_string', raw_notes[est_idx].get('string') if est_idx < len(raw_notes) else None)
                
                raw_correct = (raw_s == gt_s) if raw_s is not None else None
                ass_correct = (ass_s == gt_s)
                
                if ass_correct:
                    track_good += 1
                    
                if raw_correct is not None:
                    if raw_correct and not ass_correct:
                        track_g2b += 1
                        viterbi_overwrites_good_to_bad += 1
                        mismatch_details.append({
                            "track": track,
                            "pitch": gt_n.get('pitch'),
                            "gt_string": gt_s,
                            "raw_string": raw_s,
                            "assigned_string": ass_s,
                            "fret": ass_n.get('fret'),
                            "time": gt_n.get('start')
                        })
                    elif not raw_correct and ass_correct:
                        track_b2g += 1
                        viterbi_fixes_bad_to_good += 1
                    elif raw_correct and ass_correct:
                        viterbi_kept_good += 1
                    else:
                        viterbi_kept_bad += 1
                        
            print(f"[{i+1}/{len(TARGET_TRACKS)}] {track:25s} | Correct: {track_good}/{len(matching)} | Good->Bad (Degraded): {track_g2b} | Bad->Good (Fixed): {track_b2g}")
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    print("\n" + "="*60)
    print(" SUMMARY OF VITERBI IMPACT")
    print("="*60)
    print(f"Total Notes Evaluated (Matched): {viterbi_kept_good + viterbi_kept_bad + viterbi_overwrites_good_to_bad + viterbi_fixes_bad_to_good}")
    print(f"  - Kept Good (Both Correct)     : {viterbi_kept_good}")
    print(f"  - Viterbi Fixed (Bad -> Good)  : {viterbi_fixes_bad_to_good} (+)")
    print(f"  - Viterbi Ruined (Good -> Bad) : {viterbi_overwrites_good_to_bad} (-)")
    print(f"  - Net Viterbi Contribution     : {viterbi_fixes_bad_to_good - viterbi_overwrites_good_to_bad}")
    
    # Save details
    with open("viterbi_mismatch_details.json", "w", encoding="utf-8") as f:
        json.dump(mismatch_details, f, indent=2)
    print(f"Detailed ruined notes saved to viterbi_mismatch_details.json")

if __name__ == "__main__":
    diagnose_mismatch()
