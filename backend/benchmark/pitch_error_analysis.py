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

from e2e_pipeline_benchmark import TARGET_TRACKS, ANNOTATIONS_DIR, load_jams_notes_with_string, to_mireval
try:
    from pipeline import run_pipeline
except ImportError as e:
    print(f"Error importing pipeline: {e}")
    sys.exit(1)

def main():
    print("==================================================")
    print(" PITCH ERROR ANALYSIS (Category A)")
    print("==================================================")

    stats = {
        "A1": 0, "A2": 0, "A3": 0,
        "A3_octave": 0, "A3_semitone": 0,
        "A1_open_string": 0,
        "A1_dense": 0,
        "A2_octave_overtone": 0,
        "track": defaultdict(lambda: {"A1":0, "A2":0, "A3":0}),
        "band": {"Low (E2-E3)": {"A1":0, "A2":0, "A3":0}, "Mid (E3-B3)": {"A1":0, "A2":0, "A3":0}, "High (C4+)": {"A1":0, "A2":0, "A3":0}},
        "genre": defaultdict(lambda: {"A1":0, "A2":0, "A3":0})
    }
    
    def get_pitch_band(pitch):
        if pitch < 52:
            return 'Low (E2-E3)'
        elif pitch < 72:
            return 'Mid (E3-B3)'
        else:
            return 'High (C4+)'

    start_time = time.time()
    for i, track in enumerate(TARGET_TRACKS):
        print(f"\n[{i+1}/{len(TARGET_TRACKS)}] Evaluating Track: {track}")
        
        genre = track.split('_')[1].split('-')[0]
        if genre.startswith("SS"): genre = "SS"
        if genre.startswith("Funk"): genre = "Funk"
        if genre.startswith("Jazz"): genre = "Jazz"
        if genre.startswith("BN"): genre = "BN"
        if genre.startswith("Rock"): genre = "Rock"

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
                fast_moe=True,
                moe_vote_threshold=4
            )

            json_path = os.path.join(temp_dir, "notes_assigned_original.json")
            if not os.path.exists(json_path):
                json_path = os.path.join(temp_dir, "notes_assigned.json")

            with open(json_path, 'r', encoding='utf-8') as f:
                est_notes = json.load(f)

            gt_intervals, gt_pitches = to_mireval(gt_notes)
            est_intervals, est_pitches = to_mireval(est_notes)

            # 完全マッチ (Time + Pitch)
            matched_tp = mir_eval.transcription.match_notes(
                gt_intervals, gt_pitches, est_intervals, est_pitches,
                onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None
            )
            
            matched_gt_idx_tp = set([m[0] for m in matched_tp])
            matched_est_idx_tp = set([m[1] for m in matched_tp])
            
            # 時間マッチのみ
            matched_time = mir_eval.transcription.match_notes(
                gt_intervals, np.ones_like(gt_pitches) * 440.0, est_intervals, np.ones_like(est_pitches) * 440.0,
                onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None
            )
            
            matched_gt_idx_time = set([m[0] for m in matched_time])
            matched_est_idx_time = set([m[1] for m in matched_time])
            
            # A1: False Negative
            a1_gt_indices = set(range(len(gt_notes))) - matched_gt_idx_time
            for idx in a1_gt_indices:
                stats["A1"] += 1
                pitch = gt_notes[idx]["pitch"]
                band = get_pitch_band(pitch)
                stats["band"][band]["A1"] += 1
                stats["track"][track]["A1"] += 1
                stats["genre"][genre]["A1"] += 1
                
                if pitch in [40, 45, 50, 55, 59, 64]:
                    stats["A1_open_string"] += 1
                    
                onset = gt_notes[idx]["start"]
                is_dense = False
                for j in range(len(gt_notes)):
                    if j != idx and abs(gt_notes[j]["start"] - onset) < 0.1:
                        is_dense = True
                        break
                if is_dense:
                    stats["A1_dense"] += 1

            # A2: False Positive
            a2_est_indices = set(range(len(est_notes))) - matched_est_idx_time
            for idx in a2_est_indices:
                stats["A2"] += 1
                pitch = est_notes[idx]["pitch"]
                band = get_pitch_band(pitch)
                stats["band"][band]["A2"] += 1
                stats["track"][track]["A2"] += 1
                stats["genre"][genre]["A2"] += 1
                
                onset = est_notes[idx]["start"]
                is_overtone = False
                for gt in gt_notes:
                    if abs(gt["start"] - onset) < 0.2:
                        if abs(gt["pitch"] - (pitch - 12)) < 2 or abs(gt["pitch"] - (pitch - 24)) < 2:
                            is_overtone = True
                            break
                if is_overtone:
                    stats["A2_octave_overtone"] += 1
                    
            # A3: Pitch Mismatch
            for gt_idx, est_idx in matched_time:
                if gt_idx not in matched_gt_idx_tp or est_idx not in matched_est_idx_tp:
                    stats["A3"] += 1
                    gt_pitch = gt_notes[gt_idx]["pitch"]
                    est_pitch = est_notes[est_idx]["pitch"]
                    band = get_pitch_band(gt_pitch)
                    stats["band"][band]["A3"] += 1
                    stats["track"][track]["A3"] += 1
                    stats["genre"][genre]["A3"] += 1
                    
                    diff = est_pitch - gt_pitch
                    if abs(abs(diff) - 12) < 1.0:
                        stats["A3_octave"] += 1
                    elif abs(abs(diff) - 1) < 1.0:
                        stats["A3_semitone"] += 1

        except Exception as e:
            print(f"Error on {track}: {e}")
            import traceback
            traceback.print_exc()
            
    print("\n==================================================")
    print("=== Category A (Pitch Error) Sub-classification ===")
    print(f"Total A1 (False Negative): {stats['A1']}")
    print(f"Total A2 (False Positive): {stats['A2']}")
    print(f"Total A3 (Pitch Mismatch): {stats['A3']}")
    print()
    print("--- Distribution by Track ---")
    for track, s in stats['track'].items():
        print(f"  {track}: A1={s['A1']}, A2={s['A2']}, A3={s['A3']}")
        
    print("\n--- Distribution by Pitch Band ---")
    for band, s in stats['band'].items():
        print(f"  {band}: A1={s['A1']}, A2={s['A2']}, A3={s['A3']}")
        
    print("\n--- Distribution by Genre ---")
    for genre, s in stats['genre'].items():
        print(f"  {genre}: A1={s['A1']}, A2={s['A2']}, A3={s['A3']}")
        
    print("\n--- A1 (False Negative) Details ---")
    print(f"  Dense Passage (<100ms gap): {stats['A1_dense']} / {stats['A1']} ({stats['A1_dense']/max(1,stats['A1'])*100:.1f}%)")
    print(f"  Open String Note: {stats['A1_open_string']} / {stats['A1']} ({stats['A1_open_string']/max(1,stats['A1'])*100:.1f}%)")
    
    print("\n--- A2 (False Positive) Details ---")
    print(f"  Octave Overtone (GT note octave below): {stats['A2_octave_overtone']} / {stats['A2']} ({stats['A2_octave_overtone']/max(1,stats['A2'])*100:.1f}%)")
    
    print("\n--- A3 (Pitch Mismatch) Details ---")
    print(f"  Octave Error (±12): {stats['A3_octave']} / {stats['A3']} ({stats['A3_octave']/max(1,stats['A3'])*100:.1f}%)")
    print(f"  Semitone Error (±1): {stats['A3_semitone']} / {stats['A3']} ({stats['A3_semitone']/max(1,stats['A3'])*100:.1f}%)")
    print("==================================================")

if __name__ == '__main__':
    main()
