import os
import glob
import numpy as np
import mir_eval
import librosa
from tqdm import tqdm
import argparse
import sys
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
mt_python_dir = os.path.join(project_root, "music-transcription", "python")
if mt_python_dir not in sys.path:
    sys.path.insert(0, mt_python_dir)

from data_processing.preparation import extract_annotations_from_jams
import moe_ensemble_transcriber as et

OUTALL_DIR = r"D:\Music\all_jams_midi_V2_60000_tracks\outall"

def parse_args():
    parser = argparse.ArgumentParser(description="Objectively benchmark the Domain-Ensemble")
    parser.add_argument("--test_dir", type=str, required=True, 
                        help="Path to folder containing .flac/.wav and .jams file pairs.")
    parser.add_argument("--output_csv", type=str, default="benchmark_results.csv")
    return parser.parse_args()

def evaluate_track(wav_path, jams_path):
    # 1. Load Ground Truth
    gt_notes = extract_annotations_from_jams(jams_path)
    ref_intervals = []
    ref_pitches = []
    for n in gt_notes:
        onset, offset, string_num, fret_num, pitch_midi = n
        ref_intervals.append([onset, offset])
        ref_pitches.append(librosa.midi_to_hz(pitch_midi))
    
    ref_intervals = np.array(ref_intervals)
    if len(ref_intervals) == 0:
        ref_intervals = np.empty((0, 2))
    ref_pitches = np.array(ref_pitches)

    # 2. Run Ensemble Inference
    try:
        # Output saving is disabled or redirected
        result = et.transcribe_dynamic_ensemble(wav_path)
        est_notes = result["notes"]
    except Exception as e:
        print(f"Error inferring track {wav_path}: {e}")
        return None

    est_intervals = []
    est_pitches = []
    for n in est_notes:
        est_intervals.append([n["start"], n["end"]])
        est_pitches.append(librosa.midi_to_hz(n["pitch"]))
    
    est_intervals = np.array(est_intervals)
    if len(est_intervals) == 0:
        est_intervals = np.empty((0, 2))
    est_pitches = np.array(est_pitches)

    # 3. Compute MIR_EVAL Transcription Metrics
    # Standard mir_eval.transcription computes matches based on 50ms onset tolerance and 50 cents pitch tolerance
    try:
        scores = mir_eval.transcription.evaluate(
            ref_intervals, ref_pitches, est_intervals, est_pitches,
            onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=0.2, offset_min_tolerance=0.05
        )
    except Exception as e:
        print(f"mir_eval error: {e}")
        return None

    # Scores usually include: Precision, Recall, F-measure, Precision_no_offset, Recall_no_offset, F-measure_no_offset
    return {
        "track": os.path.basename(wav_path),
        "notes_gt": len(ref_pitches),
        "notes_est": len(est_pitches),
        "Onset_Pitch_F1": scores.get("F-measure_no_offset", 0.0),
        "Onset_Pitch_P": scores.get("Precision_no_offset", 0.0),
        "Onset_Pitch_R": scores.get("Recall_no_offset", 0.0),
        "Onset_Pitch_Offset_F1": scores.get("F-measure", 0.0)
    }

def main():
    args = parse_args()
    if not os.path.isdir(args.test_dir):
        print(f"Error: {args.test_dir} is not a valid directory.")
        sys.exit(1)

    wav_files = glob.glob(os.path.join(args.test_dir, "**", "*.wav"), recursive=True)
    flac_files = glob.glob(os.path.join(args.test_dir, "**", "*.flac"), recursive=True)
    audio_files = wav_files + flac_files

    pairs = []
    for audio_path in audio_files:
        base_name = os.path.splitext(audio_path)[0]
        # Search for jams in same directory
        jams_path = None
        if os.path.exists(base_name + ".jams"):
            jams_path = base_name + ".jams"
        else:
            # 2. Search in parent directory
            dir_name = os.path.dirname(audio_path)
            potential_jams = glob.glob(os.path.join(dir_name, "*.jams"))
            if potential_jams: 
                jams_path = potential_jams[0]
            else:
                # 3. Search in OUTALL_DIR (Specialized datasets)
                # Parent dir name corresponds to the ID in outall
                parent_id = os.path.basename(dir_name)
                outall_jams_dir = os.path.join(OUTALL_DIR, parent_id)
                if os.path.exists(outall_jams_dir):
                    potential_jams = glob.glob(os.path.join(outall_jams_dir, "*.jams"))
                    if potential_jams:
                        jams_path = potential_jams[0]

        if jams_path:
            pairs.append((audio_path, jams_path))

    print(f"Found {len(pairs)} tracks with JAMS annotations for benchmarking.")

    if len(pairs) == 0:
        print("No paired audio/jams files found. Ensure the directory contains matching base names.")
        sys.exit(0)

    results = []
    for wav, jams in tqdm(pairs, desc="Benchmarking Tracks"):
        res = evaluate_track(wav, jams)
        if res:
            results.append(res)
            print(f"\n[Result] {res['track']} - F1: {res['Onset_Pitch_F1']:.4f}")

    if len(results) > 0:
        # Aggregation
        avg_f1 = np.mean([r["Onset_Pitch_F1"] for r in results])
        avg_p = np.mean([r["Onset_Pitch_P"] for r in results])
        avg_r = np.mean([r["Onset_Pitch_R"] for r in results])
        
        print("\n\n" + "="*50)
        print("    ベンチマーク最終結果 (BENCHMARK RESULTS)")
        print("="*50)
        print(f"Total Tracks Evaluated: {len(results)}")
        print(f"Average Onset+Pitch F1:     {avg_f1:.4f}")
        print(f"Average Onset+Pitch Precision: {avg_p:.4f}")
        print(f"Average Onset+Pitch Recall:    {avg_r:.4f}")
        print("="*50)

        # Save to CSV
        import csv
        with open(args.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = results[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"Details saved to {args.output_csv}")

if __name__ == "__main__":
    main()
