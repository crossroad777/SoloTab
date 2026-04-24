import sys
import os
import jams
import numpy as np
import mir_eval

# Add backend to path so we can import guitar_transcriber
project_root = r"D:\Music\nextchord-solotab"
sys.path.insert(0, os.path.join(project_root, "backend"))
from guitar_transcriber import transcribe_guitar

def evaluate_song(wav_path, jams_path):
    print(f"\n--- Evaluating: {os.path.basename(wav_path)} ---")
    
    # 1. Load Ground Truth
    jam = jams.load(jams_path)
    gt_intervals = []
    gt_pitches = []
    
    for anno in jam.search(namespace='note_midi'):
        for obs in anno.data:
            onset = obs.time
            offset = obs.time + obs.duration
            pitch = round(obs.value)
            gt_intervals.append([onset, offset])
            gt_pitches.append(pitch)
            
    gt_intervals = np.array(gt_intervals)
    gt_pitches = np.array(gt_pitches)
    
    sort_idx = np.argsort(gt_intervals[:, 0])
    gt_intervals = gt_intervals[sort_idx]
    gt_pitches = gt_pitches[sort_idx]
    
    print(f"Ground Truth Notes: {len(gt_pitches)}")
    
    # 2. Predict using baseline_model (native transcriber without ensemble)
    # Using onset_threshold=0.5, tuning=standard
    res = transcribe_guitar(wav_path, onset_threshold=0.5)
    
    est_intervals = []
    est_pitches = []
    for n in res['notes']:
        onset = n.get('start', 0.0)
        # Without ensemble "minimum duration" tricks, we use raw duration or default
        offset = n.get('end', onset + 0.2) 
        est_intervals.append([onset, offset])
        est_pitches.append(n.get('pitch', 0))
        
    print(f"Predicted Notes: {len(est_pitches)}")
    
    if not est_intervals:
        print("TDR F1: 0.0000 (No notes predicted)")
        return 0.0
        
    ei = np.array(est_intervals)
    ep = np.array(est_pitches)
    
    # 3. Calculate TDR F1 using mir_eval
    # We evaluate using onset_tolerance=0.05, pitch_tolerance=50 (exact midi match)
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        gt_intervals, mir_eval.util.midi_to_hz(gt_pitches),
        ei, mir_eval.util.midi_to_hz(ep),
        onset_tolerance=0.05, pitch_tolerance=50.0  # standard strict TDR params
    )
    
    print(f"==> Precision: {p:.4f} | Recall: {r:.4f}")
    print(f"==> TDR F1:    {f:.4f}")
    return f

if __name__ == "__main__":
    audio_dir = r"D:\Music\datasets\GuitarSet\audio_mono-mic"
    jams_dir  = r"D:\Music\datasets\GuitarSet\annotation"
    
    test_cases = [
        ("00_BN1-129-Eb_comp_mic.wav", "00_BN1-129-Eb_comp.jams"),
        ("00_Jazz1-130-D_solo_mic.wav", "00_Jazz1-130-D_solo.jams"),
        ("00_Funk1-114-Ab_comp_mic.wav", "00_Funk1-114-Ab_comp.jams")
    ]
    
    f1_scores = []
    for wav_file, jams_file in test_cases:
        wav_path = os.path.join(audio_dir, wav_file)
        jams_path = os.path.join(jams_dir, jams_file)
        try:
            score = evaluate_song(wav_path, jams_path)
            f1_scores.append(score)
        except Exception as e:
            print(f"Error evaluating {wav_file}: {e}")
            
    print("\n" + "="*40)
    print(f"AVERAGE TDR F1 across 3 songs: {np.mean(f1_scores):.4f}")
    print("="*40)
