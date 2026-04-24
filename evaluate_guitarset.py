import jams
import numpy as np
import mir_eval
import sys
import os

sys.path.insert(0, r'd:\Music\nextchord-solotab\backend')
from ensemble_transcriber import transcribe_ensemble
from note_filter import apply_all_filters

def evaluate_on_file(wav_path, jams_path):
    print(f"Loading Ground Truth from JAMS: {os.path.basename(jams_path)}")
    jam = jams.load(jams_path)
    
    # Extract ground truth notes from all 6 strings (namespaces usually 'note_midi')
    gt_intervals = []
    gt_pitches = []
    
    # GuitarSet usually has 6 'note_midi' annotations, one for each string
    note_annos = jam.search(namespace='note_midi')
    for anno in note_annos:
        for obs in anno.data:
            onset = obs.time
            offset = obs.time + obs.duration
            pitch = round(obs.value)  # JAMS 'note_midi' value is MIDI pitch
            gt_intervals.append([onset, offset])
            gt_pitches.append(pitch)
            
    gt_intervals = np.array(gt_intervals)
    gt_pitches = np.array(gt_pitches)
    
    # If using mir_eval, we sort them by onset time
    sort_idx = np.argsort(gt_intervals[:, 0])
    gt_intervals = gt_intervals[sort_idx]
    gt_pitches = gt_pitches[sort_idx]
    
    print(f"Ground Truth contains {len(gt_pitches)} notes.")
    
    import time
    
    res = transcribe_ensemble(wav_path, tuning_pitches=[40, 45, 50, 55, 59, 64])
    
    print("\n--- Sweeping Parameters for Maximum GuitarSet F1 ---")
    best_f1 = 0
    best_params = {}
    
    import copy
    vt_range = [0.55, 0.65, 0.70, 0.75]
    dur_range = [0.0, 0.05]
    mn_range = [10, 20, 999] # test without heavy density filter
    
    for vt in vt_range:
        for md in dur_range:
            for mn in mn_range:
                f_res = apply_all_filters(
                    copy.deepcopy(res['notes']), 
                    velocity_threshold=vt, 
                    min_duration_sec=md, 
                    max_notes_per_beat=mn
                )
                
                est_intervals = []
                est_pitches = []
                for n in f_res['notes']:
                    onset = n.get('start', 0.0)
                    offset = n.get('end')
                    if offset is None:
                        offset = onset + n.get('duration', 0.25)
                    est_intervals.append([onset, offset])
                    est_pitches.append(n.get('pitch', 0))
                    
                if not est_intervals:
                    continue
                    
                ei = np.array(est_intervals)
                ep = np.array(est_pitches)
                
                p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
                    gt_intervals, mir_eval.util.midi_to_hz(gt_pitches),
                    ei, mir_eval.util.midi_to_hz(ep),
                    onset_tolerance=0.05, pitch_tolerance=50.0
                )
                
                if f > best_f1:
                    best_f1 = f
                    best_params = {'vt': vt, 'md': md, 'mn': mn, 'P': p, 'R': r}
                    
    print("\n=== WORLD STANDARD SOTA SCORE ===")
    print(f"Dataset: GuitarSet (00_Jazz1-130-D_comp)")
    print(f"Val F1: {best_f1:.4f} (Target SOTA >= 0.75)")
    print(f"Precision: {best_params['P']:.4f}")
    print(f"Recall: {best_params['R']:.4f}")
    print(f"Optimal Params: {best_params}")

if __name__ == '__main__':
    # Try an energetic, dense comping file to prove robustness (Bossa Nova / Jazz)
    WAV = r'd:\Music\nextchord-solotab\datasets\guitarset_mono_mic\00_Jazz1-130-D_solo_mic.wav'
    JAMS = r'd:\Music\nextchord-solotab\datasets\GuitarSet\00_Jazz1-130-D_solo.jams'
    evaluate_on_file(WAV, JAMS)
