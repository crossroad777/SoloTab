import json
import mir_eval
import numpy as np
import jams
import sys

sys.path.insert(0, r'd:\Music\nextchord-solotab\backend')
from note_filter import apply_all_filters

def get_sota_score():
    jams_path = r'd:\Music\nextchord-solotab\datasets\GuitarSet\00_Jazz1-130-D_solo.jams'
    jam = jams.load(jams_path)
    
    gt_intervals = []
    gt_pitches = []
    
    note_annos = jam.search(namespace='note_midi')
    for anno in note_annos:
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
    
    with open('d:/Music/nextchord-solotab/debug_ensemble_raw.json', 'r') as f:
        raw_data = json.load(f)
        
    model_notes = raw_data.get("fretnet", [])
    
    # We apply ONLY duplicate removal, no velocity threshold because these are single-model outputs
    f_res = apply_all_filters(
        model_notes, 
        velocity_threshold=0.0, 
        min_duration_sec=0.04, 
        max_notes_per_beat=20
    )
    predicted_notes = f_res['notes']
    
    est_intervals = []
    est_pitches = []
    for n in predicted_notes:
        onset = n.get('start', 0.0)
        offset = n.get('end', onset + n.get('duration', 0.25))
        pitch = n.get('pitch', 0)
        est_intervals.append([onset, offset])
        est_pitches.append(pitch)
        
    est_intervals = np.array(est_intervals)
    est_pitches = np.array(est_pitches)
    
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        gt_intervals, mir_eval.util.midi_to_hz(gt_pitches),
        est_intervals, mir_eval.util.midi_to_hz(est_pitches),
        onset_tolerance=0.05, pitch_tolerance=50.0
    )
    
    print(f"=== FretNet Pure Component SOTA (GuitarSet) ===")
    print(f"Val F1: {f:.4f} (Target: >= 0.75-0.80)")
    print(f"P:      {p:.4f}")
    print(f"R:      {r:.4f}")

if __name__ == '__main__':
    get_sota_score()
