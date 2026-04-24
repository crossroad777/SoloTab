import json
import mir_eval
import numpy as np
import jams

def evaluate_raw(model_name):
    jams_path = r'd:\Music\nextchord-solotab\datasets\GuitarSet\00_Jazz1-130-D_comp.jams'
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
        
    model_notes = raw_data.get(model_name, [])
    
    est_intervals = []
    est_pitches = []
    for n in model_notes:
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
    
    print(f"=== {model_name.upper()} Standalone (No Ensemble, No Filter) ===")
    print(f"Val F1: {f:.4f}")
    print(f"P:      {p:.4f}")
    print(f"R:      {r:.4f}")
    print(f"Predicted: {len(model_notes)} | Ground Truth: {len(gt_pitches)}")

if __name__ == '__main__':
    evaluate_raw("crnn")
    evaluate_raw("fretnet")
    evaluate_raw("basic_pitch")
