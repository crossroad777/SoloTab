"""LSTM fingering model benchmark on GuitarSet"""
import json, sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jams
from fingering_model import load_fingering_model, predict_strings, STANDARD_TUNING

annotation_dir = r"D:\Music\Datasets\GuitarSet\annotation"

# Load model
fm = load_fingering_model()
if fm is None:
    print("Model not found!")
    sys.exit(1)

jams_files = sorted(glob.glob(os.path.join(annotation_dir, "*.jams")))

total_correct = 0
total_notes = 0

for jf in jams_files:
    jam = jams.load(jf)
    gt_notes = []
    note_midi_idx = 0
    
    for ann in jam.annotations:
        if ann.namespace != "note_midi":
            continue
        string_num = 6 - note_midi_idx
        note_midi_idx += 1
        if string_num < 1 or string_num > 6:
            continue
        string_idx = 6 - string_num
        
        for obs in ann.data:
            midi_pitch = int(round(obs.value))
            start = float(obs.time)
            duration = float(obs.duration)
            fret = midi_pitch - STANDARD_TUNING[string_idx]
            if fret < 0 or fret > 19:
                continue
            gt_notes.append({
                'start': start,
                'pitch': midi_pitch,
                'duration': duration,
                'gt_string': string_num,
                'gt_fret': fret,
            })
    
    if not gt_notes:
        continue
    
    gt_notes.sort(key=lambda n: n['start'])
    
    # Predict
    predicted = predict_strings(gt_notes, tuning=STANDARD_TUNING)
    
    for n in predicted:
        if n['string'] == n['gt_string']:
            total_correct += 1
        total_notes += 1

acc = total_correct / total_notes if total_notes > 0 else 0
print(f"\nLSTM Fingering Model Benchmark:")
print(f"  String accuracy: {total_correct}/{total_notes} = {acc:.4f} ({acc*100:.2f}%)")
print(f"  (Compare: CNN-first = 96.45%)")
