import sys
import os
import jams
import numpy as np
import torch

project_root = r"D:\Music\nextchord-solotab"
sys.path.insert(0, os.path.join(project_root, "backend"))
sys.path.insert(0, os.path.join(project_root, "music-transcription", "python"))

from guitar_transcriber import transcribe_guitar
from evaluation.metrics import calculate_note_level_metrics

def evaluate_song_native(wav_path, jams_path):
    print(f"\n--- Evaluating: {os.path.basename(wav_path)} ---")
    
    jam = jams.load(jams_path)
    gt_notes_raw = []
    
    # Extract ground truth in the raw tensor format expected by metrics.py
    # Each row: [start_time, end_time, string_idx, fret_val, pitch]
    # Note: JAMS string namespace might not be directly ordered 0-5. 
    # Usually in GuitarSet, annotation.annotation_metadata.data_source tells us the string (0-5)
    
    note_annos = jam.search(namespace='note_midi')
    for anno in note_annos:
        # String index usually from 0 to 5, mapping from high E (0) to low E (5) or vice versa.
        # In GuitarSet JAMS, annotation_metadata is often just integer 0-5
        string_idx = anno.annotation_metadata.data_source
        if string_idx is None:
            # Fallback
            string_idx = 0
            
        for obs in anno.data:
            onset = obs.time
            offset = obs.time + obs.duration
            pitch = round(obs.value)
            
            # For GuitarSet JAMS: The string number in data_source is 0 (low E) to 5 (high E)
            # We want string index 0 (high E) to 5 (low E) to match model if needed, but let's just 
            # mimic raw extraction
            gt_notes_raw.append([onset, offset, string_idx, -1, pitch])
            
    # Wait, the JAMS doesn't explicitly store fret! 
    # Let's use the actual dataset code to load exact raw_labels so we don't mess up the TDR calculation!
    return

if __name__ == "__main__":
    from data_processing.preparation import extract_annotations_from_jams
    
    audio_dir = r"D:\Music\datasets\GuitarSet\audio_mono-mic"
    jams_dir  = r"D:\Music\datasets\GuitarSet\annotation"
    
    test_cases = [
        ("00_BN1-129-Eb_comp_mic.wav", "00_BN1-129-Eb_comp.jams"),
        ("00_Jazz1-130-D_solo_mic.wav", "00_Jazz1-130-D_solo.jams"),
        ("00_Funk1-114-Ab_comp_mic.wav", "00_Funk1-114-Ab_comp.jams")
    ]
    
    tdr_scores = []
    for wav_file, jams_file in test_cases:
        wav_path = os.path.join(audio_dir, wav_file)
        jams_path = os.path.join(jams_dir, jams_file)
        
        print(f"\n--- Evaluating: {wav_file} ---")
        
        raw_gt = extract_annotations_from_jams(jams_path)
        gt_tensor = torch.tensor(raw_gt)
        print(f"Ground Truth Notes: {len(raw_gt)}")
        
        res = transcribe_guitar(wav_path, onset_threshold=0.5)
        # transcriber returns "start" instead of "start_time"
        predicted_notes = []
        for n in res["notes"]:
            predicted_notes.append({
                "start_time": n["start"],
                "string": 6 - n["string"], # map back from 1-6 to 0-5
                "fret": n["fret"]
            })
            
        print(f"Predicted Notes: {len(predicted_notes)}")
        
        metrics_dict = calculate_note_level_metrics(predicted_notes, gt_tensor, onset_window=0.05)
        
        p = metrics_dict["tdr_precision"]
        r = metrics_dict["tdr_recall"]
        f = metrics_dict["tdr_f1"]
        
        print(f"==> TDR Precision: {p:.4f} | Recall: {r:.4f}")
        print(f"==> TDR F1:        {f:.4f}")
        
        tdr_scores.append(f)
        
    print("\n" + "="*40)
    print(f"AVERAGE NATIVE TDR F1 across 3 songs: {np.mean(tdr_scores):.4f}")
    print("="*40)
