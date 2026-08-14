import os
import sys
import json
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

def evaluate_predictions(y_true, y_pred, window_ms=50.0):
    if not y_true:
        return (0.0, 0.0, 0.0, 0, len(y_pred))
    
    true_notes = sorted(y_true, key=lambda x: x["start"])
    pred_notes = sorted(y_pred, key=lambda x: x["start"])
    
    matched_true = set()
    matched_pred = set()
    
    for i, p in enumerate(pred_notes):
        p_time = p["start"]
        p_pitch = p["pitch"]
        
        best_match_idx = -1
        min_dist = float('inf')
        
        for j, t in enumerate(true_notes):
            if j in matched_true:
                continue
            
            t_time = t["start"]
            t_pitch = t["pitch"]
            
            if p_pitch == t_pitch:
                dist = abs(p_time - t_time)
                if dist <= (window_ms / 1000.0) and dist < min_dist:
                    min_dist = dist
                    best_match_idx = j
                    
        if best_match_idx != -1:
            matched_true.add(best_match_idx)
            matched_pred.add(i)
            
    TP = len(matched_true)
    FP = len(pred_notes) - len(matched_pred)
    FN = len(true_notes) - len(matched_true)
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1, TP, FP, FN

def main():
    print("=== Step 1: Basic Pitch Raw Output Test ===")
    
    patterns = [
        ("test_session_0", "Pattern 1: Single note scale", 15),
        ("test_session_1", "Pattern 2: Chords", 22),
        ("test_session_2", "Pattern 3: High-density fast picking", 32),
        ("test_session_3", "Pattern 4: Arpeggio", 40),
        ("test_session_4", "Pattern 5: 2-voice melody", 16),
    ]
    
    # Needs to match the ground truth arrays from roundtrip_test.py
    # Actually, we just need to report the raw note count. I will also load the SoloTab output to compare!
    
    for session_id, name, gt_count in patterns:
        wav_path = os.path.join(os.path.dirname(__file__), "temp_sessions", session_id, "input.wav")
        solotab_json = os.path.join(os.path.dirname(__file__), "temp_sessions", session_id, "notes_assigned.json")
        
        if not os.path.exists(wav_path):
            print(f"Skipping {name}: WAV not found.")
            continue
            
        print(f"\n--- {name} ---")
        print(f"Ground Truth Notes: {gt_count}")
        
        # 1. Run Basic Pitch
        t0 = time.time()
        try:
            _, midi_data, _ = predict(wav_path, model_or_model_path=ICASSP_2022_MODEL_PATH)
            bp_notes = midi_data.instruments[0].notes if midi_data.instruments else []
            bp_count = len(bp_notes)
            print(f"Basic Pitch Raw Notes: {bp_count} (Took {time.time()-t0:.2f}s)")
        except Exception as e:
            print(f"Basic Pitch Raw Notes: Error ({e})")
            
        # 2. Read SoloTab output (from earlier run)
        if os.path.exists(solotab_json):
            with open(solotab_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                solotab_notes = data.get("notes", [])
            else:
                solotab_notes = data
                
            print(f"SoloTab Final Notes:   {len(solotab_notes)}")
        else:
            print("SoloTab Final Notes:   Not found.")

if __name__ == "__main__":
    main()
