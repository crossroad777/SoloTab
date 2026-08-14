import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
from music_theory import validate_notes_by_music_theory
from string_assigner import assign_strings_dp
from finger_assigner import assign_fingers

def main():
    print("=== Step 2: Pipeline Stage Note Count Trace ===\n")
    
    session_id = "test_session_0" # Pattern 1: Single note scale
    wav_path = os.path.join(os.path.dirname(__file__), "temp_sessions", session_id, "input.wav")
    
    # 1. Basic Pitch
    _, midi_data, _ = predict(wav_path, model_or_model_path=ICASSP_2022_MODEL_PATH)
    bp_notes = midi_data.instruments[0].notes if midi_data.instruments else []
    
    # Normalize notes
    notes = []
    for note in bp_notes:
        notes.append({
            "start": float(note.start),
            "end": float(note.end),
            "pitch": int(note.pitch),
            "velocity": float(note.velocity) / 127.0 if hasattr(note, "velocity") else 0.5,
        })
    print(f"[Stage 1] Basic Pitch: {len(notes)} notes")
    
    # 2. music_theory (validate_notes_by_music_theory)
    # validate_notes_by_music_theory(notes, beats, chords, key, threshold=0.5)
    beats = [] # mock beats
    valid_notes = validate_notes_by_music_theory(notes, beats, [], "C", threshold=0.5)
    print(f"[Stage 2] music_theory (MVS filter): {len(valid_notes)} notes")
    
    # 3. string_assigner
    from solotab_utils import STANDARD_TUNING
    stringed_notes = assign_strings_dp(valid_notes, STANDARD_TUNING, 24, 0.0)
    print(f"[Stage 3] string_assigner: {len(stringed_notes)} notes")
    
    # 4. finger_assigner
    finger_notes = assign_fingers(stringed_notes)
    print(f"[Stage 4] finger_assigner: {len(finger_notes)} notes")

if __name__ == "__main__":
    main()
