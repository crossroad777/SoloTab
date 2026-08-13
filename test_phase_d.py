import sys
from pathlib import Path
sys.path.append('backend')
from string_assigner import assign_strings_dp
from finger_assigner import assign_fingers

notes = [
    {"start": 0.0, "pitch": 64},  # E4
    {"start": 0.5, "pitch": 65},  # F4
    {"start": 1.0, "pitch": 67}   # G4
]

tuning = [40, 45, 50, 55, 59, 64]

print("--- Default ---")
notes_default = assign_strings_dp([dict(n) for n in notes], tuning=tuning)
notes_default = assign_fingers(notes_default)
for n in notes_default:
    print(f"pitch: {n['pitch']}, string: {n['string']}, fret: {n['fret']}, finger: {n.get('left_hand_finger')}")

print("\n--- Forced Anchor on F4 (string 2, fret 6, finger 3) ---")
forced_pos = {(65, 0.5): (2, 6)}
forced_fingers = {(65, 0.5): 3}

notes_forced = assign_strings_dp([dict(n) for n in notes], tuning=tuning, forced_positions=forced_pos)
notes_forced = assign_fingers(notes_forced, forced_fingers=forced_fingers)
for n in notes_forced:
    print(f"pitch: {n['pitch']}, string: {n['string']}, fret: {n['fret']}, finger: {n.get('left_hand_finger')}")
