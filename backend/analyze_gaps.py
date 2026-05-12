"""Analyze BP-only notes that are currently discarded."""
import sys, json
import numpy as np
sys.path.insert(0, '.')

session_dir = r'D:\Music\nextchord-solotab\uploads\20260512-073742'

# Load BP and MoE notes separately
# We need to re-run the detection or find cached results
# Check if bp_notes and moe_notes are saved
import os, glob

# Check session files
files = os.listdir(session_dir)
print("Session files:", [f for f in files if 'note' in f.lower() or 'bp' in f.lower() or 'moe' in f.lower()])

# We need to simulate the fusion to see what BP found that MoE didn't
# Let's use the e2e test session which has fresh data
test_dir = r'D:\Music\nextchord-solotab\uploads\romance_e2e_test'
if os.path.exists(test_dir):
    test_files = os.listdir(test_dir)
    print("Test session files:", [f for f in test_files if 'note' in f.lower()])

# Load the final notes and check for gaps in Romance pattern
with open(f'{session_dir}/notes_assigned.json', 'r', encoding='utf-8') as f:
    notes = json.load(f)

# BPM-corrected beats
bpm = 89.1
true_interval = 60.0 / bpm
first_note = min(float(n['start']) for n in notes)
first_beat = first_note - 0.01
true_beats = [first_beat + i * true_interval for i in range(200)]

# Check each beat for note count
print(f"\nTotal notes: {len(notes)}")
print(f"\n--- Notes per beat (first 36 beats = 12 bars) ---")
missing_beats = []
for bi in range(min(36, len(true_beats)-1)):
    bt = true_beats[bi]
    nbt = true_beats[bi+1]
    beat_notes = [n for n in notes if bt <= float(n['start']) < nbt]
    bar = bi // 3 + 1
    beat_in_bar = bi % 3 + 1
    count = len(beat_notes)
    pitches = [int(n['pitch']) for n in beat_notes]
    
    # Expected: melody notes should include something on string 1, 2, and 3
    high_notes = [p for p in pitches if p > 52]
    
    marker = ""
    if count < 3 and count > 0:
        marker = " << MISSING"
        missing_beats.append((bar, beat_in_bar, bt, nbt, count, pitches))
    elif count == 0:
        marker = " << EMPTY"
        missing_beats.append((bar, beat_in_bar, bt, nbt, count, pitches))
    
    print(f"  M{bar} B{beat_in_bar}: {count} notes, pitches={pitches}{marker}")

print(f"\nBeats with missing notes: {len(missing_beats)}")
for bar, bib, bt, nbt, count, pitches in missing_beats:
    print(f"  M{bar} B{bib}: time={bt:.3f}-{nbt:.3f}s, count={count}, pitches={pitches}")
