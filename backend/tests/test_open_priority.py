"""Compare before/after open string priority on Romance notes"""
import json, copy

# Load original assigned notes (with cnn_string_probs intact)
notes = json.load(open('D:/Music/nextchord-solotab/uploads/20260428-205331-yt-8123dc/notes_assigned.json'))

# Show original first 5 notes
print("=== BEFORE (original) ===")
for n in [x for x in notes if x['start'] < 5.0]:
    print(f"  t={n['start']:.3f} pitch={n['pitch']} str={n['string']} fret={n['fret']}")

# Apply open string priority rule manually
OPEN_STRING_PROB_THRESHOLD = 0.01
STANDARD_TUNING = [40, 45, 50, 55, 59, 64]
MAX_FRET = 19

modified = copy.deepcopy(notes)
changed = 0
for note in modified:
    cnn_probs = note.get('cnn_string_probs')
    if not cnn_probs:
        continue
    
    pitch = note['pitch']
    # Find all possible (string, fret) positions
    positions = []
    for i, op in enumerate(STANDARD_TUNING):
        s = 6 - i  # string number (6=low E, 1=high E)
        f = pitch - op
        if 0 <= f <= MAX_FRET:
            positions.append((s, f))
    
    # Find open string positions
    open_pos = [(s, f) for s, f in positions if f == 0]
    if not open_pos:
        continue
    
    # Check if current assignment is NOT open string
    if note['fret'] == 0:
        continue
    
    # Check CNN prob for open string candidate
    for os_s, os_f in open_pos:
        prob = cnn_probs.get(str(os_s), 0)
        if prob >= OPEN_STRING_PROB_THRESHOLD:
            old_s, old_f = note['string'], note['fret']
            note['string'] = os_s
            note['fret'] = os_f
            changed += 1
            if note['start'] < 5.0:
                print(f"  CHANGED t={note['start']:.3f} pitch={pitch}: str{old_s}f{old_f} -> str{os_s}f{os_f} (prob={prob:.4f})")
            break

print(f"\n=== AFTER (open string priority) ===")
for n in [x for x in modified if x['start'] < 5.0]:
    print(f"  t={n['start']:.3f} pitch={n['pitch']} str={n['string']} fret={n['fret']}")

print(f"\nTotal changed: {changed}/{len(notes)}")
open_before = sum(1 for n in notes if n['fret'] == 0)
open_after = sum(1 for n in modified if n['fret'] == 0)
print(f"Open strings: {open_before} -> {open_after}")
