import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import json

def get_expected_fret(note, tuning):
    cnn = note.get("cnn_string_probs", {})
    if not cnn: return 0
    try:
        best_s = max(cnn, key=cnn.get)
        return note["pitch"] - tuning[int(best_s) - 1]
    except: return 0

# Check tuning
tuning = [64, 59, 55, 50, 45, 40]
pred = json.load(open(r'uploads/20260814-013518-2a384c/notes_assigned.json', encoding='utf-8'))
if isinstance(pred, dict) and 'notes' in pred: pred = pred['notes']
for n in pred[:10]:
    print(n['pitch'], get_expected_fret(n, tuning))
