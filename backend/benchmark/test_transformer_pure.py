"""
test_transformer_pure.py
"""
import sys, os, pathlib, json
import numpy as np

sys.path.insert(0, os.path.abspath("backend"))

from solotab_utils import STANDARD_TUNING
from string_assigner import _load_fingering_transformer, _transformer_string_probs, _group_simultaneous, _assign_chord_notes

gt_json_path = pathlib.Path("backend/ground_truth/romance_forbidden_games.json")
with open(gt_json_path, "r", encoding="utf-8") as f:
    gt_data = json.load(f)
    
gt_notes = []
current_t = 0.0
for m in gt_data["measures_detailed"]:
    for n in m["notes"]:
        gt_notes.append({
            "start": current_t,
            "end": current_t + 0.22,
            "duration": 0.22,
            "string": n["string"],
            "fret": n["fret"],
            "pitch": n["pitch"],
            "role": n.get("role", "melody"),
            "velocity": 0.8
        })
        current_t += 0.25

tuning = STANDARD_TUNING
max_fret = 14

groups = _group_simultaneous([dict(n) for n in gt_notes], threshold=0.03)
flat_notes = [g[0] for g in groups]

# 初期値
for note in flat_notes:
    pitch = note.get('pitch', 60)
    best_s, best_f = 1, max_fret
    for si, op in enumerate(tuning):
        sn = 6 - si
        f = pitch - op
        if 0 <= f <= max_fret and f < best_f:
            best_s, best_f = sn, f
    note['string'] = best_s
    note['fret'] = best_f

for gi in range(len(groups)):
    group = groups[gi]
    if len(group) > 1:
        prev_f = None
        for pgi in range(gi - 1, -1, -1):
            prev_notes = groups[pgi]
            if prev_notes[0].get('fret') is not None:
                prev_f = [(n.get('string', 1), n.get('fret', 0)) for n in prev_notes]
                break
        chord_result = _assign_chord_notes(group, tuning, max_fret, prev_f)
        for j, note in enumerate(group):
            if j < len(chord_result):
                note['string'] = chord_result[j].get('string', note.get('string', 1))
                note['fret'] = chord_result[j].get('fret', note.get('fret', 0))
        continue

    note = group[0]
    pitch = note.get('pitch', 60)
    probs = _transformer_string_probs(flat_notes, gi, tuning)
    if not probs:
        continue

    valid = {}
    for s in range(1, 7):
        f = pitch - tuning[6 - s]
        if 0 <= f <= max_fret:
            raw_p = probs.get(s, 0)
            valid[s] = (f, raw_p)

    if valid:
        best_s = max(valid, key=lambda s: valid[s][1])
        best_f = valid[best_s][0]
        note['string'] = best_s
        note['fret'] = best_f

result = []
for group in groups:
    result.extend(group)

matches = 0
for i in range(len(gt_notes)):
    if gt_notes[i]["string"] == result[i]["string"] and gt_notes[i]["fret"] == result[i]["fret"]:
        matches += 1
        
print(f"Pure Transformer Match Rate: {matches} / {len(gt_notes)} ({matches / len(gt_notes):.2%})")
