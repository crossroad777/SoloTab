"""
test_transformer_v3_romance.py
"""
import sys, os, pathlib, json
import numpy as np

sys.path.insert(0, os.path.abspath("backend"))

from solotab_utils import STANDARD_TUNING
from string_assigner import _load_fingering_transformer, _transformer_string_probs, _transformer_first_assign, _group_simultaneous, assign_strings_dp

# 参照 GT データの読み込み
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

# 1. transformer_first_assign のテスト
groups = _group_simultaneous([dict(n) for n in gt_notes], threshold=0.03)
tf_assigned = _transformer_first_assign(groups, tuning, 14, 0.0)

matches = 0
for i in range(len(gt_notes)):
    if gt_notes[i]["string"] == tf_assigned[i]["string"] and gt_notes[i]["fret"] == tf_assigned[i]["fret"]:
        matches += 1
        
print(f"Transformer-First Direct Match Rate: {matches} / {len(gt_notes)} ({matches / len(gt_notes):.2%})")
