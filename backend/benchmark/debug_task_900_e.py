"""
debug_task_900_e.py — romance.gp5 MIDIバイパス先頭20ノートの並列ダンプ
"""

import sys
import os
import pathlib
import json
import numpy as np

sys.path.insert(0, os.path.abspath("backend"))

from solotab_utils import TUNINGS, STANDARD_TUNING
from string_assigner import assign_strings_dp, get_possible_positions

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
            "string": n["string"],
            "fret": n["fret"],
            "pitch": n["pitch"],
            "role": n.get("role", "melody"),
            "velocity": 0.8
        })
        current_t += 0.25

tuning = STANDARD_TUNING # [40, 45, 50, 55, 59, 64] (6弦->1弦)

# Viterbi運指割り当て実行 (直接呼出し)
assigned = assign_strings_dp(gt_notes, tuning=tuning)

print(f"Total Notes: {len(assigned)}")
print("\n--- FIRST 20 NOTES MAPPING DUMP ---")
print(f"{'Idx':<4} | {'Input Pitch':<11} | {'GT(str,frt)':<12} | {'Assigned(str,frt)':<18} | {'Computed Pitch':<14} | {'Match?':<6} | {'Pitch OK?':<9}")
print("-" * 85)

pitch_violations_before = 0
string_fret_matches = 0

for i in range(min(20, len(assigned))):
    inp_p = gt_notes[i]["pitch"]
    gt_s = gt_notes[i]["string"]
    gt_f = gt_notes[i]["fret"]
    
    as_s = assigned[i]["string"]
    as_f = assigned[i]["fret"]
    
    # 弦 s (1-6) の開放弦ピッチ: tuning[6 - as_s]
    comp_p = tuning[6 - as_s] + as_f
    
    is_match = (gt_s == as_s and gt_f == as_f)
    is_pitch_ok = (inp_p == comp_p)
    
    if not is_pitch_ok:
        pitch_violations_before += 1
    if is_match:
        string_fret_matches += 1
        
    print(f"{i+1:<4} | {inp_p:<11} | ({gt_s}, {gt_f}){'':<6} | ({as_s}, {as_f}){'':<12} | {comp_p:<14} | {str(is_match):<6} | {str(is_pitch_ok):<9}")

print(f"\nFirst 20 String/Fret Matches: {string_fret_matches} / 20")
print(f"Pitch Invariant Violations (First 20): {pitch_violations_before}")
