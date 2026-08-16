import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import sys
import os

# backend フォルダ内から実行することを想定
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from string_assigner import assign_strings_dp, TUNINGS
from finger_assigner import assign_fingers

def run_test(test_name, notes, tuning_name):
    print(f"\n{'='*20} {test_name} {'='*20}")
    tuning = TUNINGS.get(tuning_name, TUNINGS["standard"])
    print(f"Tuning: {tuning_name} {tuning}")
    
    # 1. 弦・フレット割り当て
    assigned = assign_strings_dp(notes, tuning=tuning, max_fret=24)
    
    # 2. 運指割り当て
    final = assign_fingers(assigned)
    
    # 3. 結果表示
    print(f"{'Pitch':>5} | {'String':>6} | {'Fret':>4} | {'Finger':>6}")
    print("-" * 35)
    for n in final:
        print(f"{n['pitch']:>5} | {n.get('string', '?'):>6} | {n.get('fret', '?'):>4} | {n.get('left_hand_finger', '?'):>6}")

# =====================================================================
# テストケース1: DADGAD変則チューニングでの開放弦アルペジオ
# =====================================================================
dadgad_notes = [
    {"start": 0.0, "pitch": 38, "duration": 2.0}, # 6弦 D (開放)
    {"start": 0.5, "pitch": 57, "duration": 1.0}, # 2弦 A (開放)
    {"start": 1.0, "pitch": 62, "duration": 1.0}, # 1弦 D (開放)
    {"start": 1.5, "pitch": 50, "duration": 1.0}, # 4弦 D (開放)
    {"start": 2.0, "pitch": 55, "duration": 1.0}, # 3弦 G (開放)
]
run_test("DADGAD Open Arpeggio", dadgad_notes, "dadgad")

# =====================================================================
# テストケース2: ハイポジション・タッピング風フレーズ (12f-15f-17f)
# =====================================================================
tapping_notes = [
    {"start": 0.0, "pitch": 76, "duration": 0.2}, # 1弦 12f
    {"start": 0.2, "pitch": 79, "duration": 0.2}, # 1弦 15f
    {"start": 0.4, "pitch": 81, "duration": 0.2}, # 1弦 17f
    {"start": 0.6, "pitch": 79, "duration": 0.2}, # 1弦 15f
    {"start": 0.8, "pitch": 76, "duration": 0.2}, # 1弦 12f
]
run_test("High Fret Tapping Pattern", tapping_notes, "standard")

# =====================================================================
# テストケース3: DADGAD ハイポジ・メロディ + 開放弦ドローン
# =====================================================================
dadgad_drone_notes = [
    {"start": 0.0, "pitch": 38, "duration": 2.0}, # 6弦 D (開放) - ベースドローン
    {"start": 0.5, "pitch": 57, "duration": 0.5}, # 4弦 7f (A3) - メロディ
    {"start": 1.0, "pitch": 62, "duration": 0.5}, # 3弦 7f (D4) - メロディ
    {"start": 1.5, "pitch": 62, "duration": 1.0}, # ★ 1弦 D (開放) - 高音ドローン
    {"start": 2.5, "pitch": 59, "duration": 0.5}, # 3弦 9f (B3) - メロディ
    {"start": 3.0, "pitch": 38, "duration": 2.0}, # ★ 6弦 D (開放) - ベースドローン
]
run_test("DADGAD Melody + Open Drone", dadgad_drone_notes, "dadgad")

# =====================================================================
# テストケース3b: DADGAD 真のハイポジション + 開放弦ドローン
# =====================================================================
dadgad_highpos_drone = [
    {"start": 0.0, "pitch": 69, "duration": 0.5},  # A4 → 1弦7f (最低fret 7)
    {"start": 0.5, "pitch": 71, "duration": 0.5},  # B4 → 1弦9f (最低fret 9)
    {"start": 1.0, "pitch": 69, "duration": 0.5},  # A4 → 1弦7f
    {"start": 1.5, "pitch": 71, "duration": 0.5},  # B4 → 1弦9f
    {"start": 2.0, "pitch": 62, "duration": 1.0},  # ★ D4 ドローン → 1弦開放(0f)か？
    {"start": 3.0, "pitch": 69, "duration": 0.5},  # A4 → 1弦7f
    {"start": 3.5, "pitch": 71, "duration": 0.5},  # B4 → 1弦9f
]
run_test("DADGAD High-Pos Melody + Open Drone", dadgad_highpos_drone, "dadgad")


