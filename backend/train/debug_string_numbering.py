"""Debug: 人間選好マップ vs Viterbi割り当ての食い違い原因調査"""
import sys, json
sys.path.insert(0, "backend")
import string_assigner

pref = string_assigner._load_human_preference()

# 1. 人間選好マップの内容を確認
print("=== 人間選好マップの上位パターン ===")
for pitch in [40, 45, 48, 50, 52, 55, 57, 59, 60, 62, 64, 67]:
    data = pref.get(str(pitch), {})
    prob = data.get("prob", {})
    total = data.get("total", 0)
    top3 = list(prob.items())[:3]
    cands = string_assigner.get_possible_positions(pitch)
    print(f"  MIDI {pitch}: top={top3}, total={total}, cands={cands}")

# 2. Viterbiの割り当て結果
print("\n=== Viterbi割り当て結果 ===")
notes = [
    {"pitch": 48, "start": 0.0, "duration": 0.5},
    {"pitch": 55, "start": 0.6, "duration": 0.5},
    {"pitch": 60, "start": 1.2, "duration": 0.5},
]
result = string_assigner.assign_strings_dp(notes)
for r in result:
    p = r["pitch"]
    s = r["string"]
    f = r["fret"]
    human_top = list(pref.get(str(p), {}).get("prob", {}).items())[:1]
    print(f"  MIDI {p}: Viterbi=S{s}F{f}, Human top={human_top}")

# 3. 全ノートのstring番号体系を確認（IDMTとGuitarSetで違うかも）
print("\n=== String番号体系チェック ===")
# IDMT: 1=6弦(E2,40), 2=5弦(A2,45), ..., 6=1弦(E4,64)
# GuitarSet: data_source 0=6弦(E2), ..., 5=1弦(E4)
# → DS_TO_STRING = {0:1, 1:2, ..., 5:6}  = IDMT形式
# PyGuitarPro: string 1=1弦(E4), 6=6弦(E2)
# → 変換: idmt_string = 7 - gp_string

# テスト: MIDI 40 は E2（6弦開放）
# IDMT形式: S1F0
# get_possible_positions の出力形式は？
positions_40 = string_assigner.get_possible_positions(40)
print(f"  MIDI 40 (E2): candidates = {positions_40}")
print(f"    → S1F0 は6弦開放? S6F0 は1弦開放?")

# get_possible_positions の弦番号体系を確認
positions_64 = string_assigner.get_possible_positions(64)
print(f"  MIDI 64 (E4): candidates = {positions_64}")
print(f"    → 1弦開放弦はどう表現される?")

# 4. 選好マップのキーと候補の弦番号体系が一致しているか確認
print("\n=== 弦番号体系の一致確認 ===")
for pitch in [40, 64]:
    map_keys = list(pref.get(str(pitch), {}).get("prob", {}).keys())[:3]
    cands = string_assigner.get_possible_positions(pitch)
    print(f"  MIDI {pitch}: map keys={map_keys}, candidates={cands}")
    # map key "1_0" = S1F0, candidate (1,0) = S1F0 → 一致すべき
