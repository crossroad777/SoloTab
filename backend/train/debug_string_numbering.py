"""弦番号体系のアライメント確認"""
import sys, json
sys.path.insert(0, "backend")
import string_assigner

# Viterbi出力の弦番号体系を確認
notes_e2 = [{"pitch": 40, "start": 0.0, "duration": 0.5}]
result_e2 = string_assigner.assign_strings_dp(notes_e2)
s_e2 = result_e2[0]["string"]

notes_e4 = [{"pitch": 64, "start": 0.0, "duration": 0.5}]
result_e4 = string_assigner.assign_strings_dp(notes_e4)
s_e4 = result_e4[0]["string"]

print("=== Viterbi Output Convention ===")
print(f"  MIDI 40 (E2 = 6弦開放): output string = {s_e2}")
print(f"  MIDI 64 (E4 = 1弦開放): output string = {s_e4}")

print("\n=== GuitarSet Ground Truth Convention ===")
print("  DS_TO_STRING = {0:1, 1:2, 2:3, 3:4, 4:5, 5:6}")
print("  data_source 0 = 6弦(E2) -> GT string = 1")
print("  data_source 5 = 1弦(E4) -> GT string = 6")

print("\n=== Alignment Check ===")
print(f"  MIDI 40: Viterbi={s_e2}, GT=1")
print(f"  MIDI 64: Viterbi={s_e4}, GT=6")

if s_e2 != 1:
    print(f"\n*** MISMATCH! ***")
    print(f"  Viterbi uses standard (1=1弦, 6=6弦)")
    print(f"  GT uses IDMT (1=6弦, 6=1弦)")
    print(f"  Fix: gt_string_std = 7 - gt_string_idmt")
    
    # 修正後の正解率を即座に計算
    print(f"\n  Viterbi string {s_e2} == GT converted {7-1} = {s_e2 == 7-1}")
    print(f"  Viterbi string {s_e4} == GT converted {7-6} = {s_e4 == 7-6}")
else:
    print("\n  Conventions match!")
