"""Debug: why does Viterbi choose str2/f12 instead of str1/f7 for B4?"""
import sys
sys.path.insert(0, r"D:\Music\nextchord-solotab\backend")
from string_assigner import (
    _position_cost, _transition_cost, _timbre_cost,
    get_possible_positions, STANDARD_TUNING, WEIGHTS
)

print("=== WEIGHTS relevant to this issue ===")
for k in ["w_fret_height", "w_high_fret_extra", "w_sweet_spot_bonus", 
           "w_string_switch", "w_same_string_repeat", "w_movement"]:
    print(f"  {k}: {WEIGHTS[k]}")

print("\n=== B4 (MIDI 71) possible positions ===")
positions = get_possible_positions(71, STANDARD_TUNING)
for s, f in positions:
    pos_cost = _position_cost(s, f, pitch=71)
    timbre_cost = _timbre_cost(s, f, STANDARD_TUNING)
    print(f"  str{s}/f{f}: pos_cost={pos_cost:.2f}, timbre_cost={timbre_cost:.2f}, total_static={pos_cost+timbre_cost:.2f}")

print("\n=== B3 (MIDI 59) possible positions ===")
positions_b3 = get_possible_positions(59, STANDARD_TUNING)
for s, f in positions_b3:
    pos_cost = _position_cost(s, f, pitch=59)
    timbre_cost = _timbre_cost(s, f, STANDARD_TUNING)
    print(f"  str{s}/f{f}: pos_cost={pos_cost:.2f}, timbre_cost={timbre_cost:.2f}, total_static={pos_cost+timbre_cost:.2f}")

print("\n=== G3 (MIDI 55) possible positions ===")
positions_g3 = get_possible_positions(55, STANDARD_TUNING)
for s, f in positions_g3:
    pos_cost = _position_cost(s, f, pitch=55)
    timbre_cost = _timbre_cost(s, f, STANDARD_TUNING)
    print(f"  str{s}/f{f}: pos_cost={pos_cost:.2f}, timbre_cost={timbre_cost:.2f}, total_static={pos_cost+timbre_cost:.2f}")

print("\n=== Transition costs for Romance pattern ===")
# Correct: E2(s6/f0) -> B4(s1/f7) -> B3(s2/f0) -> G3(s3/f0)
# Wrong:   E2(s6/f0) -> B4(s2/f12) -> B3(s3/f4) -> G3(s4/f5)

print("\nCorrect path:")
path_correct = [(6,0), (1,7), (2,0), (3,0)]
total = 0
for i, (s,f) in enumerate(path_correct):
    pc = _position_cost(s, f, pitch=[40,71,59,55][i])
    tc = _timbre_cost(s, f, STANDARD_TUNING)
    if i > 0:
        trans = _transition_cost(s, f, path_correct[i-1][0], path_correct[i-1][1], dt=0.23)
    else:
        trans = 0
    total += pc + tc + trans
    print(f"  [{i}] s{s}/f{f}: pos={pc:.2f}, timbre={tc:.2f}, trans={trans:.2f}, running_total={total:.2f}")
print(f"  TOTAL: {total:.2f}")

print("\nWrong path:")
path_wrong = [(6,0), (2,12), (3,4), (4,5)]
total = 0
for i, (s,f) in enumerate(path_wrong):
    pc = _position_cost(s, f, pitch=[40,71,59,55][i])
    tc = _timbre_cost(s, f, STANDARD_TUNING)
    if i > 0:
        trans = _transition_cost(s, f, path_wrong[i-1][0], path_wrong[i-1][1], dt=0.23)
    else:
        trans = 0
    total += pc + tc + trans
    print(f"  [{i}] s{s}/f{f}: pos={pc:.2f}, timbre={tc:.2f}, trans={trans:.2f}, running_total={total:.2f}")
print(f"  TOTAL: {total:.2f}")
