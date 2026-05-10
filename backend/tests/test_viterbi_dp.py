"""Test: Viterbi DP with Romance data (verify correct string assignment)"""
import json, sys, time
sys.path.insert(0, r"D:\Music\nextchord-solotab\backend")

SESSION = r"D:\Music\nextchord-solotab\uploads\20260510-082310"

# Load original notes (before string assignment)
with open(f"{SESSION}/notes_assigned.json", "r", encoding="utf-8") as f:
    notes_orig = json.load(f)

# Clear existing string assignments to force re-assignment
import copy
notes = copy.deepcopy(notes_orig)
for n in notes:
    if "string" in n:
        del n["string"]
    if "fret" in n:
        del n["fret"]
    if "cnn_string_probs" in n:
        del n["cnn_string_probs"]  # Remove CNN hints for pure DP test

print(f"Loaded {len(notes)} notes, cleared existing assignments")

# Re-assign using Viterbi DP (no CNN)
from string_assigner import assign_strings_dp, STANDARD_TUNING
t0 = time.time()
result = assign_strings_dp(notes, tuning=STANDARD_TUNING)
elapsed = time.time() - t0
print(f"Viterbi DP completed in {elapsed:.1f}s")

# Analyze results
from collections import Counter
fret_dist = Counter(n.get("fret", -1) for n in result)
str_dist = Counter(n.get("string", -1) for n in result)
print(f"\nFret distribution: {sorted(fret_dist.items())}")
print(f"Max fret: {max(fret_dist.keys())}")
print(f"String distribution: {sorted(str_dist.items())}")

# First 20 notes
MIDI_TO_NOTE = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
print(f"\nFirst 20 notes after Viterbi DP (no CNN):")
for i, n in enumerate(result[:20]):
    note_name = MIDI_TO_NOTE[n["pitch"] % 12] + str(n["pitch"] // 12 - 1)
    s = n.get("string", "?")
    f = n.get("fret", "?")
    print(f"  [{i}] t={n['start']:.3f} {note_name}(midi={n['pitch']}) => str{s}/f{f}")

# Compare first few notes with expected Romance pattern
print(f"\n=== Comparison with expected Romance tab ===")
# Expected: E2(s6/f0), B4(s1/f7), B3(s2/f0), G3(s3/f0), B4(s1/f7), ...
expected_first = [
    (40, 6, 0, "E2 bass"),
    (71, 1, 7, "B4 melody"),
    (59, 2, 0, "B3 arp"),
    (55, 3, 0, "G3 arp"),
    (71, 1, 7, "B4 melody"),
    (59, 2, 0, "B3 arp"),
]

for i, (exp_pitch, exp_s, exp_f, desc) in enumerate(expected_first):
    if i >= len(result):
        break
    n = result[i]
    s = n.get("string", -1)
    f = n.get("fret", -1)
    match = "OK" if (s == exp_s and f == exp_f) else f"WRONG (expected s{exp_s}/f{exp_f})"
    print(f"  [{i}] {desc}: s{s}/f{f} {match}")

# Pitch assignment consistency
from collections import defaultdict
pitch_strings = defaultdict(list)
for n in result:
    pitch_strings[n["pitch"]].append((n.get("string", 0), n.get("fret", 0)))

print("\nPitch assignment consistency:")
for pitch in sorted(pitch_strings.keys()):
    assignments = pitch_strings[pitch]
    unique = set(assignments)
    note_name = MIDI_TO_NOTE[pitch % 12] + str(pitch // 12 - 1)
    counts = Counter(assignments)
    dominant = counts.most_common(1)[0]
    consistency = dominant[1] / len(assignments) * 100
    if len(unique) > 1:
        print(f"  {note_name}(midi={pitch}): {len(unique)} positions, dominant=s{dominant[0][0]}/f{dominant[0][1]} ({consistency:.0f}%) | {counts.most_common(3)}")
    else:
        print(f"  {note_name}(midi={pitch}): s{dominant[0][0]}/f{dominant[0][1]} (100%)")
