"""Compare actual tab output vs expected Romance (Romanza) tab"""
import json

SESSION = r"D:\Music\nextchord-solotab\uploads\20260510-082310"

with open(f"{SESSION}/notes_assigned.json", "r", encoding="utf-8") as f:
    notes = json.load(f)

# Standard tuning
TUNING = {6: 40, 5: 45, 4: 50, 3: 55, 2: 59, 1: 64}

# Romance opening measures (standard tuning, first position):
# Measure 1 (pickup or empty)
# Measure 2: Em arpeggio 
#   Beat 1: 6th string open (E2=40), then 3-note arpeggio on treble
#   The classic pattern for each beat: bass + 3 treble notes
# 
# Expected correct tab for Romance Em section:
# E |--0--0--0--|--0--0--0--|
# B |--0--0--0--|--0--1--0--|
# G |--0--0--0--|--0--0--0--|
# D |-----------|-----------|
# A |-----------|-----------|
# E |--0-----0--|--0-----0--|
#
# So the treble notes should ALL be on strings 1,2,3 at frets 0-4
# The bass should be on strings 4,5,6

print("=== Analysis: Romance correct vs actual tab ===\n")

# Group notes by approximate time (same beat)
from collections import defaultdict
beat_groups = []
current_group = []
last_time = -999

for n in notes:
    t = n["start"]
    if t - last_time > 0.1:  # new group
        if current_group:
            beat_groups.append(current_group)
        current_group = [n]
    else:
        current_group.append(n)
    last_time = t
if current_group:
    beat_groups.append(current_group)

print(f"Total beat groups: {len(beat_groups)}")
print(f"\nFirst 30 beat groups (what the tab shows):\n")

# For Romance, the CORRECT pattern is:
# Each beat has:
#  - 0 or 1 bass note on string 4-6 (low E, A, or D)
#  - 1 treble note on string 1-3
# The 3 treble notes per beat form the arpeggio

MIDI_TO_NOTE = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

for i, group in enumerate(beat_groups[:30]):
    notes_str = []
    for n in group:
        pitch = n["pitch"]
        s = n.get("string", "?")
        f = n.get("fret", "?")
        note_name = MIDI_TO_NOTE[pitch % 12] + str(pitch // 12 - 1)
        notes_str.append(f"{note_name}(s{s}/f{f})")
    
    # Check if this makes musical sense for Romance
    issues = []
    for n in group:
        s = n.get("string", 0)
        f = n.get("fret", 0)
        pitch = n["pitch"]
        
        # Romance uses open position (frets 0-4) for the Em section
        # and 5-7 for the E major section
        if f > 12:
            issues.append(f"HIGH FRET {f}")
        
        # The melody notes (B4=71, A4=69, G#4=68, etc.) should be on strings 1-2
        # but string_assigner may put them on wrong strings
        if pitch >= 64:  # E4 and above
            # Should be on string 1 or 2
            # str1/f0=E4(64), str1/f3=G4(67), str1/f7=B4(71), str1/f5=A4(69)
            # str2/f0=B3(59), str2/f1=C4(60), str2/f5=E4(64), str2/f8=G4(67)
            # Check if fret seems reasonable
            expected_fret_s1 = pitch - 64  # string 1
            expected_fret_s2 = pitch - 59  # string 2
            if s == 2 and f == expected_fret_s2 and expected_fret_s2 <= 12:
                pass  # OK on string 2
            elif s == 1 and f == expected_fret_s1 and expected_fret_s1 <= 12:
                pass  # OK on string 1
            elif s == 3 and f == (pitch - 55) and (pitch - 55) <= 12:
                pass  # OK on string 3
            elif f > 9:
                issues.append(f"POSSIBLE BAD ASSIGN: pitch {pitch} on s{s}/f{f}")
    
    issue_str = f"  *** {', '.join(issues)}" if issues else ""
    print(f"  [{i:2d}] t={group[0]['start']:.3f}: {', '.join(notes_str)}{issue_str}")

# Summary of potential problems
print("\n=== Problem Summary ===")
problem_count = 0
for n in notes:
    pitch = n["pitch"]
    s = n.get("string", 0)
    f = n.get("fret", 0)
    
    # Check: same pitch has multiple different string assignments
    # For Romance, B4 (71) should consistently be on ONE string
    
# Check pitch-to-string consistency
pitch_strings = defaultdict(list)
for n in notes:
    pitch_strings[n["pitch"]].append((n.get("string", 0), n.get("fret", 0)))

print("\nPitch assignment consistency (should be consistent for same pitch):")
for pitch in sorted(pitch_strings.keys()):
    assignments = pitch_strings[pitch]
    unique = set(assignments)
    if len(unique) > 1:
        from collections import Counter
        counts = Counter(assignments)
        most_common = counts.most_common(3)
        note_name = MIDI_TO_NOTE[pitch % 12] + str(pitch // 12 - 1)
        print(f"  pitch {pitch} ({note_name}): {len(unique)} different positions: {most_common}")
