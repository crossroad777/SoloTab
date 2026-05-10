"""Verify tab fret values in latest session"""
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter

sys.path.insert(0, r"D:\Music\nextchord-solotab\backend")

SESSION = r"D:\Music\nextchord-solotab\uploads\20260510-082310"

# 1. notes_assigned.json
print("=== notes_assigned.json ===")
with open(f"{SESSION}/notes_assigned.json", "r", encoding="utf-8") as f:
    notes = json.load(f)
fret_dist = Counter(n.get("fret", -1) for n in notes)
str_dist = Counter(n.get("string", -1) for n in notes)
print(f"Total notes: {len(notes)}")
print(f"Fret distribution: {sorted(fret_dist.items())}")
print(f"Max fret: {max(fret_dist.keys())}")
print(f"String distribution: {sorted(str_dist.items())}")
print(f"High frets (>12): {sum(v for k,v in fret_dist.items() if k > 12)}")

print("\nFirst 20 notes (assigned):")
for i, n in enumerate(notes[:20]):
    print(f"  [{i}] t={n['start']:.3f} pitch={n['pitch']} str={n.get('string','?')} fret={n.get('fret','?')} vel={n.get('velocity',0):.2f}")

# 2. tab.musicxml Staff 2 analysis
print("\n=== tab.musicxml Staff 2 ===")
tree = ET.parse(f"{SESSION}/tab.musicxml")
root = tree.getroot()

s2_frets = []
s2_notes_sample = []
for note in root.iter("note"):
    staff = note.find("staff")
    if staff is None:
        continue
    if staff.text == "2":
        fret_el = note.find(".//fret")
        str_el = note.find(".//string")
        if fret_el is not None and str_el is not None:
            f_val = int(fret_el.text)
            s_val = int(str_el.text)
            s2_frets.append(f_val)
            if len(s2_notes_sample) < 20:
                pitch = note.find("pitch")
                p_str = ""
                if pitch is not None:
                    step = pitch.find("step")
                    octave = pitch.find("octave")
                    alter = pitch.find("alter")
                    step_t = step.text if step is not None else "?"
                    oct_t = octave.text if octave is not None else "?"
                    sharp = "#" if (alter is not None and alter.text == "1") else ""
                    p_str = f"{step_t}{sharp}{oct_t}"
                s2_notes_sample.append(f"str={s_val} fret={f_val} pitch={p_str}")

print(f"Staff 2 total notes: {len(s2_frets)}")
print(f"Staff 2 fret dist: {sorted(Counter(s2_frets).items())}")
print(f"Staff 2 max fret: {max(s2_frets) if s2_frets else 'N/A'}")
print(f"Staff 2 high frets (>12): {sum(1 for f in s2_frets if f > 12)}")

content = open(f"{SESSION}/tab.musicxml", "r", encoding="utf-8").read()
print(f"time-modification count: {content.count('<time-modification>')}")
print(f"tuplet count: {content.count('<tuplet')}")
print(f"staves: {root.find('.//staves').text if root.find('.//staves') is not None else 'none'}")

print("\nFirst 20 Staff 2 notes:")
for i, ns in enumerate(s2_notes_sample):
    print(f"  [{i}] {ns}")

# 3. AlphaTab用変換をテスト
print("\n=== _strip_to_tab_staff result ===")
sys.path.insert(0, r"D:\Music\nextchord-solotab\backend")

# Import the function from main.py directly
import importlib.util
spec = importlib.util.spec_from_file_location("main_module", r"D:\Music\nextchord-solotab\backend\main.py")
# Can't easily import main.py, so just reimplement strip logic inline
STRING_MIDI = {6: 40, 5: 45, 4: 50, 3: 55, 2: 59, 1: 64}
NOTE_NAMES = ["C", "C", "D", "D", "E", "F", "F", "G", "G", "A", "A", "B"]
NOTE_ALTER = [ 0,   1,   0,   1,   0,   0,   1,   0,   1,   0,   1,   0]

# Check what AlphaTab would see after strip
stripped_root = ET.fromstring(content)
# After strip, we verify pitch matches fret/string
mismatches = 0
total_checked = 0
mismatch_samples = []
for note in stripped_root.iter("note"):
    staff = note.find("staff")
    if staff is not None and staff.text == "2":
        fret_el = note.find(".//fret")
        str_el = note.find(".//string")
        pitch_el = note.find("pitch")
        if fret_el is not None and str_el is not None and pitch_el is not None:
            s = int(str_el.text)
            f = int(fret_el.text)
            if s in STRING_MIDI:
                expected_midi = STRING_MIDI[s] + f
                step_el = pitch_el.find("step")
                octave_el = pitch_el.find("octave")
                alter_el = pitch_el.find("alter")
                if step_el is not None and octave_el is not None:
                    # Reconstruct MIDI from pitch
                    step_to_base = {"C":0,"D":2,"E":4,"F":5,"G":7,"A":9,"B":11}
                    actual_midi = step_to_base.get(step_el.text, 0) + (int(octave_el.text)+1)*12
                    if alter_el is not None:
                        actual_midi += int(alter_el.text)
                    total_checked += 1
                    if actual_midi != expected_midi:
                        mismatches += 1
                        if len(mismatch_samples) < 5:
                            mismatch_samples.append(f"str={s} fret={f} expected_midi={expected_midi} actual_midi={actual_midi}")

print(f"Pitch-fret consistency: {total_checked} checked, {mismatches} mismatches")
if mismatch_samples:
    print("Sample mismatches:")
    for m in mismatch_samples:
        print(f"  {m}")

# 4. Romance expected frets
print("\n=== Romance (Romanza) expected pattern ===")
print("The opening of Romance (Forbidden Games) in standard tuning:")
print("  Em arpeggio: str1/f0(E4), str2/f0(B3), str3/f0(G3) or str3/f0, str2/f1(C4), str1/f0(E4)")
print("  Typical fret range: 0-7 for the entire piece")
print()
print("Actual frets in this session:")
for f_val, count in sorted(fret_dist.items()):
    pct = count / len(notes) * 100
    bar = "#" * int(pct / 2)
    print(f"  fret {f_val:2d}: {count:4d} ({pct:5.1f}%) {bar}")
