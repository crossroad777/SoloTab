"""Full pipeline test: re-run string assignment + MusicXML generation"""
import json, sys, time, copy
sys.path.insert(0, r"D:\Music\nextchord-solotab\backend")

SESSION = r"D:\Music\nextchord-solotab\uploads\20260510-082310"

# Load notes (clear old assignments)
with open(f"{SESSION}/notes_assigned.json", "r", encoding="utf-8") as f:
    notes_orig = json.load(f)
notes = copy.deepcopy(notes_orig)
for n in notes:
    if "string" in n: del n["string"]
    if "fret" in n: del n["fret"]
    if "cnn_string_probs" in n: del n["cnn_string_probs"]

# Load beats
with open(f"{SESSION}/beats.json", "r", encoding="utf-8") as f:
    beat_data = json.load(f)
beats = beat_data.get("beats", [])
bpm = beat_data.get("bpm", 89)
time_sig = beat_data.get("time_signature", "3/4")

# Load session info
with open(f"{SESSION}/session.json", "r", encoding="utf-8") as f:
    session = json.load(f)

# 1. Run Viterbi DP
from string_assigner import assign_strings_dp, STANDARD_TUNING
print("Running Viterbi DP...")
t0 = time.time()
notes = assign_strings_dp(notes, tuning=STANDARD_TUNING)
print(f"Viterbi DP: {time.time()-t0:.1f}s")

# 2. Fret clamp
MAX_FRET = 12
clamp_count = 0
for n in notes:
    if n.get("fret", 0) > MAX_FRET:
        pitch = n.get("pitch", 60)
        best_str, best_fret = None, 99
        for s_idx, op in enumerate(STANDARD_TUNING):
            s_num = 6 - s_idx
            f = pitch - op
            if 0 <= f <= MAX_FRET and (best_str is None or f < best_fret):
                best_str, best_fret = s_num, f
        if best_str is not None:
            n["string"] = best_str
            n["fret"] = best_fret
            clamp_count += 1
print(f"Fret clamp: {clamp_count} notes corrected")

# 3. Check first 6 notes
MIDI_TO_NOTE = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
expected = [(40,6,0), (71,1,7), (59,2,0), (55,3,0), (71,1,7), (59,2,0)]
print("\nFirst 6 notes vs expected:")
all_ok = True
for i, (exp_p, exp_s, exp_f) in enumerate(expected):
    n = notes[i]
    s, f = n.get("string", -1), n.get("fret", -1)
    ok = s == exp_s and f == exp_f
    if not ok: all_ok = False
    print(f"  [{i}] {MIDI_TO_NOTE[n['pitch']%12]}{n['pitch']//12-1} => s{s}/f{f} {'OK' if ok else f'WRONG(expected s{exp_s}/f{exp_f})'}")
print(f"\n{'ALL CORRECT' if all_ok else 'SOME WRONG'}")

# 4. Generate MusicXML
from tab_renderer import notes_to_tab_musicxml
from music_theory import detect_rhythm_pattern, detect_key_signature
import numpy as np

rhythm_info = detect_rhythm_pattern(notes, beats)
# 3/4 arpeggio correction
if time_sig == "3/4" and rhythm_info["subdivision"] == "straight":
    beats_arr = np.array(beats)
    notes_per_beat = []
    for bi in range(min(len(beats)-1, 60)):
        bt, nbt = beats[bi], beats[bi+1]
        count = sum(1 for n in notes if bt <= float(n["start"]) < nbt)
        if count > 0:
            notes_per_beat.append(count)
    if notes_per_beat:
        avg_npb = np.mean(notes_per_beat)
        two_or_three = sum(1 for c in notes_per_beat if c in [2, 3]) / len(notes_per_beat)
        if avg_npb >= 2.0 and two_or_three >= 0.7:
            rhythm_info["subdivision"] = "triplet"
            print(f"3/4 triplet detected: avg={avg_npb:.1f}, 2or3={two_or_three:.0%}")

key_sig = detect_key_signature(notes)
print(f"Rhythm: {rhythm_info['subdivision']}, Key: {key_sig}")

xml_content, tech_map = notes_to_tab_musicxml(
    notes, beats=beats, bpm=bpm,
    title=session.get("filename", "Romance"),
    tuning=STANDARD_TUNING,
    time_signature=time_sig,
    rhythm_info=rhythm_info,
    key_signature=key_sig,
    noise_gate=0.15,
)

# Save to session
with open(f"{SESSION}/tab.musicxml", "w", encoding="utf-8") as f:
    f.write(xml_content)
with open(f"{SESSION}/notes_assigned.json", "w", encoding="utf-8") as f:
    json.dump(notes, f, ensure_ascii=False, indent=2)

# Verify output
import xml.etree.ElementTree as ET
root = ET.fromstring(xml_content)
s2_frets = []
for note in root.iter("note"):
    staff = note.find("staff")
    if staff is not None and staff.text == "2":
        fret_el = note.find(".//fret")
        if fret_el is not None:
            s2_frets.append(int(fret_el.text))
print(f"\nMusicXML Staff 2: {len(s2_frets)} notes, max fret={max(s2_frets) if s2_frets else 'N/A'}")
print(f"time-modification: {xml_content.count('<time-modification>')}")
print(f"tuplet: {xml_content.count('<tuplet')}")
from collections import Counter
print(f"Fret distribution: {sorted(Counter(s2_frets).items())}")
print("\nDONE - Reload browser (Ctrl+Shift+R) to see updated tab")
