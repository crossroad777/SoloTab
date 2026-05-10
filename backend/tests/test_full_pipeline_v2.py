"""Full pipeline test v2: re-run string assignment with chords + compare vs PDF reference"""
import json, sys, time, copy
from collections import Counter, defaultdict
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

# Load chords
chords = []
try:
    with open(f"{SESSION}/chords.json", "r", encoding="utf-8") as f:
        chords = json.load(f)
    print(f"Loaded {len(chords)} chord segments")
except:
    print("No chords.json found")

# Load session info
with open(f"{SESSION}/session.json", "r", encoding="utf-8") as f:
    session = json.load(f)

# 1. Run Viterbi DP WITH chords
from string_assigner import assign_strings_dp, STANDARD_TUNING
print("Running Viterbi DP (with chords)...")
t0 = time.time()
notes = assign_strings_dp(notes, tuning=STANDARD_TUNING, chords=chords)
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

# 3. Compare with correct positions
MIDI_TO_NOTE = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
CORRECT_POSITIONS = {
    40: (6, 0), 43: (6, 3), 44: (6, 4), 45: (5, 0), 47: (5, 2),
    52: (4, 2), 54: (4, 4), 55: (3, 0), 56: (3, 1), 57: (3, 2),
    59: (2, 0), 60: (2, 1), 62: (2, 3), 63: (2, 4), 64: (1, 0),
    66: (1, 2), 67: (1, 3), 68: (1, 4), 69: (1, 5), 71: (1, 7),
    72: (1, 8), 73: (1, 9), 74: (1, 10), 75: (1, 11), 76: (1, 12),
}

total = 0
correct = 0
wrong = 0
wrong_details = defaultdict(list)

for i, n in enumerate(notes):
    pitch = n["pitch"]
    actual_s = n.get("string", -1)
    actual_f = n.get("fret", -1)
    
    if pitch in CORRECT_POSITIONS:
        expected_s, expected_f = CORRECT_POSITIONS[pitch]
        total += 1
        if actual_s == expected_s and actual_f == expected_f:
            correct += 1
        else:
            wrong += 1
            note_name = MIDI_TO_NOTE[pitch % 12] + str(pitch // 12 - 1)
            wrong_details[pitch].append({
                "idx": i, "time": n["start"],
                "actual": f"s{actual_s}/f{actual_f}",
                "expected": f"s{expected_s}/f{expected_f}",
            })
    else:
        total += 1
        if 40 <= pitch <= 76:
            TUNING = [40, 45, 50, 55, 59, 64]
            best = None
            for s_idx, op in enumerate(TUNING):
                f = pitch - op
                if 0 <= f <= 12:
                    s_num = 6 - s_idx
                    if best is None or f < best[1]:
                        best = (s_num, f)
            if best and (actual_s != best[0] or actual_f != best[1]):
                wrong += 1
                wrong_details[pitch].append({
                    "idx": i, "time": n["start"],
                    "actual": f"s{actual_s}/f{actual_f}",
                    "expected": f"s{best[0]}/f{best[1]} (推定)",
                })
            else:
                correct += 1
        else:
            correct += 1

accuracy = correct / total * 100 if total > 0 else 0
print(f"\n{'='*70}")
print(f"全体精度: {correct}/{total} = {accuracy:.1f}%")
print(f"正解: {correct}, 不正解: {wrong}")
print(f"{'='*70}")

if wrong_details:
    print(f"\nピッチ別 不正解一覧 ({len(wrong_details)} ピッチ)")
    for pitch in sorted(wrong_details.keys()):
        items = wrong_details[pitch]
        note_name = MIDI_TO_NOTE[pitch % 12] + str(pitch // 12 - 1)
        expected = items[0]["expected"]
        actual_counts = Counter(item["actual"] for item in items)
        total_for_pitch = sum(1 for n in notes if n["pitch"] == pitch)
        print(f"\n  {note_name} (MIDI {pitch}): {len(items)}/{total_for_pitch} 不正解")
        print(f"    正解: {expected}")
        print(f"    不正解パターン: {dict(actual_counts)}")
        for item in items[:3]:
            print(f"      [{item['idx']}] t={item['time']:.3f}s: {item['actual']} (正解: {item['expected']})")
else:
    print("\n全ノート正解！")

# 4. Generate MusicXML
from tab_renderer import notes_to_tab_musicxml
from music_theory import detect_rhythm_pattern, detect_key_signature
import numpy as np

rhythm_info = detect_rhythm_pattern(notes, beats)
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

key_sig = detect_key_signature(notes)
print(f"\nRhythm: {rhythm_info['subdivision']}, Key: {key_sig}")

xml_content, tech_map = notes_to_tab_musicxml(
    notes, beats=beats, bpm=bpm,
    title=session.get("filename", "Romance"),
    tuning=STANDARD_TUNING,
    time_signature=time_sig,
    rhythm_info=rhythm_info,
    key_signature=key_sig,
    noise_gate=0.15,
)

# Save
with open(f"{SESSION}/tab.musicxml", "w", encoding="utf-8") as f:
    f.write(xml_content)
with open(f"{SESSION}/notes_assigned.json", "w", encoding="utf-8") as f:
    json.dump(notes, f, ensure_ascii=False, indent=2)

print("\nMusicXML + notes_assigned.json 保存完了")
print("DONE - ブラウザでCtrl+Shift+Rでリロードして確認")
