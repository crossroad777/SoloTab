"""
推論パイプラインの各段階のデータを検証する。
1. notes.json — CRNNアンサンブルの生ピッチ検出出力
2. notes_assigned.json — Viterbi DP による弦割り当て後
3. gp_renderer — GP5生成時の小節割り当て後

Romanceの正解ピッチと比較し、どの段階で誤りが入っているか特定する。
"""
import json, sys
sys.path.append(r"D:\Music\nextchord-solotab\backend")

session_dir = r"D:\Music\nextchord-solotab\uploads\20260512-073742"

# 1. notes.json (raw CRNN output)
with open(f"{session_dir}/notes.json") as f:
    raw_notes = json.load(f)

# 2. notes_assigned.json (after Viterbi DP)
with open(f"{session_dir}/notes_assigned.json") as f:
    assigned_notes = json.load(f)

# 3. beats.json
with open(f"{session_dir}/beats.json") as f:
    bd = json.load(f)

beats = bd["beats"]
bpm = bd["bpm"]

print(f"=== Pipeline Data Summary ===")
print(f"Raw notes (CRNN output): {len(raw_notes)} notes")
print(f"Assigned notes (Viterbi): {len(assigned_notes)} notes")
print(f"Beats: {len(beats)}, BPM: {bpm}, Time sig: {bd.get('time_signature','?')}")
print()

# Check raw notes structure
if raw_notes:
    sample = raw_notes[0]
    print(f"Raw note keys: {list(sample.keys())}")
    print(f"Sample raw note: {sample}")
    print()

# Check assigned notes structure  
if assigned_notes:
    sample = assigned_notes[0]
    print(f"Assigned note keys: {list(sample.keys())}")
    print(f"Sample assigned: {sample}")
    print()

# Romance Bar 0 starts at beat[0]. 3/4 time = 3 beats per bar.
# Bar 0: beats[0] to beats[3]
# Bar 1: beats[3] to beats[6]
# etc.

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
def midi_name(m):
    return f"{NOTE_NAMES[int(m) % 12]}{int(m) // 12 - 1}"

print(f"=== Raw CRNN pitches for first 4 bars ===")
for bar_num in range(8):
    beat_start_idx = bar_num * 3
    beat_end_idx = beat_start_idx + 3
    if beat_start_idx >= len(beats):
        break
    t_start = beats[beat_start_idx]
    t_end = beats[beat_end_idx] if beat_end_idx < len(beats) else t_start + 3 * (60/bpm)
    
    bar_raw = [n for n in raw_notes if float(n["start"]) >= t_start and float(n["start"]) < t_end]
    bar_assigned = [n for n in assigned_notes if float(n["start"]) >= t_start and float(n["start"]) < t_end]
    
    print(f"\nBar {bar_num} (t={t_start:.2f}-{t_end:.2f}s):")
    
    # Raw pitches
    raw_pitches = sorted([int(n["pitch"]) for n in bar_raw])
    raw_names = [f"{midi_name(p)}({p})" for p in raw_pitches]
    print(f"  Raw CRNN:  {' '.join(raw_names)} [{len(bar_raw)} notes]")
    
    # Assigned pitches with string/fret
    assigned_sorted = sorted(bar_assigned, key=lambda x: float(x["start"]))
    assigned_info = []
    for n in assigned_sorted:
        p = int(n["pitch"])
        s = n.get("string", "?")
        f = n.get("fret", "?")
        assigned_info.append(f"s{s}f{f}({midi_name(p)})")
    print(f"  Assigned:  {' '.join(assigned_info)} [{len(bar_assigned)} notes]")

# Reference for comparison
print(f"\n=== Reference pitches (romance_page_1.png) ===")
ref_bars = {
    0: "B4(71) B3(59) B4(71) B3(59) B4(71) B3(59) + Bass E2(40)",
    1: "B4(71) B3(59) A4(69) B3(59) G4(67) B3(59) + Bass E2(40)",
    2: "G4(67) B3(59) F#4(66) B3(59) E4(64) B3(59) + Bass E2(40)",
    3: "E4(64) B3(59) G4(67) B3(59) B4(71) B3(59) + Bass E2(40)",
    4: "B4(76) B3(59) B4(76) B3(59) B4(76) B3(59) + Bass E2(40)",
    5: "B4(76) B3(59) D5(74) B3(59) C5(72) B3(59) + Bass E2(40)",
    6: "C5(72) C4(60) B4(71) C4(60) A4(69) C4(60) + Bass A2(45)",
    7: "A4(69) C4(60) B4(71) C4(60) C5(72) C4(60) + Bass A2(45)",
}
for b, desc in sorted(ref_bars.items()):
    print(f"  Bar {b}: {desc}")
