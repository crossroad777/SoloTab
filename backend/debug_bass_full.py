import sys, json
sys.path.append(r"D:\Music\nextchord-solotab\backend")
from tab_renderer import _assign_to_bars, _group_by_time

with open(r"D:\Music\nextchord-solotab\uploads\20260512-073742\notes_assigned.json") as f:
    notes = json.load(f)
with open(r"D:\Music\nextchord-solotab\uploads\20260512-073742\beats.json") as f:
    bd = json.load(f)

entries = _assign_to_bars(notes, bd["beats"], 3, rhythm_info={"subdivision": "triplet"})
total_bars = max(int(e["bar"]) for e in entries) + 1

bass_present = 0
bass_absent = 0
bass_misaligned = 0

for bar_num in range(total_bars):
    bar_notes = [e for e in entries if e["bar"] == bar_num]
    bass = [e for e in bar_notes if int(e.get("pitch", 60)) <= 52]
    melody = [e for e in bar_notes if int(e.get("pitch", 60)) > 52]
    
    if bass:
        bass_present += 1
        positions = [int(float(b["beat_pos"])) for b in bass]
        min_pos = min(positions)
        status = "OK" if min_pos <= 4 else f"MISALIGNED(pos={min_pos})"
        if min_pos > 4:
            bass_misaligned += 1
        pitches = [b["pitch"] for b in bass]
        frets = [b.get("fret", "?") for b in bass]
        strings = [b.get("string", "?") for b in bass]
        print(f"Bar {bar_num:2d}: bass={len(bass)} mel={len(melody)} pos={positions} pitch={pitches} fret={frets} str={strings} {status}")
    else:
        bass_absent += 1
        if melody:
            print(f"Bar {bar_num:2d}: NO BASS    mel={len(melody)}")
        else:
            print(f"Bar {bar_num:2d}: EMPTY")

print(f"\nSummary: total={total_bars}, bass_present={bass_present}, bass_absent={bass_absent}, bass_misaligned={bass_misaligned}")
