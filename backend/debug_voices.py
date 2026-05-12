import sys, json
sys.path.append(r"D:\Music\nextchord-solotab\backend")
from tab_renderer import _assign_to_bars, _group_by_time

with open(r"D:\Music\nextchord-solotab\uploads\20260512-073742\notes_assigned.json") as f:
    notes = json.load(f)
with open(r"D:\Music\nextchord-solotab\uploads\20260512-073742\beats.json") as f:
    bd = json.load(f)

entries = _assign_to_bars(notes, bd["beats"], 3, rhythm_info={"subdivision": "triplet"})

# DIVISIONS = 12, 3/4 = bar_total_divs = 36
bar_total_divs = 36

for bar_num in range(4):
    bar_notes = [e for e in entries if e["bar"] == bar_num]
    bass = [e for e in bar_notes if int(e.get("pitch", 60)) <= 52]
    melody = [e for e in bar_notes if int(e.get("pitch", 60)) > 52]
    
    print(f"\n=== BAR {bar_num} (melody={len(melody)}, bass={len(bass)}) ===")
    
    if bass:
        groups = _group_by_time(bass, threshold=0.1)
        print(f"  Bass groups: {len(groups)}")
        for gi, g in enumerate(groups):
            raw_pos = int(float(g[0]["beat_pos"]))
            print(f"    Group {gi}: beat_pos={raw_pos}, pitch={g[0]['pitch']}, fret={g[0].get('fret','?')}, str={g[0].get('string','?')}")
            # With force_legato, dur_divs = gap_to_next (next group or bar end)
            if gi + 1 < len(groups):
                next_pos = int(float(groups[gi+1][0]["beat_pos"]))
                gap = next_pos - raw_pos
            else:
                gap = bar_total_divs - raw_pos
            print(f"      -> force_legato dur_divs would be: {gap}")
    
    if melody:
        groups = _group_by_time(melody, threshold=0.1)
        print(f"  Melody groups: {len(groups)}")
        for gi, g in enumerate(groups):
            raw_pos = int(float(g[0]["beat_pos"]))
            print(f"    Group {gi}: beat_pos={raw_pos}, notes={len(g)}")
