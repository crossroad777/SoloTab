"""Verify: updated thresholds detect Romance as triplet"""
import json
import numpy as np

with open(r"D:\Music\nextchord-solotab\uploads\20260510-070202\notes_assigned.json", "r") as f:
    notes = json.load(f)
with open(r"D:\Music\nextchord-solotab\uploads\20260510-070202\beats.json", "r") as f:
    beat_data = json.load(f)

beats = beat_data.get("beats", [])
time_sig = beat_data.get("time_signature", "3/4")
beats_arr = np.array(beats)
notes_per_beat = []
for bi in range(min(len(beats)-1, 60)):
    bt, nbt = beats[bi], beats[bi+1]
    count = sum(1 for n in notes if bt <= float(n["start"]) < nbt)
    if count > 0:
        notes_per_beat.append(count)

avg_npb = np.mean(notes_per_beat)
three_ratio = sum(1 for c in notes_per_beat if c == 3) / len(notes_per_beat)
two_or_three = sum(1 for c in notes_per_beat if c in [2, 3]) / len(notes_per_beat)
is_triplet = avg_npb >= 2.0 and two_or_three >= 0.7 and time_sig == "3/4"

print(f"avg_npb: {avg_npb:.2f} (threshold: >=2.0)")
print(f"2-or-3 ratio: {two_or_three:.1%} (threshold: >=70%)")
print(f"Triplet: {'YES' if is_triplet else 'NO'}")

if is_triplet:
    import sys; sys.path.insert(0, r"D:\Music\nextchord-solotab\backend")
    TUNING = [40, 45, 50, 55, 59, 64]
    for n in notes:
        if n.get("fret", 0) > 12:
            pitch = n.get("pitch", 60)
            best_str, best_fret = None, 99
            for s_idx, op in enumerate(TUNING):
                s_num = 6 - s_idx
                f = pitch - op
                if 0 <= f <= 12 and (best_str is None or f < best_fret):
                    best_str, best_fret = s_num, f
            if best_str:
                n["string"] = best_str
                n["fret"] = best_fret
    
    from tab_renderer import notes_to_tab_musicxml
    import re
    xml, _ = notes_to_tab_musicxml(
        notes, beats=beats, bpm=89, title="Test",
        time_signature="3/4",
        rhythm_info={"subdivision": "triplet", "triplet_ratio": three_ratio},
        noise_gate=0.15,
    )
    tm = xml.count('<time-modification>')
    tp = xml.count('<tuplet')
    frets = re.findall(r'<fret>(\d+)</fret>', xml)
    print(f"\nMusicXML result:")
    print(f"  time-modification: {tm}")
    print(f"  tuplet: {tp}")
    print(f"  max fret: {max(int(f) for f in frets) if frets else 0}")
    print(f"  total notes: {len(frets)}")
