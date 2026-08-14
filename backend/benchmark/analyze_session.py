import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import json
from collections import Counter

session = r"D:\Music\nextchord-solotab\uploads\20260522-202216"

# 1. Techniques
with open(f"{session}/techniques.json", "r", encoding="utf-8") as f:
    tech = json.load(f)

if isinstance(tech, list) and tech:
    if isinstance(tech[0], str):
        c = Counter(tech)
    elif isinstance(tech[0], dict):
        c = Counter(t.get("technique", "normal") for t in tech)
    else:
        c = Counter()
    print("=== Technique Distribution ===")
    for k, v in sorted(c.items(), key=lambda x: -x[1]):
        print(f"  {k:15s} {v:4d}")
    print()

# 2. Notes
with open(f"{session}/notes_assigned.json", "r", encoding="utf-8") as f:
    notes = json.load(f)

print(f"Total notes: {len(notes)}")

# Beats/BPM
with open(f"{session}/beats.json", "r", encoding="utf-8") as f:
    beats_data = json.load(f)
beats = beats_data if isinstance(beats_data, list) else beats_data.get("beats", [])
if beats and isinstance(beats[0], dict):
    beats = [b["time"] for b in beats]
bpm = 60 / (beats[1] - beats[0]) if len(beats) > 1 else 120
print(f"BPM: {bpm:.1f}, Beats: {len(beats)}")

# Non-normal techniques
print("\n=== Non-normal techniques (all) ===")
tech_notes = []
for n in notes:
    t = n.get("technique", "normal")
    if t and t != "normal":
        s = float(n.get("start", 0))
        tech_notes.append(n)
        
print(f"Total non-normal: {len(tech_notes)}")
tc = Counter(n.get("technique") for n in tech_notes)
for k, v in sorted(tc.items(), key=lambda x: -x[1]):
    print(f"  {k:15s} {v:4d}")

# Bar 3 (0-indexed 2) notes
print("\n=== Bar 3 notes ===")
bar_start_beat = 8  # bar 3 in 4/4 = beats 8-11
bar_end_beat = 12
if len(beats) > bar_end_beat:
    t0 = beats[bar_start_beat]
    t1 = beats[bar_end_beat]
    print(f"Time: {t0:.3f}s - {t1:.3f}s")
    print(f"{'start':>7} {'pitch':>5} {'str':>3} {'fret':>4} {'tech':>15} {'dur':>7}")
    print("-" * 50)
    for n in sorted(notes, key=lambda x: float(x.get("start", 0))):
        s = float(n.get("start", 0))
        if t0 <= s < t1:
            e = float(n.get("end", s+0.5))
            t = str(n.get("technique", "normal"))
            print(f"{s:7.3f} {n.get('pitch',0):5} {n.get('string',0):3} {n.get('fret',0):4} {t:>15} {e-s:7.3f}")

# Bend/vibrato specifically  
print("\n=== Bends and Vibratos ===")
for n in sorted(notes, key=lambda x: float(x.get("start", 0))):
    t = n.get("technique", "normal")
    if t in ("b", "bend", "b_half", "bend_half", "vibrato", "~"):
        s = float(n.get("start", 0))
        # which bar?
        bar = 0
        for bi in range(0, len(beats)-1, 4):
            if bi+4 < len(beats) and beats[bi] <= s < beats[bi+4]:
                bar = bi // 4 + 1
                break
        print(f"  Bar {bar:2d} t={s:.3f} pitch={n.get('pitch',0)} str={n.get('string',0)} fret={n.get('fret',0)} tech={t}")
