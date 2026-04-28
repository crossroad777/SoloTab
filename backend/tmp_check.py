"""quantizationの検証"""
import json, numpy as np

notes = json.load(open('D:/Music/nextchord-solotab/uploads/20260429-003136/notes_assigned.json'))
beats_data = json.load(open('D:/Music/nextchord-solotab/uploads/20260429-003136/beats.json'))
beats = np.array(beats_data['beats'])

divisions = 12
grid = [0, 3, 4, 6, 8, 9, 12]
snap_counts = {g: 0 for g in grid}

for note in notes[:30]:
    t = note['start']
    idx = int(np.searchsorted(beats, t, side='right')) - 1
    idx = max(0, min(idx, len(beats) - 1))
    
    beat_time = beats[idx]
    next_beat_time = beats[idx + 1] if idx + 1 < len(beats) else beat_time + 0.5
    frac = (t - beat_time) / (next_beat_time - beat_time) if next_beat_time > beat_time else 0.0
    frac = max(0.0, min(frac, 0.99))
    
    raw_sub = frac * divisions
    sub_divs = min(grid, key=lambda x: abs(x - raw_sub))
    snap_counts[sub_divs] = snap_counts.get(sub_divs, 0) + 1
    
    is_triplet = sub_divs in [4, 8]
    print(f"  t={t:.3f} frac={frac:.3f} raw={raw_sub:.1f} -> snap={sub_divs} {'TRIPLET' if is_triplet else 'straight'}")

print(f"\nSnap distribution: {snap_counts}")
