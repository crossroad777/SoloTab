"""Trace the duration data flow for Romance GP5."""
import sys, json
sys.path.insert(0, '.')

# Find the session directory for the Romance file
import os, glob
uploads_dir = r'D:\Music\nextchord-solotab\uploads'
# Find the most recent session
sessions = sorted(glob.glob(os.path.join(uploads_dir, '*')), key=os.path.getmtime, reverse=True)
print("Recent sessions:")
for s in sessions[:5]:
    notes_file = os.path.join(s, 'notes_assigned.json')
    beats_file = os.path.join(s, 'beats.json')
    if os.path.exists(notes_file) and os.path.exists(beats_file):
        with open(beats_file, 'r', encoding='utf-8') as f:
            bd = json.load(f)
        ts = bd.get('time_signature', '?')
        bpm = bd.get('bpm', '?')
        print(f"  {os.path.basename(s)}: ts={ts}, bpm={bpm}")

print("\n--- Using session 20260512-073742 (Romance) ---")
session_dir = r'D:\Music\nextchord-solotab\uploads\20260512-073742'

with open(f'{session_dir}/notes_assigned.json', 'r', encoding='utf-8') as f:
    notes = json.load(f)
with open(f'{session_dir}/beats.json', 'r', encoding='utf-8') as f:
    bd = json.load(f)

beats = bd['beats']
bpm = bd['bpm']
ts = bd.get('time_signature', '4/4')

print(f"Time sig: {ts}, BPM: {bpm}")
print(f"Total notes: {len(notes)}")
print(f"Total beats: {len(beats)}")

# Show first 20 notes with start/end/duration
print("\n--- CRNN Raw Note Data (first 30) ---")
print(f"{'#':>3} {'start':>8} {'end':>8} {'dur_sec':>8} {'pitch':>5} {'str':>3} {'fret':>4} {'vel':>5}")
for i, n in enumerate(notes[:30]):
    start = float(n.get('start', 0))
    end = float(n.get('end', start + 0.5))
    dur = end - start
    pitch = n.get('pitch', '?')
    s = n.get('string', '?')
    f = n.get('fret', '?')
    v = float(n.get('velocity', 0))
    print(f"{i:3d} {start:8.3f} {end:8.3f} {dur:8.3f} {pitch:5} {s:3} {f:4} {v:5.2f}")

# Now trace through _assign_to_bars
print("\n--- Beat Grid (first 12 beats) ---")
for i in range(min(12, len(beats))):
    bar = i // 3
    beat_in_bar = i % 3
    print(f"  beat[{i}] = {beats[i]:.3f}s  -> bar={bar}, beat={beat_in_bar}")

# Calculate beat interval
if len(beats) >= 2:
    intervals = [beats[i+1] - beats[i] for i in range(min(20, len(beats)-1))]
    avg_interval = sum(intervals) / len(intervals)
    print(f"\nAverage beat interval: {avg_interval:.4f}s")
    print(f"Notes per beat interval: {avg_interval:.4f}s / 3 = {avg_interval/3:.4f}s per triplet-eighth")

# Show how notes map to beats for first 2 bars
import numpy as np
beats_arr = np.array(beats)
print("\n--- Note-to-Beat Mapping (first 2 bars) ---")
bar_end_time = beats[5] if len(beats) > 5 else beats[-1]
bar_notes = [n for n in notes if float(n['start']) < bar_end_time]
for n in bar_notes:
    t = float(n['start'])
    idx = int(np.searchsorted(beats_arr, t, side='right')) - 1
    idx = max(0, min(idx, len(beats_arr) - 1))
    bt = float(beats_arr[idx])
    nbt = float(beats_arr[idx + 1]) if idx + 1 < len(beats_arr) else bt + avg_interval
    frac = (t - bt) / (nbt - bt) if nbt > bt else 0
    bar = idx // 3
    bib = idx % 3
    sub = frac * 12  # raw sub-divisions
    # triplet grid snap
    grid = [0, 4, 8, 12]
    snapped = min(grid, key=lambda x: abs(x - sub))
    print(f"  t={t:.3f} pitch={n['pitch']:3d} -> beat[{idx}]={bt:.3f} frac={frac:.3f} raw_sub={sub:.1f} snapped={snapped} bar={bar} beat={bib}")
