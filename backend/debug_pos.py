"""Debug: check beat_pos assignment for first bar."""
import sys, json
import numpy as np
sys.path.insert(0, '.')

session_dir = r'D:\Music\nextchord-solotab\uploads\20260512-073742'
with open(f'{session_dir}/notes_assigned.json', 'r', encoding='utf-8') as f:
    notes = json.load(f)

bpm = 89.1
true_interval = 60.0 / bpm
first_note = min(float(n['start']) for n in notes)
first_beat = first_note - 0.01
num_beats = int(round((max(float(n['start']) for n in notes) - first_beat) / true_interval)) + 2
true_beats = [first_beat + i * true_interval for i in range(num_beats)]

# Call _assign_to_bars directly
from tab_renderer import _assign_to_bars
rhythm_info = {'subdivision': 'triplet', 'triplet_ratio': 0.80}

# Filter notes like gp_renderer does
from gp_renderer import _filter_noise
filtered = _filter_noise(notes, 0.20)

entries = _assign_to_bars(filtered, true_beats, beats_per_bar=3, rhythm_info=rhythm_info)

# Show first 2 bars
print("--- Bar 0 entries ---")
bar0 = [e for e in entries if e['bar'] == 0]
for e in bar0:
    print(f"  t={e['start_time']:.3f} pitch={e['pitch']:3d} str={e['string']} fret={e['fret']:2d} "
          f"beat_pos={e['beat_pos']:2d} dur_divs={e['duration_divs']:2d}")

print("\n--- Bar 1 entries ---")
bar1 = [e for e in entries if e['bar'] == 1]
for e in bar1:
    print(f"  t={e['start_time']:.3f} pitch={e['pitch']:3d} str={e['string']} fret={e['fret']:2d} "
          f"beat_pos={e['beat_pos']:2d} dur_divs={e['duration_divs']:2d}")
