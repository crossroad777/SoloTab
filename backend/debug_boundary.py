"""Debug bar boundary assignment to find why some bars get 8 notes instead of 9."""
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
true_beats = [first_beat + i * true_interval for i in range(200)]
true_beats = [b for b in true_beats if b <= max(float(n['start']) for n in notes) + true_interval]

from tab_renderer import _assign_to_bars
from music_theory import detect_rhythm_pattern

rhythm_info = {'subdivision': 'triplet', 'triplet_ratio': 0.67}
from gp_renderer import _filter_noise
filtered = _filter_noise(notes, 0.20)

entries = _assign_to_bars(filtered, true_beats, beats_per_bar=3, rhythm_info=rhythm_info)

# Count notes per bar
from collections import Counter
bar_counts = Counter()
for e in entries:
    bar_counts[e['bar']] += 1

print("--- Notes per bar ---")
for bar in sorted(bar_counts.keys())[:16]:
    bar_entries = [e for e in entries if e['bar'] == bar]
    melody = [e for e in bar_entries if e['pitch'] > 52]
    bass = [e for e in bar_entries if e['pitch'] <= 52]
    
    marker = ""
    if len(melody) < 9:
        marker = f" << MELODY SHORT ({len(melody)})"
    if len(melody) > 9:
        marker = f" << MELODY EXCESS ({len(melody)})"
    
    print(f"  Bar {bar:2d}: total={len(bar_entries):2d} melody={len(melody):2d} bass={len(bass)} {marker}")

# Show detail for bars with <9 melody notes
print("\n--- Detail for short bars ---")
for bar in sorted(bar_counts.keys())[:16]:
    bar_entries = [e for e in entries if e['bar'] == bar]
    melody = sorted([e for e in bar_entries if e['pitch'] > 52], key=lambda e: e['beat_pos'])
    
    if len(melody) < 9:
        print(f"\n  Bar {bar} ({len(melody)} melody notes):")
        for e in melody:
            print(f"    t={e['start_time']:.3f} pitch={e['pitch']:3d} beat_pos={e['beat_pos']:2d}")
        
        # Check neighboring bars
        prev_bar = [e for e in entries if e['bar'] == bar - 1 and e['pitch'] > 52]
        next_bar = [e for e in entries if e['bar'] == bar + 1 and e['pitch'] > 52]
        if prev_bar:
            last_prev = sorted(prev_bar, key=lambda e: e['start_time'])[-1]
            print(f"    prev bar last: t={last_prev['start_time']:.3f} pitch={last_prev['pitch']:3d} beat_pos={last_prev['beat_pos']}")
        if next_bar:
            first_next = sorted(next_bar, key=lambda e: e['start_time'])[0]
            print(f"    next bar first: t={first_next['start_time']:.3f} pitch={first_next['pitch']:3d} beat_pos={first_next['beat_pos']}")
        
        # Show beat boundaries for this bar
        bar_beat_start = bar * 3
        if bar_beat_start + 3 < len(true_beats):
            bt = true_beats[bar_beat_start]
            et = true_beats[bar_beat_start + 3]
            print(f"    bar time range: {bt:.3f} - {et:.3f}s")
