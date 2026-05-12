"""Regenerate GP5 from existing session data and dump rhythm structure."""
import sys, json
sys.path.append(r'D:\Music\nextchord-solotab\backend')

session_dir = r'D:\Music\nextchord-solotab\uploads\20260512-073742'

# Load data
with open(f'{session_dir}/notes_assigned.json', 'r', encoding='utf-8') as f:
    notes = json.load(f)
with open(f'{session_dir}/beats.json', 'r', encoding='utf-8') as f:
    bd = json.load(f)

from gp_renderer import notes_to_gp5
gp5_bytes = notes_to_gp5(
    notes, beats=bd['beats'], bpm=bd['bpm'], title='Romance Test',
    time_signature=bd.get('time_signature', '3/4'),
    rhythm_info={'subdivision': 'triplet', 'triplet_ratio': 0.27},
    noise_gate=0.20,
)
out_path = f'{session_dir}/tab.gp5'
with open(out_path, 'wb') as f:
    f.write(gp5_bytes)
print(f'GP5 written: {len(gp5_bytes)} bytes')

# Verify structure
import guitarpro as gp
song = gp.parse(out_path)
t = song.tracks[0]
for mi in range(min(8, len(t.measures))):
    m = t.measures[mi]
    for vi, v in enumerate(m.voices):
        if not v.beats: continue
        parts = []
        for b in v.beats:
            dur = b.duration.value
            tp = getattr(b.duration, 'tuplet', None)
            ns = [(n.string, n.value) for n in b.notes]
            parts.append(f'd={dur} t={tp} n={ns}')
        sep = ' | '
        print(f'M{mi+1} V{vi+1}: {sep.join(parts)}')
