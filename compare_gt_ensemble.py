from __future__ import annotations
import os
import json
import collections
from typing import Dict, List, Any
from collections import Counter

gt_path = r'd:\Music\nextchord-solotab\backend\ground_truth\romance_forbidden_games.json'
with open(gt_path, 'r', encoding='utf-8') as f:
    gt = json.load(f)

sessions = {
    'v2 (BP)': r'd:\Music\nextchord-solotab\uploads\20260319-230319-yt-064ae0',
    'Ensemble (pre-filter)': r'd:\Music\nextchord-solotab\uploads\20260320-000434-yt-c1de90',
    'Ensemble (post-filter)': r'd:\Music\nextchord-solotab\uploads\20260320-000856-yt-052e93',
}

names_map = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
# Calculate expected pitches for first 6 seconds dynamically from measures_detailed
# BPM=88, so 6s = 6 * (88/60) = 8.8 beats
bpm = gt['metadata']['bpm']
max_beat = 6.0 * (bpm / 60.0)

gt_pitches = Counter()
gt_total = 0
for measure in gt['measures_detailed']:
    measure_base_beat = (measure['measure'] - 1) * 3 # 3/4 time
    for note in measure['notes']:
        note_abs_beat = measure_base_beat + note['beat']
        if note_abs_beat <= max_beat:
            gt_pitches[note['pitch']] += 1
            gt_total += 1

gt_full = 0
for measure in gt['measures_detailed']:
    gt_full += len(measure['notes'])

all_data = {}
for label, path in sessions.items():
    with open(os.path.join(path, 'notes_assigned.json'), 'r', encoding='utf-8') as f:
        notes_list = json.load(f)
    all_data[label] = {
        'notes': notes_list,
        'total': len(notes_list),
    }

print('=' * 80)
print(' Romance - BP vs Ensemble (YouTube GT)')
print('=' * 80)

# 6s comparison
print(f'\n--- 6 seconds (GT={gt_total}) ---')
header = f'{"":>6s} {"GT":>3s}'
for l in sessions: header += f' {l:>22s}'
print(header)
print('-' * 75)

pcs = {}
all_p = set(gt_pitches.keys())
for label, data in all_data.items():
    s = [n for n in data['notes'] if n.get('start',0)<6]
    pc = Counter(n.get('pitch',0) for n in s)
    pcs[label] = (pc, len(s))
    all_p.update(pc.keys())

for p in sorted(all_p):
    nn = names_map[p%12]+str(p//12-1)
    gc = gt_pitches.get(p,0)
    row = f'{nn:>6s} {gc:>3d}'
    for l in sessions:
        row += f' {pcs[l][0].get(p,0):>22d}'
    print(row)

row = f'{"Total":>6s} {gt_total:>3d}'
for l in sessions:
    row += f' {pcs[l][1]:>22d}'
print(row)

# Full piece
print(f'\n--- Full (GT~{gt_full}) ---')
for l,d in all_data.items():
    n=d['total']
    print(f' {l}: {n:>5} (x{float(n)/float(gt_full):.1f})')

# F1 scores
print(f'\n--- Accuracy (6s) ---')
for l in sessions:
    pc = pcs[l][0]; tot = pcs[l][1]
    correct = sum(min(gt_pitches.get(p,0), pc.get(p,0)) for p in all_p)
    if tot>0:
        prec = correct/tot
        rec = correct/gt_total
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
        print(f' {l}: P={prec:.2f} R={rec:.2f} F1={f1:.2f}')
