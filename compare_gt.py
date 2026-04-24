import json, os
from collections import Counter

gt_path = r'd:\Music\nextchord-solotab\backend\ground_truth\romance_forbidden_games.json'
with open(gt_path, 'r', encoding='utf-8') as f:
    gt = json.load(f)

sessions = {
    'v1':   r'd:\Music\nextchord-solotab\uploads\20260319-223132-yt-012afc',
    'v2':   r'd:\Music\nextchord-solotab\uploads\20260319-230319-yt-064ae0',
    'v3.2': r'd:\Music\nextchord-solotab\uploads\20260319-232751-yt-ce1a95',
}

names_map = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
gt_expected = gt['expected_first_6s']['expected_pitches']
gt_pitches = Counter()
for k, v in gt_expected.items():
    gt_pitches[int(k.split('_')[1])] = v
gt_total = gt['expected_first_6s']['expected_notes']
gt_full = gt['expected_metrics']['total_notes_estimated']

all_data = {}
for label, path in sessions.items():
    with open(os.path.join(path, 'notes.json'), 'r', encoding='utf-8') as f:
        d = json.load(f)
    with open(os.path.join(path, 'session.json'), 'r', encoding='utf-8') as f:
        s = json.load(f)
    all_data[label] = {
        'notes': d.get('notes', []),
        'raw': d.get('raw_count', '?'),
        'total': s.get('total_notes', '?'),
    }

print('=' * 60)
print(' Romance - Final Comparison (YouTube GT)')
print('=' * 60)

for label, data in all_data.items():
    print(f' {label}: raw={data["raw"]} final={data["total"]}')

# 6s comparison
print(f'\n--- 6 seconds (GT={gt_total}) ---')
header = f'{"":>6s} {"GT":>3s}'
for l in sessions: header += f' {l:>5s}'
print(header)
print('-' * 50)

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
        row += f' {pcs[l][0].get(p,0):>5d}'
    # verdict on latest
    c = pcs['v3.2'][0].get(p, 0)
    if gc > 0:
        if c == 0: v = '*MISS'
        elif c <= gc*1.5: v = 'OK'
        else: v = f'x{c/gc:.0f}'
    else:
        v = 'FALSE' if c > 0 else ''
    row += f'  {v}'
    print(row)

row = f'{"Total":>6s} {gt_total:>3d}'
for l in sessions:
    row += f' {pcs[l][1]:>5d}'
print(row)

# Full piece
print(f'\n--- Full (GT~{gt_full}) ---')
for l,d in all_data.items():
    n=len(d['notes'])
    print(f' {l}: {n:>5d} (x{n/gt_full:.1f})')

# F1 scores
print(f'\n--- F1 Score ---')
for l in sessions:
    pc = pcs[l][0]; tot = pcs[l][1]
    correct = sum(min(gt_pitches.get(p,0), pc.get(p,0)) for p in all_p)
    if tot>0:
        prec = correct/tot
        rec = correct/gt_total
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
        print(f' {l}: P={prec:.2f} R={rec:.2f} F1={f1:.2f}')
