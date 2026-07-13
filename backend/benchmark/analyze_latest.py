import sys, json, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:\Music\nextchord-solotab\backend')
from collections import Counter

latest = r'D:\Music\nextchord-solotab\uploads\20260522-212553'
with open(os.path.join(latest, 'notes_assigned.json'), 'r', encoding='utf-8') as f:
    notes = json.load(f)

print(f'=== SoloTab Session (latest completed) ===')
print(f'Total notes: {len(notes)}')
tc = Counter(n.get('technique', 'normal') for n in notes)
print('Techniques:')
for k, v in sorted(tc.items(), key=lambda x: -x[1]):
    print(f'  {k:15s} {v:4d}')

fc = Counter(n.get('fret', 0) for n in notes)
print('Fret dist (top 10):')
for fret, count in sorted(fc.items(), key=lambda x: -x[1])[:10]:
    print(f'  fret {fret:2d}: {count:3d}')

bend_open = [n for n in notes if n.get('technique','') in ('b','bend','b_half','b_quarter') and n.get('fret',0)==0]
vib_open = [n for n in notes if n.get('technique','')=='~' and n.get('fret',0)==0]
bend_low = [n for n in notes if n.get('technique','') in ('b','bend','b_half','b_quarter') and 0 < n.get('fret',0) < 3]

print('')
r1 = 'FAIL' if bend_open else 'OK'
r2 = 'FAIL' if vib_open else 'OK'
r3 = 'FAIL' if bend_low else 'OK'
print(f'Bend fret=0:  {len(bend_open)} {r1}')
print(f'Vib fret=0:   {len(vib_open)} {r2}')
print(f'Bend fret<3:  {len(bend_low)} {r3}')

import guitarpro as gp
song = gp.parse(os.path.join(latest, 'tab.gp5'))
track = song.tracks[0]
overflow = 0
for m in track.measures:
    for v in m.voices:
        total = 0
        for b in v.beats:
            d = {1:48,2:24,4:12,8:6,16:3,32:2,64:1}.get(b.duration.value, 12)
            if b.duration.isDotted: d = int(d*1.5)
            if hasattr(b.duration,'tuplet') and b.duration.tuplet:
                if b.duration.tuplet.enters==3: d = int(d*2/3)
            total += d
        if total > 49: overflow += 1
r4 = 'FAIL' if overflow else 'OK'
print(f'')
print(f'GP5 measures: {len(track.measures)}')
print(f'GP5 overflow: {overflow} {r4}')
gp_notes = sum(len(b.notes) for m in track.measures for v in m.voices for b in v.beats)
print(f'GP5 notes:    {gp_notes}')
