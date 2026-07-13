import sys, os, json
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:\Music\nextchord-solotab\backend')
import guitarpro as gp

# 最新セッション
uploads = r'D:\Music\nextchord-solotab\uploads'
latest = None
for d in sorted(os.listdir(uploads), reverse=True):
    p = os.path.join(uploads, d, 'tab.gp5')
    if os.path.exists(p):
        latest = os.path.join(uploads, d)
        break

print(f'Session: {os.path.basename(latest)}')
gp5 = os.path.join(latest, 'tab.gp5')
song = gp.parse(gp5)
track = song.tracks[0]

print(f'Tempo: {song.tempo} BPM')
print(f'Measures: {len(track.measures)}')

overflow_v1 = 0
overflow_v2 = 0
total_notes = 0
bar_errors = []

for m_idx, m in enumerate(track.measures):
    ts = m.header.timeSignature
    num = ts.numerator
    den = ts.denominator.value
    bar_total = num * (6 if den == 8 else 12)
    
    for v_idx, v in enumerate(m.voices):
        total_dur = 0
        v_notes = 0
        for beat in v.beats:
            d = {1:48,2:24,4:12,8:6,16:3,32:2,64:1}.get(beat.duration.value, 12)
            if beat.duration.isDotted: d = int(d*1.5)
            if hasattr(beat.duration,'tuplet') and beat.duration.tuplet:
                if beat.duration.tuplet.enters==3: d = int(d*2/3)
            total_dur += d
            v_notes += len(beat.notes)
        total_notes += v_notes
        if total_dur > bar_total + 1:
            if v_idx == 0:
                overflow_v1 += 1
            else:
                overflow_v2 += 1
            bar_errors.append(f'  M{m_idx+1} V{v_idx+1}: {total_dur}/{bar_total}')

print(f'Total GP5 notes: {total_notes}')
print(f'Voice 1 overflow: {overflow_v1}')
print(f'Voice 2 overflow: {overflow_v2}')
if bar_errors:
    for e in bar_errors[:10]:
        print(e)

# notes_assigned check
na = os.path.join(latest, 'notes_assigned.json')
with open(na, 'r', encoding='utf-8') as f:
    notes = json.load(f)
from collections import Counter
tc = Counter(n.get('technique','normal') for n in notes)
bo = len([n for n in notes if n.get('technique','') in ('b','bend','b_half','b_quarter') and n.get('fret',0)==0])
vo = len([n for n in notes if n.get('technique','')=='~' and n.get('fret',0)==0])
bl = len([n for n in notes if n.get('technique','') in ('b','bend','b_half','b_quarter') and 0<n.get('fret',0)<3])
s0 = len([n for n in notes if n.get('string',1)<1])
d0 = len([n for n in notes if n.get('duration_divs',1)<1])
bp = len([n for n in notes if n.get('beat_pos',0)<0])

print(f'')
print(f'Assigned notes: {len(notes)}')
print(f'Techniques: {dict(tc)}')
print(f'')
checks = {
    'Voice1 overflow': overflow_v1,
    'Voice2 overflow': overflow_v2,
    'Bend fret=0': bo,
    'Vib fret=0': vo,
    'Bend fret<3': bl,
    'String=0': s0,
    'Duration<1': d0,
    'BeatPos<0': bp,
}
all_ok = True
for name, val in checks.items():
    status = 'OK' if val == 0 else 'FAIL'
    if val > 0: all_ok = False
    print(f'  {name:20s} {val:3d}  {status}')

print(f'')
if all_ok:
    print('VERDICT: ALL CHECKS PASSED')
else:
    print('VERDICT: ISSUES FOUND')
