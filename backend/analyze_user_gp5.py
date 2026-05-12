"""Analyze the user-provided GP5 file (3).gp5"""
import sys, glob, os
sys.path.insert(0, r'D:\Music\nextchord-solotab\backend')
import guitarpro as gp

# Find (3).gp5
files = [f for f in glob.glob(r'D:\Music\*.gp5') if '(3)' in f]
path = files[0]
print(f"File: {os.path.basename(path)}")
print(f"Size: {os.path.getsize(path)} bytes")

song = gp.parse(path)
print(f"Title: {song.title}")
print(f"Tempo: {song.tempo}")
print(f"Tracks: {len(song.tracks)}")
t = song.tracks[0]
print(f"Track: {t.name}")
print(f"Measures: {len(t.measures)}")
mh = song.measureHeaders[0]
print(f"Time Sig: {mh.timeSignature.numerator}/{mh.timeSignature.denominator.value}")
print()

for mi in range(min(12, len(t.measures))):
    m = t.measures[mi]
    for vi, v in enumerate(m.voices):
        if not v.beats:
            continue
        parts = []
        nc = 0
        rc = 0
        for b in v.beats:
            dur = b.duration.value
            dot = '.d' if b.duration.isDotted else ''
            tp = b.duration.tuplet
            tp_str = '3:2' if tp and tp.enters == 3 else '-'
            if 'rest' in str(b.status).lower():
                rc += 1
                parts.append(f'R(d={dur}{dot})')
            else:
                nc += 1
                ns = [(n.string, n.value) for n in b.notes]
                parts.append(f's{ns}')
        print(f"M{mi+1:2d} V{vi+1} [{nc}n/{rc}r]: {' | '.join(parts)}")
