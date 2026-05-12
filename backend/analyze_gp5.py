"""Analyze a GP5 file's rhythm structure."""
import sys
sys.path.insert(0, '.')
import guitarpro as gp

path = r'D:\Music\禁じられた遊び　(ロマンス ) ギター Tab譜 楽譜　コードネーム付 - アコースティック 名曲 ギター タブ 楽譜ギター タブ譜 (128k).gp5'
song = gp.parse(path)
print(f'Title: {song.title}')
print(f'Tempo: {song.tempo}')
t = song.tracks[0]
print(f'Track: {t.name}, Measures: {len(t.measures)}')
print()

for mi in range(min(16, len(t.measures))):
    m = t.measures[mi]
    mh = song.measureHeaders[mi]
    ts = f'{mh.timeSignature.numerator}/{mh.timeSignature.denominator.value}'
    for vi, v in enumerate(m.voices):
        if not v.beats:
            continue
        parts = []
        for b in v.beats:
            dur_val = b.duration.value
            dotted = b.duration.isDotted
            tuplet = b.duration.tuplet
            status = b.status
            ns = [(n.string, n.value) for n in b.notes]
            tp_str = '3:2' if tuplet and tuplet.enters == 3 else '-'
            dot_str = '.dot' if dotted else ''
            st_str = 'REST ' if 'rest' in str(status).lower() else ''
            parts.append(f'd={dur_val}{dot_str} tp={tp_str} {st_str}n={ns}')
        print(f'M{mi+1} [{ts}] V{vi+1}: ' + ' | '.join(parts))
