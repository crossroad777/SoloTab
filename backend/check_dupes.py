"""Check duplicate pitch time differences in M7-M10."""
import json
notes = json.load(open(r'D:\Music\nextchord-solotab\uploads\20260512-073742\notes_assigned.json','r',encoding='utf-8'))

bpm = 89.1
interval = 60/bpm
first = min(float(n['start']) for n in notes)

# Check M7-M10 (bars 6-9, 0-indexed)
for bar in range(6, 10):
    bar_start = first + bar * 3 * interval
    bar_end = bar_start + 3 * interval
    bar_notes = sorted(
        [n for n in notes if bar_start <= float(n['start']) < bar_end],
        key=lambda n: float(n['start'])
    )
    print(f"Bar {bar+1} ({bar_start:.2f}-{bar_end:.2f}s):")
    for n in bar_notes:
        t = float(n['start'])
        p = int(n['pitch'])
        s = n.get('string', '?')
        f = n.get('fret', '?')
        print(f"  t={t:.3f} pitch={p:3d} str={s} fret={f}")
    
    # Find same-pitch pairs
    for i in range(len(bar_notes)):
        for j in range(i+1, len(bar_notes)):
            p1 = int(bar_notes[i]['pitch'])
            p2 = int(bar_notes[j]['pitch'])
            if p1 == p2:
                diff = float(bar_notes[j]['start']) - float(bar_notes[i]['start'])
                print(f"  ** DUPLICATE pitch={p1}, onset diff={diff:.3f}s")
