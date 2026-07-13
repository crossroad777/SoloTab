import sys, io
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:\Music\nextchord-solotab\backend')
from gp_renderer import notes_to_gp5
import guitarpro as gp

notes = []
beats = [i * 0.6 for i in range(21)]
for i in range(15):
    t = 0.3 + i * 0.6
    notes.append({'start': t, 'end': t+0.4, 'pitch': 60+i%5, 'string': 1, 'fret': i%5+3, 'velocity': 0.7})

gp5 = notes_to_gp5(notes, beats=beats, bpm=100, title='test', time_signature='3/4')
song = gp.parse(io.BytesIO(gp5))
track = song.tracks[0]

for m_idx, m in enumerate(track.measures):
    for v_idx, v in enumerate(m.voices):
        total = 0
        details = []
        for beat in v.beats:
            d = {1:48,2:24,4:12,8:6,16:3,32:2,64:1}.get(beat.duration.value, 12)
            if beat.duration.isDotted: d = int(d*1.5)
            if hasattr(beat.duration,'tuplet') and beat.duration.tuplet:
                if beat.duration.tuplet.enters==3: d = int(d*2/3)
            nm = len(beat.notes)
            details.append(str(d) + ("n" + str(nm) if nm else "r"))
            total += d
        if total > 37:
            sep = " + "
            print(f"Bar {m_idx+1} V{v_idx+1}: total={total} [{sep.join(details)}]")
