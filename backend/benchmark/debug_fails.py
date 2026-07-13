"""A9/A28の詳細調査"""
import sys, io, random
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r"D:\Music\nextchord-solotab\backend")
from gp_renderer import notes_to_gp5
import guitarpro as gp

rng = random.Random(12345)

def make_beats(bpm, duration_sec):
    interval = 60.0 / bpm
    return [i * interval for i in range(int(duration_sec / interval) + 2)]

def debug_gp5(label, notes, beats, bpm, time_sig="4/4"):
    bpb = int(time_sig.split("/")[0])
    bar_total = bpb * 12
    gp5 = notes_to_gp5(notes, beats=beats, bpm=bpm, title=label, time_signature=time_sig)
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
            if total > bar_total + 1:
                sep = " + "
                print(f"  {label} Bar{m_idx+1} V{v_idx+1}: total={total}/{bar_total} [{sep.join(details)}]")

# Reproduce A9: skip first 8 (same rng seed)
for i in range(8):
    bpm = rng.choice([60, 80, 100, 120, 140, 160, 180, 200])
    n_notes = rng.randint(5, 100)
    dur = max(8, n_notes * 0.3)
    for j in range(n_notes):
        rng.uniform(0.1, dur-0.5); rng.uniform(0.1,1.0); rng.randint(40,84); rng.randint(1,6); rng.randint(0,12); rng.uniform(0.3,1.0)

# A9
bpm = rng.choice([60, 80, 100, 120, 140, 160, 180, 200])
n_notes = rng.randint(5, 100)
dur = max(8, n_notes * 0.3)
beats = make_beats(bpm, dur)
notes = []
for j in range(n_notes):
    t = rng.uniform(0.1, dur - 0.5)
    notes.append({"start": t, "end": t + rng.uniform(0.1, 1.0),
                  "pitch": rng.randint(40, 84), "string": rng.randint(1, 6),
                  "fret": rng.randint(0, 12), "velocity": rng.uniform(0.3, 1.0)})
debug_gp5("A9", notes, beats, bpm)

# A28: reproduce
rng2 = random.Random(12345)
for i in range(10):
    rng2.choice([60,80,100,120,140,160,180,200]); rng2.randint(5,100)
    for j in range(200): rng2.uniform(0,1); rng2.uniform(0,1); rng2.randint(0,100); rng2.randint(0,10); rng2.randint(0,20); rng2.uniform(0,1)
for i in range(10):
    for j in range(200): rng2.uniform(0,1); rng2.uniform(0,1); rng2.randint(0,100); rng2.randint(0,10); rng2.randint(0,20); rng2.uniform(0,1)

# Simpler: just test 3/4 with 140bpm random notes
rng3 = random.Random(999)
notes28 = []
for j in range(30):
    t = rng3.uniform(0, 8-0.3)
    notes28.append({"start": t, "end": t+rng3.uniform(0.1, 0.8), "pitch": rng3.randint(40, 80),
                    "string": rng3.randint(1, 6), "fret": rng3.randint(0, 15), "velocity": rng3.uniform(0.3, 1.0)})
debug_gp5("A28_repro", notes28, make_beats(140, 10), 140, "3/4")
