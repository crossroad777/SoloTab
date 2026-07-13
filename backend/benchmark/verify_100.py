"""
Phase 1 ストレステスト — 100パターン自動検証
各テストでGP5生成→パース→voice overflow + 整合性チェック
"""
import json, sys, os, io, random, math
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r"D:\Music\nextchord-solotab\backend")

from gp_renderer import notes_to_gp5
from music_quantizer import quantize_notes_music21
from technique_detector import detect_techniques
import guitarpro as gp
import numpy as np

PASS_COUNT = 0
FAIL_COUNT = 0
FAIL_DETAILS = []

def make_beats(bpm, duration_sec):
    interval = 60.0 / bpm
    return [i * interval for i in range(int(duration_sec / interval) + 2)]

def check_gp5(label, notes, beats, bpm=120, time_sig="4/4"):
    global PASS_COUNT, FAIL_COUNT, FAIL_DETAILS
    bpb = int(time_sig.split("/")[0])
    bt = int(time_sig.split("/")[1])
    if bt == 8:
        bar_total = bpb * 6  # 8th note = 6 divs
    else:
        bar_total = bpb * 12
    
    try:
        gp5_bytes = notes_to_gp5(notes, beats=beats, bpm=bpm, title=label, time_signature=time_sig)
        song = gp.parse(io.BytesIO(gp5_bytes))
        track = song.tracks[0]
        
        overflow = 0
        for m_idx, m in enumerate(track.measures):
            for v_idx, v in enumerate(m.voices):
                total = 0
                for beat in v.beats:
                    d = {1:48,2:24,4:12,8:6,16:3,32:2,64:1}.get(beat.duration.value, 12)
                    if beat.duration.isDotted: d = int(d*1.5)
                    if hasattr(beat.duration,'tuplet') and beat.duration.tuplet:
                        if beat.duration.tuplet.enters==3: d = int(d*2/3)
                    total += d
                if total > bar_total + 1:
                    overflow += 1
        
        if overflow == 0:
            PASS_COUNT += 1
            return True
        else:
            FAIL_COUNT += 1
            FAIL_DETAILS.append(f"{label}: {overflow} overflows in {len(track.measures)} bars")
            return False
    except Exception as e:
        FAIL_COUNT += 1
        FAIL_DETAILS.append(f"{label}: ERROR {e}")
        return False

def check_technique(label, notes, bpm=120):
    global PASS_COUNT, FAIL_COUNT, FAIL_DETAILS
    try:
        result = detect_techniques(notes, bpm=bpm)
        bad = []
        for n in result:
            tech = n.get("technique", "")
            fret = n.get("fret", 0)
            if tech in ("b","bend","b_half","b_quarter","b_1half","b_2") and fret < 3:
                bad.append(f"bend@fret{fret}")
            if tech in ("~","vibrato") and fret == 0:
                bad.append(f"vib@fret{fret}")
        if not bad:
            PASS_COUNT += 1
            return True
        else:
            FAIL_COUNT += 1
            FAIL_DETAILS.append(f"{label}: {bad[:3]}")
            return False
    except Exception as e:
        FAIL_COUNT += 1
        FAIL_DETAILS.append(f"{label}: ERROR {e}")
        return False

rng = random.Random(12345)

# ======================================
# BLOCK A: GP5 Voice Overflow (60 tests)
# ======================================
print("BLOCK A: GP5 Voice Overflow Tests (60 patterns)")
print("-" * 50)

# A1-A10: Random note counts at different BPMs
for i in range(10):
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
    check_gp5(f"A{i+1}_random_{bpm}bpm_{n_notes}n", notes, beats, bpm)

# A11-A20: Extreme densities
for i in range(10):
    bpm = 120
    notes = []
    for j in range(rng.randint(50, 200)):
        t = rng.uniform(0, 4)
        notes.append({"start": t, "end": t+0.05, "pitch": 60+j%12, "string": (j%6)+1, "fret": j%12, "velocity": 0.5})
    check_gp5(f"A{i+11}_dense_{len(notes)}n", notes, make_beats(bpm, 6), bpm)

# A21-A30: Different time signatures
for i, ts in enumerate(["4/4","4/4","4/4","3/4","3/4","3/4","4/4","3/4","4/4","3/4"]):
    bpm = rng.choice([80, 100, 120, 140])
    n_notes = rng.randint(10, 40)
    dur = 8
    notes = []
    for j in range(n_notes):
        t = rng.uniform(0, dur-0.3)
        notes.append({"start": t, "end": t+rng.uniform(0.1, 0.8),
                      "pitch": rng.randint(40, 80), "string": rng.randint(1, 6),
                      "fret": rng.randint(0, 15), "velocity": rng.uniform(0.3, 1.0)})
    check_gp5(f"A{i+21}_ts{ts}_{bpm}bpm", notes, make_beats(bpm, dur+2), bpm, ts)

# A31-A40: Bass + melody separation stress
for i in range(10):
    bpm = rng.choice([80, 120, 160])
    bars = rng.randint(4, 16)
    notes = []
    beat_dur = 60/bpm
    for b in range(bars):
        t_base = b * 4 * beat_dur
        # Bass (string 4-6)
        for k in range(rng.randint(1, 3)):
            t = t_base + k * beat_dur
            notes.append({"start": t, "end": t+beat_dur*2, "pitch": rng.randint(40, 52),
                          "string": rng.randint(4, 6), "fret": rng.randint(0, 5), "velocity": 0.7})
        # Melody (string 1-3)
        for k in range(rng.randint(2, 8)):
            t = t_base + rng.uniform(0, 4*beat_dur-0.1)
            notes.append({"start": t, "end": t+rng.uniform(0.1, 0.5), "pitch": rng.randint(55, 84),
                          "string": rng.randint(1, 3), "fret": rng.randint(0, 15), "velocity": 0.6})
    check_gp5(f"A{i+31}_bass_mel_{bars}bar", notes, make_beats(bpm, bars*4*beat_dur+2), bpm)

# A41-A50: Rubato (variable beat spacing)
for i in range(10):
    bpm = rng.choice([80, 100, 120])
    base = 60/bpm
    beats = [0.0]
    for j in range(80):
        jitter = rng.uniform(-base*0.15, base*0.15)
        beats.append(beats[-1] + base + jitter)
    n_notes = rng.randint(15, 60)
    notes = []
    for j in range(n_notes):
        t = rng.uniform(0.1, beats[-1]-0.5)
        notes.append({"start": t, "end": t+rng.uniform(0.1, 1.0),
                      "pitch": rng.randint(40, 84), "string": rng.randint(1, 6),
                      "fret": rng.randint(0, 15), "velocity": rng.uniform(0.3, 1.0)})
    check_gp5(f"A{i+41}_rubato_{n_notes}n", notes, beats, bpm)

# A51-A60: Edge cases
# Single note per bar
for i in range(5):
    bpm = 120
    notes = [{"start": i*2.0+0.5, "end": i*2.0+1.5, "pitch": 60+i, "string": 1, "fret": i, "velocity": 0.7}]
    check_gp5(f"A{i+51}_single_bar{i}", notes, make_beats(bpm, (i+1)*2+2), bpm)

# Empty bars between notes
notes_sparse = [
    {"start": 0.5, "end": 1.0, "pitch": 60, "string": 1, "fret": 5, "velocity": 0.7},
    {"start": 10.5, "end": 11.0, "pitch": 64, "string": 1, "fret": 9, "velocity": 0.7},
]
check_gp5("A56_sparse_10s_gap", notes_sparse, make_beats(120, 14), 120)

# All notes at same time (big chord)
notes_chord = [{"start": 1.0, "end": 3.0, "pitch": 40+i*5, "string": 6-i, "fret": 0, "velocity": 0.8} for i in range(6)]
check_gp5("A57_6string_chord", notes_chord, make_beats(120, 4), 120)

# Very slow BPM
check_gp5("A58_slow_40bpm", [{"start": 1.0, "end": 5.0, "pitch": 60, "string": 1, "fret": 5, "velocity": 0.7}], make_beats(40, 12), 40)

# Very fast BPM
fast_notes = [{"start": i*0.075, "end": i*0.075+0.05, "pitch": 60+i%12, "string": 1, "fret": i%12, "velocity": 0.5} for i in range(40)]
check_gp5("A59_fast_240bpm", fast_notes, make_beats(240, 4), 240)

# Zero velocity notes
zero_vel = [{"start": i*0.5, "end": i*0.5+0.3, "pitch": 60+i%5, "string": 1, "fret": i%5, "velocity": 0.01} for i in range(10)]
check_gp5("A60_zero_velocity", zero_vel, make_beats(120, 6), 120)

print(f"  Block A: {PASS_COUNT} pass, {FAIL_COUNT} fail")

# ======================================
# BLOCK B: Technique Guard (30 tests)
# ======================================
pa_before = PASS_COUNT
fa_before = FAIL_COUNT
print("\nBLOCK B: Technique Guard Tests (30 patterns)")
print("-" * 50)

# B1-B10: Open string pairs (should NOT get bend/vibrato)
for i in range(10):
    bpm = rng.choice([80, 120, 160])
    s = rng.randint(1, 6)
    open_pitches = [64, 59, 55, 50, 45, 40]
    p = open_pitches[s-1]
    notes = [
        {"start": 0.0, "end": 0.5, "pitch": p, "string": s, "fret": 0, "velocity": 0.7},
        {"start": 0.15, "end": 0.5, "pitch": p, "string": s, "fret": 0, "velocity": 0.6},
    ]
    check_technique(f"B{i+1}_open_str{s}_{bpm}bpm", notes, bpm)

# B11-B20: Low fret pairs (fret 1-2, should NOT get bend)
for i in range(10):
    bpm = rng.choice([80, 120, 160])
    s = rng.randint(1, 6)
    f = rng.randint(1, 2)
    notes = [
        {"start": 0.0, "end": 0.5, "pitch": 40+f, "string": s, "fret": f, "velocity": 0.7},
        {"start": 0.15, "end": 0.5, "pitch": 40+f+1, "string": s, "fret": f, "velocity": 0.6},
    ]
    check_technique(f"B{i+11}_lowfret{f}_str{s}", notes, bpm)

# B21-B30: High fret pairs (fret 3+, bend IS allowed)
for i in range(10):
    bpm = rng.choice([80, 120, 160])
    s = rng.randint(1, 3)
    f = rng.randint(5, 12)
    notes = [
        {"start": 0.0, "end": 0.5, "pitch": 50+f, "string": s, "fret": f, "velocity": 0.7},
        {"start": 0.15, "end": 0.5, "pitch": 50+f, "string": s, "fret": f, "velocity": 0.6},
    ]
    # High fret: we just check it doesn't crash, bend may or may not be detected
    try:
        result = detect_techniques(notes, bpm=bpm)
        PASS_COUNT += 1
    except Exception as e:
        FAIL_COUNT += 1
        FAIL_DETAILS.append(f"B{i+21}_highfret{f}: ERROR {e}")

print(f"  Block B: {PASS_COUNT - pa_before} pass, {FAIL_COUNT - fa_before} fail")

# ======================================
# BLOCK C: music21 quantizer (10 tests)
# ======================================
pc_before = PASS_COUNT
fc_before = FAIL_COUNT
print("\nBLOCK C: music21 Quantizer Tests (10 patterns)")
print("-" * 50)

for i in range(10):
    bpm = rng.choice([60, 80, 100, 120, 140, 160])
    n_notes = rng.randint(5, 30)
    dur = max(4, n_notes * 0.4)
    beats = make_beats(bpm, dur)
    notes = []
    for j in range(n_notes):
        t = rng.uniform(0.1, dur - 0.5)
        notes.append({"start": t, "end": t + rng.uniform(0.1, 1.0),
                      "pitch": rng.randint(40, 84), "string": rng.randint(1, 6),
                      "fret": rng.randint(0, 15), "velocity": rng.uniform(0.3, 1.0)})
    try:
        result = quantize_notes_music21(notes, beats, bpm, time_signature="4/4",
                                         beats_per_bar=4, rhythm_subdivision="straight")
        # Check all entries have required fields
        required = ["bar", "beat_pos", "duration_divs", "pitch", "string", "fret"]
        ok = True
        for e in result:
            for f in required:
                if f not in e:
                    ok = False
                    FAIL_DETAILS.append(f"C{i+1}: missing field '{f}'")
                    break
            if e["duration_divs"] < 1:
                ok = False
                FAIL_DETAILS.append(f"C{i+1}: duration_divs={e['duration_divs']}")
            if e["beat_pos"] < 0:
                ok = False
                FAIL_DETAILS.append(f"C{i+1}: beat_pos={e['beat_pos']}")
        if ok:
            PASS_COUNT += 1
        else:
            FAIL_COUNT += 1
    except Exception as e:
        FAIL_COUNT += 1
        FAIL_DETAILS.append(f"C{i+1}: ERROR {e}")

print(f"  Block C: {PASS_COUNT - pc_before} pass, {FAIL_COUNT - fc_before} fail")

# ======================================
# SUMMARY
# ======================================
print("\n" + "=" * 60)
print(f"TOTAL: {PASS_COUNT} PASS, {FAIL_COUNT} FAIL out of {PASS_COUNT + FAIL_COUNT} tests")
if FAIL_COUNT == 0:
    print("ALL 100 TESTS PASSED!")
else:
    print(f"\nFailed tests:")
    for d in FAIL_DETAILS:
        print(f"  - {d}")
print("=" * 60)
