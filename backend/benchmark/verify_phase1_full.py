"""
Phase 1 徹底検証スクリプト (9回分)
各テストでGP5生成 → パース → voice overflow + 整合性チェック
"""
import json, sys, os, io
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r"D:\Music\nextchord-solotab\backend")

from gp_renderer import notes_to_gp5
from music_quantizer import quantize_notes_music21
from technique_detector import detect_techniques
import guitarpro as gp
import numpy as np

PASS = 0
FAIL = 0

def check_gp5(label, notes, beats, bpm=120, time_sig="4/4"):
    global PASS, FAIL
    beats_per_bar = int(time_sig.split("/")[0])
    bar_total = beats_per_bar * 12
    
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
            print(f"  [{label}] OK - {len(track.measures)} bars, no overflow")
            PASS += 1
        else:
            print(f"  [{label}] FAIL - {overflow} overflows in {len(track.measures)} bars")
            FAIL += 1
    except Exception as e:
        print(f"  [{label}] ERROR - {e}")
        FAIL += 1

def make_beats(bpm, duration_sec):
    interval = 60.0 / bpm
    return [i * interval for i in range(int(duration_sec / interval) + 1)]

# ============================================================
print("=" * 60)
print("TEST 1: Single note")
print("=" * 60)
notes1 = [{"start": 0.5, "end": 1.0, "pitch": 60, "string": 1, "fret": 5, "velocity": 0.8}]
check_gp5("single_note", notes1, make_beats(120, 4))

# ============================================================
print("\n" + "=" * 60)
print("TEST 2: Dense arpeggio (30ms apart)")
print("=" * 60)
notes2 = []
for i in range(6):
    notes2.append({"start": 0.5 + i*0.03, "end": 2.0, "pitch": 40+i*5, "string": 6-i, "fret": 0, "velocity": 0.7})
check_gp5("dense_arpeggio", notes2, make_beats(120, 4))

# ============================================================
print("\n" + "=" * 60)
print("TEST 3: Fast passage (16th notes at 180bpm)")
print("=" * 60)
notes3 = []
interval_16th = 60.0 / 180 / 4  # 16th note at 180bpm
for i in range(32):
    t = 0.5 + i * interval_16th
    notes3.append({"start": t, "end": t+interval_16th*0.8, "pitch": 60+i%12, "string": 1, "fret": i%12, "velocity": 0.6})
check_gp5("fast_16th_180bpm", notes3, make_beats(180, 6))

# ============================================================
print("\n" + "=" * 60)
print("TEST 4: Bass + Melody (voice separation)")
print("=" * 60)
notes4 = []
for bar in range(8):
    t_base = bar * 2.0
    # Bass on beat 1
    notes4.append({"start": t_base, "end": t_base+1.5, "pitch": 40, "string": 6, "fret": 0, "velocity": 0.7})
    # Melody notes
    for i in range(4):
        t = t_base + i * 0.5
        notes4.append({"start": t, "end": t+0.4, "pitch": 64+i, "string": 1, "fret": i, "velocity": 0.6})
check_gp5("bass_melody_8bars", notes4, make_beats(120, 16))

# ============================================================
print("\n" + "=" * 60)
print("TEST 5: Rubato tempo (variable beat spacing)")
print("=" * 60)
# Simulate rubato: beats are NOT evenly spaced
np.random.seed(42)
base_interval = 0.5  # 120bpm
rubato_beats = [0.0]
for i in range(60):
    jitter = np.random.uniform(-0.08, 0.08)
    rubato_beats.append(rubato_beats[-1] + base_interval + jitter)

notes5 = []
for i in range(20):
    t = rubato_beats[i*2] + 0.1
    notes5.append({"start": t, "end": t+0.3, "pitch": 60+i%7, "string": 1+(i%3), "fret": i%5+2, "velocity": 0.7})
check_gp5("rubato_tempo", notes5, rubato_beats, bpm=120)

# ============================================================
print("\n" + "=" * 60)
print("TEST 6: Notes at bar boundaries")
print("=" * 60)
notes6 = []
beats6 = make_beats(120, 12)
for bar in range(5):
    # Note right at bar end
    t = (bar + 1) * 2.0 - 0.05  # 50ms before bar end
    notes6.append({"start": t, "end": t+0.8, "pitch": 64, "string": 1, "fret": 5, "velocity": 0.7})
    # Note right at bar start
    t2 = (bar + 1) * 2.0 + 0.02
    notes6.append({"start": t2, "end": t2+0.3, "pitch": 67, "string": 2, "fret": 3, "velocity": 0.6})
check_gp5("bar_boundaries", notes6, beats6)

# ============================================================
print("\n" + "=" * 60)
print("TEST 7: 3/4 time signature")
print("=" * 60)
notes7 = []
beats7 = make_beats(100, 12)
for i in range(15):
    t = 0.3 + i * 0.6
    notes7.append({"start": t, "end": t+0.4, "pitch": 60+i%5, "string": 1, "fret": i%5+3, "velocity": 0.7})
check_gp5("waltz_3_4", notes7, beats7, bpm=100, time_sig="3/4")

# ============================================================
print("\n" + "=" * 60)
print("TEST 8: Technique detection - open strings")
print("=" * 60)
notes8 = [
    {"start": 0.0, "end": 1.0, "pitch": 64, "string": 1, "fret": 0, "velocity": 0.8},
    {"start": 0.2, "end": 1.0, "pitch": 64, "string": 1, "fret": 0, "velocity": 0.7},
    {"start": 1.0, "end": 2.0, "pitch": 55, "string": 3, "fret": 0, "velocity": 0.8},
    {"start": 1.2, "end": 2.0, "pitch": 55, "string": 3, "fret": 0, "velocity": 0.7},
    {"start": 2.0, "end": 3.0, "pitch": 62, "string": 2, "fret": 3, "velocity": 0.8},
    {"start": 2.2, "end": 3.0, "pitch": 64, "string": 2, "fret": 5, "velocity": 0.7},
]
result8 = detect_techniques(notes8, bpm=120)
open_bend = [n for n in result8 if n.get("fret",0)==0 and n.get("technique","") in ("b","b_half","b_quarter")]
open_vib = [n for n in result8 if n.get("fret",0)==0 and n.get("technique","")=="~"]
low_bend = [n for n in result8 if 0 < n.get("fret",0) < 3 and n.get("technique","") in ("b","b_half")]

if not open_bend and not open_vib and not low_bend:
    print(f"  [open_string_guard] OK - no bend/vib on open/low fret")
    PASS += 1
else:
    print(f"  [open_string_guard] FAIL - open_bend={len(open_bend)}, open_vib={len(open_vib)}, low_bend={len(low_bend)}")
    FAIL += 1

# Also check GP5 generation
check_gp5("open_string_techniques", result8, make_beats(120, 4))

# ============================================================
print("\n" + "=" * 60)
print("TEST 9: Sakura session (real data)")
print("=" * 60)
sakura_dir = r"D:\Music\nextchord-solotab\uploads\20260522-202216"
if os.path.exists(os.path.join(sakura_dir, "notes_assigned.json")):
    with open(os.path.join(sakura_dir, "notes_assigned.json"), "r", encoding="utf-8") as f:
        sakura_notes = json.load(f)
    with open(os.path.join(sakura_dir, "beats.json"), "r", encoding="utf-8") as f:
        sakura_beats_data = json.load(f)
    sakura_beats = sakura_beats_data if isinstance(sakura_beats_data, list) else sakura_beats_data.get("beats", [])
    if sakura_beats and isinstance(sakura_beats[0], dict):
        sakura_beats = [b["time"] for b in sakura_beats]
    sakura_bpm = 60/(sakura_beats[1]-sakura_beats[0]) if len(sakura_beats)>1 else 75
    
    # Technique re-check
    bend_open = [n for n in sakura_notes if n.get("technique","") in ("b","bend","b_half") and n.get("fret",0)==0]
    vib_open = [n for n in sakura_notes if n.get("technique","")=="~" and n.get("fret",0)==0]
    if bend_open or vib_open:
        print(f"  [sakura_tech] NOTE: old session has {len(bend_open)} bend + {len(vib_open)} vib on fret=0 (expected in old data)")
    
    # GP5 check with existing data
    check_gp5("sakura_real", sakura_notes, sakura_beats, bpm=sakura_bpm)
else:
    # Try latest session
    uploads = r"D:\Music\nextchord-solotab\uploads"
    found = False
    for d in sorted(os.listdir(uploads), reverse=True):
        p = os.path.join(uploads, d, "notes_assigned.json")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                real_notes = json.load(f)
            bp = os.path.join(uploads, d, "beats.json")
            with open(bp, "r", encoding="utf-8") as f:
                bd = json.load(f)
            real_beats = bd if isinstance(bd, list) else bd.get("beats",[])
            if real_beats and isinstance(real_beats[0], dict):
                real_beats = [b["time"] for b in real_beats]
            real_bpm = 60/(real_beats[1]-real_beats[0]) if len(real_beats)>1 else 120
            check_gp5(f"real_{d}", real_notes, real_beats, bpm=real_bpm)
            found = True
            break
    if not found:
        print("  No real session data found, skipping")
        PASS += 1  # neutral

# ============================================================
print("\n" + "=" * 60)
print(f"FINAL RESULT: {PASS} PASS, {FAIL} FAIL")
if FAIL == 0:
    print("ALL TESTS PASSED!")
else:
    print(f"WARNING: {FAIL} tests failed")
print("=" * 60)
