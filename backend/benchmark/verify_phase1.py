"""Phase 1 verification"""
import json, sys, os
sys.stdout.reconfigure(encoding='utf-8')

session_dir = r"D:\Music\nextchord-solotab\uploads\20260522-204525"
sys.path.insert(0, r"D:\Music\nextchord-solotab\backend")

# Find latest session with notes
if not os.path.exists(os.path.join(session_dir, "notes_assigned.json")):
    uploads = r"D:\Music\nextchord-solotab\uploads"
    for d in sorted(os.listdir(uploads), reverse=True):
        p = os.path.join(uploads, d, "notes_assigned.json")
        if os.path.exists(p):
            session_dir = os.path.join(uploads, d)
            break

notes_file = os.path.join(session_dir, "notes_assigned.json")
print(f"Session: {session_dir}")

# 1. Technique check
print("\n=== CHECK 1: Technique Distribution ===")
if os.path.exists(notes_file):
    with open(notes_file, "r", encoding="utf-8") as f:
        notes = json.load(f)
    
    from collections import Counter
    tc = Counter(n.get("technique", "normal") for n in notes)
    print(f"  Total notes: {len(notes)}")
    for k, v in sorted(tc.items(), key=lambda x: -x[1]):
        print(f"    {k:15s} {v:4d}")
    
    bend_open = [n for n in notes if n.get("technique","") in ("b","bend","b_half") and n.get("fret",0)==0]
    vib_open = [n for n in notes if n.get("technique","")=="~" and n.get("fret",0)==0]
    print(f"  Bend on fret=0: {len(bend_open)} {'FAIL' if bend_open else 'OK'}")
    print(f"  Vibrato on fret=0: {len(vib_open)} {'FAIL' if vib_open else 'OK'}")
else:
    notes = None
    print("  notes_assigned.json not found")

# 2. GP5 voice overflow
print("\n=== CHECK 2: GP5 Voice Overflow ===")
if notes:
    try:
        from gp_renderer import notes_to_gp5
        import guitarpro as gp
        import io
        
        beats_file = os.path.join(session_dir, "beats.json")
        with open(beats_file, "r", encoding="utf-8") as f:
            beats_data = json.load(f)
        beats = beats_data if isinstance(beats_data, list) else beats_data.get("beats", [])
        if beats and isinstance(beats[0], dict):
            beats = [b["time"] for b in beats]
        bpm = 60 / (beats[1] - beats[0]) if len(beats) > 1 else 120

        gp5_bytes = notes_to_gp5(notes, beats=beats, bpm=bpm, title="Verify", time_signature="4/4")
        song = gp.parse(io.BytesIO(gp5_bytes))
        track = song.tracks[0]
        
        bar_total = 48  # 4/4 * 12
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
                    if overflow <= 5:
                        print(f"  OVERFLOW: Bar {m_idx+1} Voice {v_idx+1}: {total} > {bar_total}")
        print(f"  Total measures: {len(track.measures)}, Overflows: {overflow} {'FAIL' if overflow else 'OK'}")
    except Exception as e:
        print(f"  Error: {e}")
        import traceback; traceback.print_exc()

# 3. music21
print("\n=== CHECK 3: music21 ===")
try:
    import music21
    print(f"  music21 {music21.VERSION_STR} OK")
except: print("  music21 MISSING")
try:
    from music_quantizer import quantize_notes_music21
    print("  music_quantizer OK")
except Exception as e: print(f"  music_quantizer FAIL: {e}")

# 4. Local beat interval
print("\n=== CHECK 4: Local Beat Interval ===")
try:
    import inspect
    from music_quantizer import quantize_notes_music21
    src = inspect.getsource(quantize_notes_music21)
    print(f"  local_beat_dur: {'OK' if 'local_beat_dur' in src else 'MISSING'}")
except Exception as e: print(f"  Error: {e}")

from tab_renderer import _assign_to_bars
src2 = open(r"D:\Music\nextchord-solotab\backend\tab_renderer.py","r",encoding="utf-8").read()
if "beats_arr[idx + 1]" in src2 or "beats_arr[idx+1]" in src2:
    print("  tab_renderer local beat: OK")
else:
    print("  tab_renderer local beat: MISSING")

print("\n=== DONE ===")
