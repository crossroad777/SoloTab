"""
sakurasakukoro.tg (TuxGuitar) vs SoloTab tab.gp5 比較
"""
import sys, os, io, json
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r"D:\Music\nextchord-solotab\backend")

# 1. TGファイルからノート情報を抽出
import zipfile
import xml.etree.ElementTree as ET

tg_path = r"D:\sakurasakukoro (2).tg"
with zipfile.ZipFile(tg_path, 'r') as z:
    with z.open('content.xml') as f:
        tree = ET.parse(f)
        root = tree.getroot()

# TGファイル基本情報
headers = root.findall('.//TGMeasureHeader')
print(f"=== TG Reference File ===")
print(f"  Title: {root.findtext('TGSong/name', 'N/A')}")
print(f"  Tempo: {headers[0].findtext('tempo', 'N/A')} BPM")
ts = headers[0].find('timeSignature')
print(f"  Time Sig: {ts.get('numerator')}/{ts.get('denominator')}")
print(f"  Measures: {len(headers)}")

# Track 1のノートを抽出
tracks = root.findall('.//TGTrack')
print(f"  Tracks: {len(tracks)}")

for t_idx, track in enumerate(tracks):
    track_name = track.findtext('name', f'Track {t_idx+1}')
    measures = track.findall('.//TGMeasure')
    total_notes = 0
    techniques = {}
    fret_dist = {}
    
    for m in measures:
        voices = m.findall('.//TGVoice')
        for v in voices:
            beats = v.findall('.//TGBeat')
            for beat in beats:
                notes = beat.findall('.//TGNote')
                for note in notes:
                    total_notes += 1
                    fret = int(note.get('fret', 0))
                    string = int(note.get('string', 0))
                    fret_dist[fret] = fret_dist.get(fret, 0) + 1
                    
                    # Check for effects
                    bend = note.find('.//TGBend')
                    vibrato = note.find('.//TGVibrato')
                    hammer = note.find('.//TGHammer')
                    slide = note.find('.//TGSlide')
                    harmonic = note.find('.//TGHarmonic')
                    
                    if bend is not None:
                        techniques['bend'] = techniques.get('bend', 0) + 1
                    if vibrato is not None:
                        techniques['vibrato'] = techniques.get('vibrato', 0) + 1
                    if hammer is not None:
                        techniques['hammer'] = techniques.get('hammer', 0) + 1
                    if slide is not None:
                        techniques['slide'] = techniques.get('slide', 0) + 1
                    if harmonic is not None:
                        techniques['harmonic'] = techniques.get('harmonic', 0) + 1
    
    print(f"\n  Track '{track_name}': {total_notes} notes")
    print(f"  Fret distribution (top 10):")
    for fret, count in sorted(fret_dist.items(), key=lambda x: -x[1])[:10]:
        print(f"    fret {fret:2d}: {count:3d} notes")
    print(f"  Techniques: {techniques if techniques else 'none detected in TG format'}")

# 2. SoloTab最新セッションと比較
print(f"\n{'='*50}")
print(f"=== SoloTab Latest Session ===")

# 最新セッション
uploads = r"D:\Music\nextchord-solotab\uploads"
latest = None
for d in sorted(os.listdir(uploads), reverse=True):
    if os.path.exists(os.path.join(uploads, d, "notes_assigned.json")):
        latest = os.path.join(uploads, d)
        break

if latest:
    print(f"  Session: {os.path.basename(latest)}")
    
    with open(os.path.join(latest, "notes_assigned.json"), "r", encoding="utf-8") as f:
        st_notes = json.load(f)
    
    with open(os.path.join(latest, "session.json"), "r", encoding="utf-8") as f:
        session = json.load(f)
    
    print(f"  Status: {session.get('status', 'unknown')}")
    print(f"  Total notes: {len(st_notes)}")
    
    # Technique distribution
    from collections import Counter
    tc = Counter(n.get("technique", "normal") for n in st_notes)
    print(f"  Techniques:")
    for k, v in sorted(tc.items(), key=lambda x: -x[1]):
        print(f"    {k:15s} {v:4d}")
    
    # Fret distribution
    fc = Counter(n.get("fret", 0) for n in st_notes)
    print(f"  Fret distribution (top 10):")
    for fret, count in sorted(fc.items(), key=lambda x: -x[1])[:10]:
        print(f"    fret {fret:2d}: {count:3d} notes")
    
    # Bug checks
    bend_open = [n for n in st_notes if n.get("technique","") in ("b","bend","b_half","b_quarter") and n.get("fret",0)==0]
    vib_open = [n for n in st_notes if n.get("technique","")=="~" and n.get("fret",0)==0]
    bend_low = [n for n in st_notes if n.get("technique","") in ("b","bend","b_half","b_quarter") and 0 < n.get("fret",0) < 3]
    
    print(f"\n  === Bug Checks ===")
    print(f"  Bend on fret=0: {len(bend_open)} {'FAIL' if bend_open else 'OK'}")
    print(f"  Vibrato on fret=0: {len(vib_open)} {'FAIL' if vib_open else 'OK'}")
    print(f"  Bend on fret 1-2: {len(bend_low)} {'FAIL' if bend_low else 'OK'}")
    
    # GP5 voice overflow check
    print(f"\n  === GP5 Integrity ===")
    gp5_path = os.path.join(latest, "tab.gp5")
    if os.path.exists(gp5_path):
        import guitarpro as gp
        song = gp.parse(gp5_path)
        track = song.tracks[0]
        bar_total = 48  # 4/4
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
        print(f"  Measures: {len(track.measures)}")
        print(f"  Voice overflows: {overflow} {'FAIL' if overflow else 'OK'}")
        
        # Note count in GP5
        gp_notes = sum(len(beat.notes) for m in track.measures for v in m.voices for beat in v.beats)
        print(f"  GP5 note count: {gp_notes}")
    else:
        print(f"  tab.gp5 not found")
else:
    print("  No session with notes_assigned.json found")
    print("  Please D&D sakurasakukoro.wav into SoloTab first")

print(f"\n{'='*50}")
print("COMPARISON COMPLETE")
