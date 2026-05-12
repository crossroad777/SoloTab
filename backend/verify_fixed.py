import sys
sys.path.append(r"D:\Music\nextchord-solotab\backend")
import guitarpro as gp

song = gp.parse(r"D:\Music\nextchord-solotab\uploads\20260512-073742\tab_fixed.gp5")
track = song.tracks[0]

print(f"Title: {song.title}, Tempo: {song.tempo}, Measures: {len(track.measures)}")

bass_ok = 0
bass_missing = 0
bass_wrong_dur = 0

for i, m in enumerate(track.measures):
    v0_notes = sum(len(b.notes) for b in m.voices[0].beats)
    v1_notes = 0
    v1_info = ""
    if len(m.voices) > 1:
        for b in m.voices[1].beats:
            if b.notes:
                v1_notes += len(b.notes)
                dot = "D" if b.duration.isDotted else ""
                dur_names = {1: "W", 2: "H", 4: "Q", 8: "8", 16: "16"}
                dn = dur_names.get(b.duration.value, str(b.duration.value))
                v1_info += f" [{dn}{dot} s{b.notes[0].string}f{b.notes[0].value}]"
    
    status = ""
    if v0_notes > 0:
        if v1_notes > 0:
            bass_ok += 1
            # Check if dotted half (expected for 3/4 full-bar bass)
            for b in m.voices[1].beats:
                if b.notes and b.duration.value == 2 and b.duration.isDotted:
                    status = "OK"
                    break
            if not status:
                status = "PARTIAL"
                bass_wrong_dur += 1
        else:
            bass_missing += 1
            status = "NO_BASS"
    else:
        status = "EMPTY" if v1_notes == 0 else "BASS_ONLY"
    
    print(f"M{i:2d}: V0={v0_notes}n V1={v1_notes}n {status}{v1_info}")

print(f"\nSummary: bass_ok={bass_ok}, bass_missing={bass_missing}, bass_wrong_dur={bass_wrong_dur}")
