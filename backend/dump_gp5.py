"""
生成GP5のBar 0-7のフレット番号を正確にダンプし、
正解画像のTAB行と直接比較可能にする。
"""
import sys
sys.path.append(r"D:\Music\nextchord-solotab\backend")
import guitarpro as gp

TUNING = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def midi_to_name(midi):
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"

song = gp.parse(r"D:\Music\nextchord-solotab\uploads\20260512-073742\tab_fixed.gp5")
track = song.tracks[0]

print("=== Generated GP5: Note-by-note dump (Bars 0-14) ===")
print("Format: [string]f[fret] (pitch_name)\n")

for bar_num in range(15):
    m = track.measures[bar_num]
    
    # Voice 0 (melody)
    mel_str = ""
    for b in m.voices[0].beats:
        if b.notes:
            for n in b.notes:
                p = TUNING[n.string] + n.value
                mel_str += f"s{n.string}f{n.value}({midi_to_name(p)}) "
        else:
            mel_str += "rest "
    
    # Voice 1 (bass)
    bass_str = ""
    if len(m.voices) > 1:
        for b in m.voices[1].beats:
            if b.notes:
                for n in b.notes:
                    p = TUNING[n.string] + n.value
                    bass_str += f"s{n.string}f{n.value}({midi_to_name(p)})"
    
    print(f"Bar {bar_num:2d} Melody: {mel_str.strip()}")
    print(f"       Bass:   {bass_str if bass_str else 'none'}")
    print()

# Also show what the screenshot TAB shows for first 4 bars
print("=== Reference (romance_page_1.png) TAB readout ===")
print("Bar 0: TAB 7-0-7-0-7-0 Bass 0(6th) = E4+7=B4, B3, B4, B3, B4, B3 / E2")
print("Bar 1: TAB 7-0-5-0-3-0 Bass 0(6th) = B4, B3, A4, B3, G4, B3 / E2")
print("Bar 2: TAB 3-0-2-0-0-0 Bass 0(6th) = G4, B3, F#4, B3, E4, B3 / E2")
print("Bar 3: TAB 0-0-3-0-7-0 Bass 0(6th) = E4, B3, G4, B3, B4, B3 / E2")
