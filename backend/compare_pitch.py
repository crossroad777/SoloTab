"""
ピッチレベルでの精度比較。
弦・フレット組み合わせではなく、ピッチ(MIDI note)で比較する。
正解TABのフレット番号から標準チューニングでピッチを算出し、
生成GP5のピッチと比較する。
"""
import sys
sys.path.append(r"D:\Music\nextchord-solotab\backend")
import guitarpro as gp

# Standard tuning MIDI: string 1=64(E4), 2=59(B3), 3=55(G3), 4=50(D3), 5=45(A2), 6=40(E2)
TUNING = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}

def to_pitch(string, fret):
    return TUNING[string] + fret

# === Reference data (from romance_page_1.png) ===
# Bar: melody notes (string, fret) in time order, bass (string, fret)
ref = {}
# Row 1: Bars 1-4 (TAB line: 7 0 7 0 7 0 / 7 0 5 0 3 0 / etc)
ref[0] = {"mel": [(1,7),(2,0),(1,7),(2,0),(1,7),(2,0)], "bass": (6,0)}
ref[1] = {"mel": [(1,7),(2,0),(1,5),(2,0),(1,3),(2,0)], "bass": (6,0)}
ref[2] = {"mel": [(1,3),(2,0),(1,2),(2,0),(1,0),(2,0)], "bass": (6,0)}
ref[3] = {"mel": [(1,0),(2,0),(1,3),(2,0),(1,7),(2,0)], "bass": (6,0)}
ref[4] = {"mel": [(1,12),(2,0),(1,12),(2,0),(1,12),(2,0)], "bass": (6,0)}
ref[5] = {"mel": [(1,12),(2,0),(1,10),(2,0),(1,8),(2,0)], "bass": (6,0)}
ref[6] = {"mel": [(1,8),(3,5),(1,7),(3,5),(1,5),(3,5)], "bass": (5,0)}
ref[7] = {"mel": [(1,5),(3,5),(1,7),(3,5),(1,8),(3,5)], "bass": (5,0)}
ref[8] = {"mel": [(1,7),(2,7),(1,8),(2,7),(1,7),(2,8)], "bass": (6,7)}
ref[9] = {"mel": [(1,11),(2,7),(1,8),(2,7),(1,7),(2,8)], "bass": (6,7)}
ref[10] = {"mel": [(1,7),(3,0),(1,5),(3,0),(1,3),(3,0)], "bass": (5,0)}
ref[11] = {"mel": [(1,3),(3,0),(1,2),(3,0),(1,0),(3,0)], "bass": (5,0)}
ref[12] = {"mel": [(1,2),(3,0),(1,2),(3,2),(1,2),(3,2)], "bass": (5,2)}
ref[13] = {"mel": [(1,2),(3,0),(1,3),(3,2),(1,2),(3,2)], "bass": (5,2)}
ref[14] = {"mel": [(2,0),(3,0),(2,0),(3,0),(2,0),(3,0)], "bass": (4,2)}, 

# Convert reference to pitches
ref_pitches = {}
for bar_num, data in ref.items():
    if isinstance(data, tuple):
        data = data[0]  # unwrap
    mel_p = sorted([to_pitch(s, f) for s, f in data["mel"]])
    bass_p = to_pitch(*data["bass"])
    ref_pitches[bar_num] = {"mel": mel_p, "bass": bass_p}

# === Load generated GP5 ===
song = gp.parse(r"D:\Music\nextchord-solotab\uploads\20260512-073742\tab_fixed.gp5")
track = song.tracks[0]

total_pitch_match = 0
total_pitch_count = 0
total_bass_match = 0
total_bass_count = 0

for bar_num in sorted(ref_pitches.keys()):
    rp = ref_pitches[bar_num]
    if bar_num >= len(track.measures):
        continue
    
    m = track.measures[bar_num]
    
    # Get generated melody pitches (Voice 0)
    gen_mel_pitches = []
    for b in m.voices[0].beats:
        for n in b.notes:
            p = TUNING[n.string] + n.value
            gen_mel_pitches.append(p)
    gen_mel_sorted = sorted(gen_mel_pitches)
    
    # Get generated bass pitch (Voice 1)
    gen_bass_p = None
    if len(m.voices) > 1:
        for b in m.voices[1].beats:
            for n in b.notes:
                gen_bass_p = TUNING[n.string] + n.value
                break
            if gen_bass_p:
                break
    
    # Compare melody as multiset (sorted pitch lists)
    ref_mel = rp["mel"]
    
    # Count matching pitches (multiset intersection)
    ref_copy = list(ref_mel)
    matched = 0
    for gp_val in gen_mel_sorted:
        if gp_val in ref_copy:
            matched += 1
            ref_copy.remove(gp_val)
    
    total_count = max(len(ref_mel), len(gen_mel_sorted))
    total_pitch_match += matched
    total_pitch_count += total_count
    
    # Compare bass
    bass_match = (gen_bass_p == rp["bass"]) if gen_bass_p else False
    total_bass_match += int(bass_match)
    total_bass_count += 1
    
    pct = matched / total_count * 100 if total_count > 0 else 0
    ref_names = [str(p) for p in ref_mel]
    gen_names = [str(p) for p in gen_mel_sorted]
    
    print(f"Bar {bar_num:2d}: pitch {matched}/{total_count} ({pct:.0f}%) bass={'OK' if bass_match else 'NG'}")
    if matched < total_count:
        missing = list(ref_copy)
        extra = list(gen_mel_sorted)
        for p in rp["mel"]:
            if p in extra:
                extra.remove(p)
        if missing:
            print(f"  Missing: {missing}")
        if extra:
            print(f"  Extra:   {extra}")

print(f"\n{'='*50}")
print(f"MELODY PITCH ACCURACY: {total_pitch_match}/{total_pitch_count} = {total_pitch_match/total_pitch_count*100:.1f}%")
print(f"BASS ACCURACY: {total_bass_match}/{total_bass_count} = {total_bass_match/total_bass_count*100:.1f}%")
print(f"OVERALL: {(total_pitch_match+total_bass_match)}/{(total_pitch_count+total_bass_count)} = {(total_pitch_match+total_bass_match)/(total_pitch_count+total_bass_count)*100:.1f}%")
