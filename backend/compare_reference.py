"""
正解画像から読み取ったフレット番号と、生成GP5の実データを小節ごとに比較する。
正解データ: romance_page_1.png / romance_page_2.png から手動転記
"""
import sys
sys.path.append(r"D:\Music\nextchord-solotab\backend")
import guitarpro as gp

# ================================================================
# 正解データ（romance_page_1.png + romance_page_2.png から転記）
# 各小節: [(string, fret), ...] の6音 (melody) + bass
# フォーマット: {"melody": [(str, fret), ...], "bass": [(str, fret)]}
# 弦番号: 1=1弦(E4), 6=6弦(E2) — GP方式
# ================================================================

# Page 1: Em section (measures 1-16, actually bars with repeat)
# 正解 romance_page_1.png から読み取り
# Bar 1: TAB行 = 7 0 7 0 7 0 / Bass: 0 (6弦)
# → s1f7 s2f0 s1f7 s2f0 s1f7 s2f0, bass=s6f0
reference = {}

# Row 1: Bars 1-4
reference[0] = {"melody": [(1,7),(2,0),(1,7),(2,0),(1,7),(2,0)], "bass": [(6,0)]}
reference[1] = {"melody": [(1,7),(2,0),(1,5),(2,0),(1,3),(2,0)], "bass": [(6,0)]}
reference[2] = {"melody": [(1,3),(2,0),(1,2),(2,0),(1,0),(2,0)], "bass": [(6,0)]}
reference[3] = {"melody": [(1,0),(2,0),(1,3),(2,0),(1,7),(2,0)], "bass": [(6,0)]}

# Row 2: Bars 5-8
reference[4] = {"melody": [(1,12),(2,0),(1,12),(2,0),(1,12),(2,0)], "bass": [(6,0)]}
reference[5] = {"melody": [(1,12),(2,0),(1,10),(2,0),(1,8),(2,0)], "bass": [(6,0)]}
reference[6] = {"melody": [(1,8),(3,5),(1,7),(3,5),(1,5),(3,5)], "bass": [(5,0)]}
reference[7] = {"melody": [(1,5),(3,5),(1,7),(3,5),(1,8),(3,5)], "bass": [(5,0)]}

# Row 3: Bars 9-12
reference[8] = {"melody": [(1,7),(2,7),(1,8),(2,7),(1,7),(2,8)], "bass": [(6,7)]}
reference[9] = {"melody": [(1,11),(2,7),(1,8),(2,7),(1,7),(2,8)], "bass": [(6,7)]}
reference[10] = {"melody": [(1,7),(3,0),(1,5),(3,0),(1,3),(3,0)], "bass": [(5,0)]}
reference[11] = {"melody": [(1,3),(3,0),(1,2),(3,0),(1,0),(3,0)], "bass": [(5,0)]}

# Row 4: Bars 13-16 (including repeat bar)
reference[12] = {"melody": [(1,2),(3,0),(1,2),(3,2),(1,2),(3,2)], "bass": [(5,2)]}
reference[13] = {"melody": [(1,2),(3,0),(1,3),(3,2),(1,2),(3,2)], "bass": [(5,2)]}
reference[14] = {"melody": [(2,0),(3,0),(2,0),(3,0),(2,0),(3,0)], "bass": [(4,2)]}  # penultimate
reference[15] = {"melody": [], "bass": []}  # final chord / repeat

# ================================================================
# GP5 データ読み込み
# ================================================================
song = gp.parse(r"D:\Music\nextchord-solotab\uploads\20260512-073742\tab_fixed.gp5")
track = song.tracks[0]

total_correct = 0
total_notes = 0
bar_results = []

for bar_num, ref in sorted(reference.items()):
    if not ref["melody"] and not ref["bass"]:
        continue
    
    if bar_num >= len(track.measures):
        bar_results.append((bar_num, "MISSING", 0, 0))
        continue
    
    m = track.measures[bar_num]
    
    # Extract melody from Voice 0
    gen_melody = []
    for b in m.voices[0].beats:
        for n in b.notes:
            gen_melody.append((n.string, n.value))
    
    # Extract bass from Voice 1
    gen_bass = []
    if len(m.voices) > 1:
        for b in m.voices[1].beats:
            for n in b.notes:
                gen_bass.append((n.string, n.value))
    
    # Compare melody
    ref_mel = ref["melody"]
    correct = 0
    compared = 0
    details = []
    
    for i, (rs, rf) in enumerate(ref_mel):
        if i < len(gen_melody):
            gs, gf = gen_melody[i]
            match = (gs == rs and gf == rf)
            if match:
                correct += 1
            else:
                details.append(f"  n{i}: ref=s{rs}f{rf} gen=s{gs}f{gf} {'OK' if match else 'NG'}")
            compared += 1
        else:
            details.append(f"  n{i}: ref=s{rs}f{rf} gen=MISSING")
            compared += 1
    
    # Extra notes in generated
    for i in range(len(ref_mel), len(gen_melody)):
        gs, gf = gen_melody[i]
        details.append(f"  n{i}: ref=NONE gen=s{gs}f{gf} EXTRA")
        compared += 1
    
    # Compare bass
    bass_ok = False
    if ref["bass"]:
        rb_s, rb_f = ref["bass"][0]
        if gen_bass:
            gb_s, gb_f = gen_bass[0]
            bass_ok = (gb_s == rb_s and gb_f == rb_f)
            if not bass_ok:
                details.append(f"  BASS: ref=s{rb_s}f{rb_f} gen=s{gb_s}f{gb_f} NG")
            else:
                correct += 1
        else:
            details.append(f"  BASS: ref=s{rb_s}f{rb_f} gen=MISSING")
        compared += 1
    
    total_correct += correct
    total_notes += compared
    
    status = "PERFECT" if correct == compared else f"{correct}/{compared}"
    print(f"Bar {bar_num:2d}: {status} (mel={len(gen_melody)}/{len(ref_mel)}, bass={'OK' if bass_ok else 'NG'})")
    if details:
        for d in details:
            print(d)

print(f"\n{'='*50}")
print(f"TOTAL ACCURACY: {total_correct}/{total_notes} = {total_correct/total_notes*100:.1f}%")
print(f"(Bars 0-14 of Em section)")
