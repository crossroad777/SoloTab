"""
Romance de Amor 正解タブ（PDF読み取り）との完全比較
PDF: D:\Music\romance.pdf (2 pages, dictation: Makoto Sakai)
"""
import json, sys
from collections import Counter, defaultdict
sys.path.insert(0, r"D:\Music\nextchord-solotab\backend")

# ===================================================================
# PDF読み取り結果: 全34小節の正解タブ
# 3/4拍子, 3連符アルペジオ
# フォーマット: 各拍 = (T_fret, A_fret, B_fret) + optional bass
# T=1弦(top), A=2弦, B=3弦(bottom of arpeggio)
# Bass on 4th/5th/6th string
# ===================================================================
# Page 1:
# M1:  T:7,0,0 | 7,0,0 | 7,0,0  B:0(6th)
# M2:  T:7,0,0 | 5,0,0 | 3,0,0  B:0(6th)  
# M3:  T:3,0,0 | 2,0,0 | 0,0,0  B:0(6th)
# M4:  T:0,0,0 | 3,0,0 | 7,0,0  B:0(6th)
# --- Line 2 ---
# M5:  T:12,0,0 | 12,0,0 | 12,0,0  B:0(6th)
# M6:  T:12,0,0 | 10,0,0 | 8,0,0   B:0(6th)
# M7:  T:8,5,5 | 7,5,5 | 5,5,5     B:0(5th)
# M8:  T:5,5,5 | 7,5,5 | 8,5,5     B:0(5th)
# --- Line 3 ---
# M9:  T:7,7,8 | 7,7,8 | 7,7,8     B:7(6th)
# M10: T:11,7,8 | 7,7,8 | 7,7,8    B:7(6th)
# M11: T:7,0,0 | 5,0,0 | 3,0,0     B:0(6th)
# M12: T:3,0,0 | 2,0,0 | 0,0,0     B:0(6th)
# --- Line 4 ---
# M13: T:2,0,2 | 2,0,2 | 2,0,2     B:2(6th)
# M14: T:2,0,2 | 3,0,2 | 2,0,2     B:2(6th)
# M15: T:0,0,0 | 0,0,0 | 0,0,0     B:2(5th),0(6th),3(5th)
# M16: (final chord + repeat)
# --- E major section (key change ####) ---
# M18: T:4,0,1 | 4,0,1 | 4,0,1     B:0(6th)
# M19: T:4,2,0 | 0,0,1 | 0,0,1     B:0(6th)  
# M20: T:5,2,4 | 4,2,4 | 2,2,4     B:2(5th)
# M21: T:4,2,4 | 3,2,4 | 2,2,4     B:2(5th)
# --- etc ---

# 正解ポジションマップ（ピッチ→弦/フレット）
# Em部（小節1-16）: ローポジション中心
CORRECT_EM = {
    40: (6, 0),   # E2
    43: (6, 3),   # G2 - not in this piece typically
    45: (5, 0),   # A2
    47: (5, 2),   # B2
    50: (4, 0),   # D3
    52: (4, 2),   # E3
    55: (3, 0),   # G3
    56: (3, 1),   # G#3
    57: (3, 2),   # A3
    59: (2, 0),   # B3
    60: (2, 1),   # C4 -- PDF clearly shows fret 0 on A line for this
    62: (2, 3),   # D4
    64: (1, 0),   # E4
    66: (1, 2),   # F#4
    67: (1, 3),   # G4
    69: (1, 5),   # A4
    71: (1, 7),   # B4
    72: (1, 8),   # C5
    74: (1, 10),  # D5
    76: (1, 12),  # E5
}

# Emaj部（小節18-32）: ポジションが変わる
# PDFから読み取り:
# M18のパターン: T:4, A:0, B:1 → G#4(s1/f4), E4?(s2/f0?→wait...)
# 実はEmaj部のチューニングは同じだが、キーがE majorなので
# G#3=s3/f1 → but PDF shows A line=0, B line=1
# 
# Re-reading PDF M18 more carefully:
# T line: 4  0  1   4  0  1   4  0  1
# A line: (empty)
# B line: 0 (bass on 6th string)
#
# Wait - the PDF TAB has 3 lines: T, A, B
# T = top string (1st string, high E)
# A = middle strings (2nd, 3rd)
# B = bottom strings (4th, 5th, 6th)
#
# Let me re-read more carefully...
# PDF TAB notation uses standard 6-line tab:
# Line 1 (top) = 1st string (high E)
# Line 2 = 2nd string (B)
# Line 3 = 3rd string (G)
# Line 4 = 4th string (D) - sometimes shown as part of bass
# Line 5 = 5th string (A)
# Line 6 (bottom) = 6th string (low E)
#
# But this PDF shows only 3 TAB lines labeled T, A, B
# This is a condensed format!
# T = treble voice (melody on 1st string)
# A = accompaniment (2nd/3rd string arpeggios)
# B = bass

# Let me re-read M1 from PDF:
# T: 7  0  0  7  0  0  7  0  0
# A:          (numbers below T)
# B: 0
# 
# Actually looking more carefully at the image:
# Line labeled "T": 7 _ 0 _ 7 _ 0 _ 0 ...
# Line labeled "A": _ 0 _ 0 _ 0 _ 0 _ ...  
# Line labeled "B": 0 ...
#
# This is a standard 6-line tab but displayed as T/A/B groups!
# The fret numbers on each of the 6 strings in M1:
# str1(e): 7  -  -  7  -  -  7  -  -
# str2(B): -  0  -  -  0  -  -  0  -
# str3(G): -  -  0  -  -  0  -  -  0
# str4(D): -  -  -  -  -  -  -  -  -
# str5(A): -  -  -  -  -  -  -  -  -
# str6(E): 0  -  -  -  -  -  -  -  -

# This matches! Let me now build the correct note-by-note comparison.

# From the PDF, I'll extract the MIDI pitch -> (string, fret) mapping
# for BOTH sections.

# Em section correct positions (already verified with M1-M4):
EM_CORRECT = {
    40: (6, 0),   # E2
    45: (5, 0),   # A2  
    47: (5, 2),   # B2 -- M13 shows bass B=2 on bottom line
    55: (3, 0),   # G3
    57: (3, 2),   # A3
    59: (2, 0),   # B3
    64: (1, 0),   # E4
    66: (1, 2),   # F#4
    67: (1, 3),   # G4
    69: (1, 5),   # A4
    71: (1, 7),   # B4
    72: (1, 8),   # C5
    74: (1, 10),  # D5
    76: (1, 12),  # E5
}

# M7-M8 shows: T:8,5,5 A:5,5,5 with bass A2(5th/f0)
# str1: 8,5,5 → C5(72), A4(69)...wait
# Actually M7 TAB: 
#   8  5  5 | 7  5  5 | 5  5  5
#   with A-line showing 5s
# Hmm, this means:
# str1: 8, then 7, then 5
# str2: 5, 5, 5
# str3: 5, 5, 5  
# This is Am7 or C pattern at 5th position
# C5(72)=s1/f8, B4(71)=s1/f7, A4(69)=s1/f5
# F4(65)=s2/f6? No... s2/f5 would be E4(64). Actually B+5=64, wrong
# Wait: str2 open = B3(59), so str2/f5 = D#4(63+1=64)? No. 
# B3=59, 59+5=64=E4. So str2/f5 = E4
# str3 open = G3(55), 55+5=60=C4. So str3/f5 = C4
# 
# So M7 is: C5(s1/f8), E4(s2/f5), C4(s3/f5) = Am chord arpeggio!

# For M7-M8, the correct positions:
# E4(64) on s2/f5 (not s1/f0!)  
# C4(60) on s3/f5 (not s2/f1!)

# This is CONTEXTUAL - same pitch gets different string based on position!
# Em section: E4=s1/f0, C4 doesn't appear much
# Am passage (M7-M8): E4=s2/f5, C4=s3/f5

# M9-M10 (B7 area): 
# TAB: 7,7,8 | 7,7,8 | bass=7(6th)
# str1/f7=B4, str2/f7=F#4, str3/f8=D#4(55+8=63)
# So: B4=s1/f7, F#4=s2/f7(!), D#4=s3/f8(!)

# M11-M12 back to Em: normal positions again

# Emaj section (M18+):
# M18: T:4,0,1 A:0,1 B:0(6th)
# str1/f4=G#4(68), str2/f0=B3(59), str3/f1=G#3(56)
# Bass: str6/f0=E2(40)

# M20: T:5,2,4 A:2,4 B:2(5th)
# str1/f5=A4, str2/f2=C#4(61), str3/f4=B3? No: G3+4=59=B3? wait
# str3 open=55(G3), +4=59=B3. Hmm that seems wrong for E major...
# Actually str2/f4=63=D#4/Eb4, str3/f4=59=B3
# No wait, this is E major so the sharps are F#,C#,G#,D#
# M20 bass = A2(s5/f2)? No: s5/f0=A2(45), s5/f2=B2(47). Hmm.
# PDF shows B line = 2, which is 5th string fret 2 = B2
# With T:5=A4(69), A-line middle values...

# OK this is getting complex. Let me focus on what matters:
# Build a CONTEXT-AWARE correct position map based on time ranges.

# KEY INSIGHT from PDF:
# 1. Em section (M1-M16): Standard open position
# 2. B7 passage (M9-M10): Position VII (7th fret area)
#    - F#4 = s2/f7 (not s1/f2!)
#    - D#4 = s3/f8 (not s2/f4!)
# 3. Am passage (M7-M8): Position V (5th fret area)
#    - E4 = s2/f5 (not s1/f0!)
#    - C4 = s3/f5 (not s2/f1!)
# 4. Emaj section (M18+): Different positions entirely
#    - G#4 = s1/f4
#    - G#3 = s3/f1
#    - C#4 = s2/f2

# For proper comparison, I need to map time ranges to positions.
# Let me just do pitch-by-pitch with context awareness.

SESSION = r"D:\Music\nextchord-solotab\uploads\20260510-082310"
with open(f"{SESSION}/notes_assigned.json", "r", encoding="utf-8") as f:
    notes = json.load(f)
with open(f"{SESSION}/beats.json", "r", encoding="utf-8") as f:
    beat_data = json.load(f)
beats = beat_data.get("beats", [])

MIDI_TO_NOTE = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

# Build measures from beats (3/4 time)
measures = []
for bi in range(0, len(beats)-3, 3):
    bar_start = beats[bi]
    bar_end = beats[bi + 3] if bi + 3 < len(beats) else beats[-1] + 1
    bar_notes = [n for n in notes if bar_start <= n["start"] < bar_end]
    measures.append({"start": bar_start, "end": bar_end, "notes": bar_notes, "idx": len(measures)+1})

print(f"Total measures: {len(measures)}, Total notes: {len(notes)}")

# === Measure-by-measure TAB readout ===
print("\n" + "="*70)
print("SoloTab出力 小節ごとの確認")
print("="*70)

for mi, m in enumerate(measures[:34]):
    bar_notes = sorted(m["notes"], key=lambda n: n["start"])
    tab_str = []
    for n in bar_notes:
        s = n.get("string", "?")
        f = n.get("fret", "?")
        nn = MIDI_TO_NOTE[n["pitch"] % 12] + str(n["pitch"] // 12 - 1)
        tab_str.append(f"s{s}/f{f}({nn})")
    print(f"\nM{mi+1} (t={m['start']:.2f}s, {len(bar_notes)} notes):")
    # Group by beat
    for b in range(3):
        bi = mi * 3 + b
        if bi >= len(beats) - 1:
            break
        bt, nbt = beats[bi], beats[bi + 1]
        beat_notes = [n for n in bar_notes if bt <= n["start"] < nbt]
        beat_str = " ".join(f"s{n.get('string','?')}/f{n.get('fret','?')}" for n in sorted(beat_notes, key=lambda x: x["pitch"]))
        pitches = " ".join(MIDI_TO_NOTE[n["pitch"]%12]+str(n["pitch"]//12-1) for n in sorted(beat_notes, key=lambda x: x["pitch"]))
        print(f"  Beat{b+1}: {beat_str:30s} | {pitches}")

# === PDF正解との比較（読み取った値） ===
# PDF M1: str1=7,7,7 str2=0,0,0 str3=0,0,0 str6=0
# PDF M2: str1=7,5,3 str2=0,0,0 str3=0,0,0 str6=0
# PDF M3: str1=3,2,0 str2=0,0,0 str3=0,0,0 str6=0
# PDF M4: str1=0,3,7 str2=0,0,0 str3=0,0,0 str6=0
# PDF M5: str1=12,12,12 str2=0,0,0 str3=0,0,0 str6=0
# PDF M6: str1=12,10,8 str2=0,0,0 str3=0,0,0 str6=0
# PDF M7: str1=8,7,5 str2=5,5,5 str3=5,5,5 str5=0
# PDF M8: str1=5,7,8 str2=5,5,5 str3=5,5,5 str5=0
# PDF M9: str1=7,7,7 str2=7,7,7 str3=8,8,8 str6=7
# PDF M10: str1=11,7,7 str2=7,7,7 str3=8,8,8 str6=7
# PDF M11: str1=7,5,3 str2=0,0,0 str3=0,0,0 str6=0
# PDF M12: str1=3,2,0 str2=0,0,0 str3=0,0,0 str6=0
# PDF M13: str1=2,2,2 str2=0,0,0 str3=2,2,2 str6=2
# PDF M14: str1=2,3,2 str2=0,0,0 str3=2,2,2 str6=2
# PDF M15-16: ending cadence
# PDF M18: str1=4,4,4 str2=0,0,0 str3=1,1,1 str6=0
# PDF M19: str1=4,2,0 str2=0,0,0 str3=1,1,1 str6=0 (approx)

print("\n" + "="*70)
print("PDF正解との不一致サマリー")
print("="*70)

# Build expected per-measure patterns from PDF
# Each entry: list of (string, fret) expected for melody notes
PDF_MEASURES = {
    # Em section - open position
    # M1: B4(s1/f7) x3, B3(s2/f0) x3, G3(s3/f0) x3, E2(s6/f0)
    1: {"melody_str": 1, "melody_frets": [7,7,7], "arp_str2": 0, "arp_str3": 0, "bass": (6,0)},
    2: {"melody_str": 1, "melody_frets": [7,5,3], "arp_str2": 0, "arp_str3": 0, "bass": (6,0)},
    3: {"melody_str": 1, "melody_frets": [3,2,0], "arp_str2": 0, "arp_str3": 0, "bass": (6,0)},
    4: {"melody_str": 1, "melody_frets": [0,3,7], "arp_str2": 0, "arp_str3": 0, "bass": (6,0)},
    5: {"melody_str": 1, "melody_frets": [12,12,12], "arp_str2": 0, "arp_str3": 0, "bass": (6,0)},
    6: {"melody_str": 1, "melody_frets": [12,10,8], "arp_str2": 0, "arp_str3": 0, "bass": (6,0)},
    # M7-M8: Am position (5th fret)
    7: {"melody_str": 1, "melody_frets": [8,7,5], "arp_str2": 5, "arp_str3": 5, "bass": (5,0)},
    8: {"melody_str": 1, "melody_frets": [5,7,8], "arp_str2": 5, "arp_str3": 5, "bass": (5,0)},
    # M9-M10: B7 position (7th fret)
    9: {"melody_str": 1, "melody_frets": [7,7,7], "arp_str2": 7, "arp_str3": 8, "bass": (6,7)},
    10: {"melody_str": 1, "melody_frets": [11,7,7], "arp_str2": 7, "arp_str3": 8, "bass": (6,7)},
    # M11-M12: back to Em open
    11: {"melody_str": 1, "melody_frets": [7,5,3], "arp_str2": 0, "arp_str3": 0, "bass": (6,0)},
    12: {"melody_str": 1, "melody_frets": [3,2,0], "arp_str2": 0, "arp_str3": 0, "bass": (6,0)},
    # M13-M14: Am/B position
    13: {"melody_str": 1, "melody_frets": [2,2,2], "arp_str2": 0, "arp_str3": 2, "bass": (6,2)},
    14: {"melody_str": 1, "melody_frets": [2,3,2], "arp_str2": 0, "arp_str3": 2, "bass": (6,2)},
}

# Compare measure by measure
total_checked = 0
total_correct = 0
issues_by_measure = {}

for mi_key, expected in PDF_MEASURES.items():
    if mi_key - 1 >= len(measures):
        break
    m = measures[mi_key - 1]
    bar_notes = sorted(m["notes"], key=lambda n: n["start"])
    
    measure_issues = []
    measure_checked = 0
    measure_correct = 0
    
    for n in bar_notes:
        s = n.get("string", -1)
        f = n.get("fret", -1)
        pitch = n["pitch"]
        nn = MIDI_TO_NOTE[pitch % 12] + str(pitch // 12 - 1)
        
        # Determine expected position based on role
        # Bass note check
        if pitch < 48:  # bass range
            exp_s, exp_f = expected["bass"]
            measure_checked += 1
            if s == exp_s and f == exp_f:
                measure_correct += 1
            else:
                measure_issues.append(f"{nn}: s{s}/f{f} → 正解 s{exp_s}/f{exp_f}")
        # Arpeggio str2 check (B3 typically)
        elif s == 2 or (pitch == 59 and expected["arp_str2"] == 0):
            exp_f = expected["arp_str2"]
            measure_checked += 1
            if s == 2 and f == exp_f:
                measure_correct += 1
            elif pitch == 59 and expected["arp_str2"] == 0 and s == 2 and f == 0:
                measure_correct += 1
            else:
                measure_issues.append(f"{nn}: s{s}/f{f} → 正解 s2/f{exp_f}")
        # Arpeggio str3 check
        elif s == 3 and pitch < 64:
            exp_f = expected["arp_str3"]
            measure_checked += 1
            if f == exp_f:
                measure_correct += 1
            else:
                measure_issues.append(f"{nn}: s{s}/f{f} → 正解 s3/f{exp_f}")
        # Melody (str1)
        elif s == 1 or pitch >= 64:
            measure_checked += 1
            if s == expected["melody_str"]:
                measure_correct += 1
            else:
                measure_issues.append(f"{nn}: s{s}/f{f} → 正解 s{expected['melody_str']}")
    
    total_checked += measure_checked
    total_correct += measure_correct
    
    if measure_issues:
        issues_by_measure[mi_key] = measure_issues
        print(f"\n  M{mi_key}: {measure_correct}/{measure_checked} 正解")
        for issue in measure_issues[:5]:
            print(f"    ⚠ {issue}")
    else:
        print(f"  M{mi_key}: ✅ {measure_correct}/{measure_checked} 全正解")

print(f"\n{'='*70}")
print(f"Em部(M1-M14) 精度: {total_correct}/{total_checked} = {total_correct/max(total_checked,1)*100:.1f}%")
print(f"不正解小節: {list(issues_by_measure.keys())}")
