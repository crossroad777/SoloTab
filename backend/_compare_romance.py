"""
原曲 Romance de Amor (romance.pdf) との小節ごとの照合。

原曲の1弦(melody)フレット番号 (3/4拍子、3連符アルペジオ):
各小節は3拍×3音=9音だが、メロディは1弦の第1音。

=== Em Section (Minor Part) ===
M1:  7  0  0 | 7  0  0 | 7  0  0   melody=B4(7)
M2:  7  0  0 | 5  0  0 | 3  0  0   melody=B4,A4,G4(7→5→3)
M3:  3  0  0 | 2  0  0 | 0  0  0   melody=G4,F#4,E4(3→2→0) ... Am section
M4:  0  0  0 | 3  0  0 | 7  0  0   melody=E4,G4,B4(0→3→7)

=== Em/Am Fret 12 Section ===
M5:  12 0  0 | 12 0  0 | 12 0  0   melody=E5(12)
M6:  12 0  0 | 10 0  0 | 8  0  0   melody=E5,D5,C5(12→10→8)
M7:  8  5  0 | 7  5  0 | 5  5  0   melody=C5,B4,A4(8→7→5) inner=A3(5)
M8:  5  5  0 | 5  5  0 | 7  5  0 | 8  melody(5→5→7→8)

=== Dm/B Section ===
M9:  7  8  0 | 7  8  0 | 7  8  0   
M10: 11 7  0 | 8  7  0 | 7  8  0
M11: 7  0  0 | 5  0  0 | 3  0  0
M12: 3  0  0 | 2  0  0 | 0  0  0

=== Am ending ===
M13-M16: 2-0-2-0 pattern (low position)

=== E Major Section (M17+, repeat with ###) ===
M17 (=M1 of E major): 9  9  9 | 9  9  9   (C#5 = s1f9)
M18: 9  7  5                               (C#5→B4→A4)  
M19: 5  4  2                               (A4→G#4→F#4)
M20: 2  4  5
M21: 9  7  5  (repeat similar)
...

=== E Major Fret 12 Section (M26+) ===  
M26: 12 9  0 | 12 9  0 | 12 9  0
M27: 12 9  0 | 11 9  0 | 10 9  0
M28: 9  5  0 | 9  5  0 | 9  5  0
...
"""

import io, sys, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Load AI transcription
with open('D:/Music/nextchord-solotab/uploads/20260429-092245/notes_assigned.json') as f:
    notes = json.load(f)
    notes = notes if isinstance(notes, list) else notes.get('notes', [])

# Load beat info for measure boundaries
with open('D:/Music/nextchord-solotab/uploads/20260429-092245/beats.json') as f:
    beat_data = json.load(f)
    beats = beat_data['beats']
    bpm = beat_data['bpm']

print(f"Total notes: {len(notes)}, BPM: {bpm}, Beats: {len(beats)}")

# 3/4 time: 3 beats per measure
beats_per_measure = 3
measures = []
for i in range(0, len(beats), beats_per_measure):
    measure_beats = beats[i:i+beats_per_measure]
    if len(measure_beats) >= 2:
        start = measure_beats[0]
        # end = next measure start or last beat + beat interval
        if i + beats_per_measure < len(beats):
            end = beats[i + beats_per_measure]
        else:
            end = measure_beats[-1] + (measure_beats[-1] - measure_beats[0]) / (len(measure_beats) - 1)
        measures.append((start, end))

print(f"Measures: {len(measures)}")

# 原曲の1弦メロディフレット (各拍の最初の1弦ノート)
# M1-M16: Em section, M17-M32: E major section
ORIGINAL_S1_MELODY = {
    1: [7, 7, 7],        # B4 B4 B4
    2: [7, 5, 3],        # B4 A4 G4
    3: [3, 2, 0],        # G4 F#4 E4 (Am)
    4: [0, 3, 7],        # E4 G4 B4
    5: [12, 12, 12],     # E5 E5 E5
    6: [12, 10, 8],      # E5 D5 C5
    7: [8, 7, 5],        # C5 B4 A4
    8: [5, 5, 7],        # A4 A4 B4
    9: [7, 7, 7],        # repeat of M1 pattern (B section)
    10: [11, 8, 7],      # different
    11: [7, 5, 3],       
    12: [3, 2, 0],       
    13: [2, 2, 2],       # Am low
    14: [2, 3, 2],
    15: [0, 0, 0],
    16: [0, 0, 0],
    # E Major section
    17: [9, 9, 9],       # C#5
    18: [9, 7, 5],       
    19: [5, 4, 2],
    20: [2, 4, 5],
    21: [9, 9, 9],       
    22: [9, 7, 8],       # 9, 7/8 pattern
    23: [7, 8, 7],
    24: [9, 9, 11],
    25: [9, 9, 9],
    26: [12, 12, 12],    # E5 again
    27: [12, 11, 10],
    28: [9, 9, 9],
    29: [9, 7, 5],
    30: [4, 4, 4],       # low again
    31: [4, 5, 2],
    32: [0, 0, 0],
}

print(f"\n{'M':>3} {'Beat1':>10} {'AI s1':>20} {'Orig s1':>20} {'Match':>6}")
print("=" * 65)

total_match = 0
total_compare = 0

for m_idx, (m_start, m_end) in enumerate(measures[:34]):
    m_num = m_idx + 1
    # Get notes in this measure
    m_notes = [n for n in notes if m_start <= n['start'] < m_end]
    
    # Get 1st string notes (melody), sorted by time
    s1_notes = sorted([n for n in m_notes if n.get('string') == 1], key=lambda n: n['start'])
    
    # Get frets for beat positions (first note in each beat)
    beat_frets = []
    for b in range(3):
        beat_start = m_start + (m_end - m_start) * b / 3
        beat_end = m_start + (m_end - m_start) * (b + 1) / 3
        beat_s1 = [n for n in s1_notes if beat_start <= n['start'] < beat_end]
        if beat_s1:
            beat_frets.append(beat_s1[0]['fret'])
        else:
            beat_frets.append('-')
    
    orig = ORIGINAL_S1_MELODY.get(m_num, None)
    
    if orig:
        match_count = 0
        for ai_f, orig_f in zip(beat_frets, orig):
            if ai_f == orig_f:
                match_count += 1
                total_match += 1
            total_compare += 1
        status = f"{match_count}/3"
    else:
        status = "n/a"
    
    ai_str = str(beat_frets)
    orig_str = str(orig) if orig else "?"
    print(f"{m_num:>3} {m_start:>8.2f}s  {ai_str:>20} {orig_str:>20} {status:>6}")

if total_compare > 0:
    print(f"\n1弦メロディ一致率: {total_match}/{total_compare} = {total_match/total_compare*100:.1f}%")
