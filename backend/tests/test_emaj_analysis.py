"""E majorセクションの不正解パターンを詳細分析"""
import json, sys
from collections import Counter, defaultdict
sys.path.insert(0, r"D:\Music\nextchord-solotab\backend")

SESSION = r"D:\Music\nextchord-solotab\uploads\20260510-082310"

with open(f"{SESSION}/notes_assigned.json", "r", encoding="utf-8") as f:
    notes = json.load(f)
with open(f"{SESSION}/chords.json", "r", encoding="utf-8") as f:
    chords = json.load(f)
with open(f"{SESSION}/beats.json", "r", encoding="utf-8") as f:
    beat_data = json.load(f)
beats = beat_data.get("beats", [])

MIDI_TO_NOTE = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

# Em部の正解ポジション（ローポジション）
CORRECT_EM = {
    40: (6, 0), 43: (6, 3), 44: (6, 4), 45: (5, 0), 47: (5, 2),
    52: (4, 2), 54: (4, 4), 55: (3, 0), 56: (3, 1), 57: (3, 2),
    59: (2, 0), 60: (2, 1), 62: (2, 3), 63: (2, 4), 64: (1, 0),
    66: (1, 2), 67: (1, 3), 68: (1, 4), 69: (1, 5), 71: (1, 7),
    72: (1, 8), 73: (1, 9), 74: (1, 10), 75: (1, 11), 76: (1, 12),
}

# --- コードセクション別の分析 ---
print("=" * 70)
print("コードセクション別の弦割り当て分析")
print("=" * 70)

for ci, chord in enumerate(chords):
    c_start = chord.get('start', chord.get('time', 0))
    c_end = chord.get('end', c_start + chord.get('duration', 999))
    c_name = chord.get('chord', '?')
    
    section_notes = [n for n in notes if c_start <= n.get('start', 0) < c_end]
    if not section_notes:
        continue
    
    # 不正解を数える
    wrong_in_section = []
    for n in section_notes:
        pitch = n["pitch"]
        if pitch in CORRECT_EM:
            exp_s, exp_f = CORRECT_EM[pitch]
            act_s, act_f = n.get("string", -1), n.get("fret", -1)
            if act_s != exp_s or act_f != exp_f:
                nn = MIDI_TO_NOTE[pitch % 12] + str(pitch // 12 - 1)
                wrong_in_section.append(f"{nn}:s{act_s}/f{act_f}→s{exp_s}/f{exp_f}")
    
    if wrong_in_section:
        print(f"\n[{ci}] {c_name} (t={c_start:.1f}-{c_end:.1f}s) "
              f"notes={len(section_notes)}, wrong={len(wrong_in_section)}")
        # 不正解パターンを集計
        patterns = Counter(wrong_in_section)
        for pat, cnt in patterns.most_common(5):
            print(f"    {pat} x{cnt}")

# --- E majorセクション全体の特徴分析 ---
print(f"\n{'=' * 70}")
print("E majorセクション(後半)の弦/フレット分布")
print("=" * 70)

# PDF p2: E majorセクションは小節18から始まる（key change ####）
# chords.jsonからE major関連コードを特定
emaj_chords = ["E", "Emaj", "E:maj", "A", "Amaj", "B", "B7", "C#m", "F#m", "G#m"]
emaj_notes = []
em_notes = []

for n in notes:
    t = n.get('start', 0)
    # コードセクション判定
    current_chord = None
    for chord in chords:
        cs = chord.get('start', chord.get('time', 0))
        ce = chord.get('end', cs + chord.get('duration', 999))
        if cs <= t < ce:
            current_chord = chord.get('chord', '')
            break
    
    if current_chord:
        # E major系のコードかどうか
        # 前半Em部: Em, Am, B, B7, C, D, G
        # 後半E部: E, A, B, B7(E major key)
        # 簡易判定: 時間的に後半か前半か
        if t >= 30:  # 約30秒以降がE majorセクション（概算）
            emaj_notes.append(n)
        else:
            em_notes.append(n)

print(f"\nEm部（前半 ~30s）: {len(em_notes)} notes")
print(f"E major部（後半 30s~）: {len(emaj_notes)} notes")

# 前半と後半のフレット分布比較
print("\n--- フレット分布比較 ---")
em_frets = Counter(n.get('fret', 0) for n in em_notes)
emaj_frets = Counter(n.get('fret', 0) for n in emaj_notes)
print(f"Em部:  {sorted(em_frets.items())}")
print(f"Emaj部: {sorted(emaj_frets.items())}")

# 前半と後半の弦分布比較
print("\n--- 弦分布比較 ---")
em_strings = Counter(n.get('string', 0) for n in em_notes)
emaj_strings = Counter(n.get('string', 0) for n in emaj_notes)
print(f"Em部:  {sorted(em_strings.items())}")
print(f"Emaj部: {sorted(emaj_strings.items())}")

# --- PDF正解のE majorセクションのポジション特徴 ---
print(f"\n{'=' * 70}")
print("PDF正解: E majorセクションのポジション特徴")
print("=" * 70)
print("""
PDFの正解を読み取ると、E majorセクション（小節18~）の特徴:

小節18-19 (E): fret 4,0,1 / 4,0,1 → 1stポジション  
  メロディ: G#4=s1/f4, アルペジオ: E4=s1/f0, C#4=s2/f2(推定)
  ベース: E2=s6/f0

小節20-21 (A/E): fret 5,2,4 / 5,2,4  
  メロディ: A4=s1/f5, アルペジオ: D#4=s2/f4
  ベース: A2=s5/f0

小節22-23 (B7): fret 9,7,8 / 9,7,8  
  メロディ: C#5=s1/f9, アルペジオ: B3=s2/f7(推定), G#3=s3/f8(推定)
  → ここは7thポジション！

小節24-25 (E/B7): fret 7,9,9 / 7,9,9
  → 7thポジション維持

正解パターンの特徴:
1. E majorセクションでも基本は1stポジション（fret 0-5）
2. B7/E7セクション（小節22-25）のみ7thポジションに移動
3. ポジション変更はコード進行に完全に連動
""")

# --- 不正解パターンの根本原因分析 ---
print(f"\n{'=' * 70}")
print("不正解の根本原因分析")
print("=" * 70)

# 各不正解ピッチについて、前後のノートのフレット位置を確認
wrong_pitches = {63: "D#4", 68: "G#4", 66: "F#4", 60: "C4", 47: "B2", 56: "G#3", 61: "C#4"}
for pitch, name in sorted(wrong_pitches.items()):
    wrong_notes = [n for n in notes 
                   if n["pitch"] == pitch 
                   and pitch in CORRECT_EM
                   and (n.get("string"), n.get("fret")) != CORRECT_EM[pitch]]
    if not wrong_notes:
        continue
    
    exp_s, exp_f = CORRECT_EM[pitch]
    print(f"\n{name} (MIDI {pitch}): {len(wrong_notes)} 件不正解")
    print(f"  正解: s{exp_s}/f{exp_f}")
    
    # 不正解パターン
    act_patterns = Counter((n.get('string'), n.get('fret')) for n in wrong_notes)
    for (s, f), cnt in act_patterns.most_common(3):
        print(f"  → s{s}/f{f} x{cnt}")
    
    # 前後のノートのフレット位置を確認（最初の3件）
    for wn in wrong_notes[:3]:
        idx = notes.index(wn)
        prev_frets = [notes[max(0,idx-j)].get('fret', 0) for j in range(1, 4)]
        next_frets = [notes[min(len(notes)-1,idx+j)].get('fret', 0) for j in range(1, 4)]
        prev_strings = [notes[max(0,idx-j)].get('string', 0) for j in range(1, 4)]
        
        # このノートのコード
        t = wn.get('start', 0)
        cur_chord = "?"
        for chord in chords:
            cs = chord.get('start', chord.get('time', 0))
            ce = chord.get('end', cs + chord.get('duration', 999))
            if cs <= t < ce:
                cur_chord = chord.get('chord', '?')
                break
        
        print(f"    t={wn['start']:.1f}s chord={cur_chord} "
              f"prev_f={prev_frets} next_f={next_frets} "
              f"prev_s={prev_strings}")
