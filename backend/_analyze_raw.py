"""
前半(Em) → 後半(Emaj) のパターン対応を検証
MoEがEmとEmajで同じpitchを出力しているなら、前半の正しいフレットを+2して後半に使える
"""
import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sid = '20260429-114939'
base = f'D:/Music/nextchord-solotab/uploads/{sid}'

with open(f'{base}/notes.json') as f:
    raw = json.load(f)
    if isinstance(raw, dict): raw = raw.get('notes', [])

with open(f'{base}/beats.json') as f:
    beats = json.load(f)['beats']

measures = []
for i in range(0, len(beats)-2, 3):
    measures.append((beats[i], beats[i+3] if i+3 < len(beats) else beats[-1]+0.67))

PC = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

# 前半(M1-M16) と 後半(M17-M32) の1弦メロディ pitch を小節・ビート単位で比較
print("=== 前半(Em) vs 後半(Emaj) MoE raw pitch 比較 ===")
print(f"{'':>6} | {'Em (前半)':>15} | {'Emaj (後半)':>15} | 差分")
print("-" * 60)

em_pitches = []
emaj_pitches = []

for m_offset in range(16):
    em_m = m_offset        # M1-M16
    emaj_m = m_offset + 16 # M17-M32
    
    if em_m >= len(measures) or emaj_m >= len(measures):
        break
    
    em_ms, em_me = measures[em_m]
    emaj_ms, emaj_me = measures[emaj_m]
    
    for b in range(3):
        em_bs = em_ms + (em_me - em_ms) * b / 3
        em_be = em_ms + (em_me - em_ms) * (b + 1) / 3
        emaj_bs = emaj_ms + (emaj_me - emaj_ms) * b / 3
        emaj_be = emaj_ms + (emaj_me - emaj_ms) * (b + 1) / 3
        
        em_notes = [n for n in raw if em_bs <= n['start'] < em_be and n.get('string') == 1]
        emaj_notes = [n for n in raw if emaj_bs <= n['start'] < emaj_be and n.get('string') == 1]
        
        em_p = em_notes[0]['pitch'] if em_notes else None
        emaj_p = emaj_notes[0]['pitch'] if emaj_notes else None
        
        if em_p: em_pitches.append(em_p)
        if emaj_p: emaj_pitches.append(emaj_p)
        
        em_str = f"p={em_p} ({PC[em_p%12]})" if em_p else "---"
        emaj_str = f"p={emaj_p} ({PC[emaj_p%12]})" if emaj_p else "---"
        
        if em_p and emaj_p:
            diff = emaj_p - em_p
            print(f"M{em_m+1:>2}B{b+1} | {em_str:>15} | {emaj_str:>15} | {diff:+d}")
        else:
            print(f"M{em_m+1:>2}B{b+1} | {em_str:>15} | {emaj_str:>15} | -")

# 差分の統計
if em_pitches and emaj_pitches:
    min_len = min(len(em_pitches), len(emaj_pitches))
    diffs = [emaj_pitches[i] - em_pitches[i] for i in range(min_len)]
    zero_count = sum(1 for d in diffs if d == 0)
    near_zero = sum(1 for d in diffs if abs(d) <= 1)
    print(f"\n差分=0: {zero_count}/{len(diffs)} ({zero_count/len(diffs)*100:.0f}%)")
    print(f"差分±1以内: {near_zero}/{len(diffs)} ({near_zero/len(diffs)*100:.0f}%)")
    print(f"差分の平均: {sum(diffs)/len(diffs):.1f}")
