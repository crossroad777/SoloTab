"""正解TABとSoloTab出力の差分分析"""
import json
from music_theory import PC_NAMES

notes = json.load(open('D:/Music/nextchord-solotab/uploads/20260429-003136/notes_assigned.json'))

# 正解: 禁じられた遊び m.1 (Em) - 3/4拍子, 3連符, 9ノート/小節
# Beat1: E2(6弦0), B4(1弦7), B3(2弦0)
# Beat2: G3(3弦0), B4(1弦7), B3(2弦0)
# Beat3: G3(3弦0), B4(1弦7), B3(2弦0)

correct_m1 = [
    {'pitch': 40, 'string': 6, 'fret': 0, 'name': 'E2 bass'},
    {'pitch': 71, 'string': 1, 'fret': 7, 'name': 'B4 melody'},
    {'pitch': 59, 'string': 2, 'fret': 0, 'name': 'B3 inner'},
    {'pitch': 55, 'string': 3, 'fret': 0, 'name': 'G3 inner'},
    {'pitch': 71, 'string': 1, 'fret': 7, 'name': 'B4 melody'},
    {'pitch': 59, 'string': 2, 'fret': 0, 'name': 'B3 inner'},
    {'pitch': 55, 'string': 3, 'fret': 0, 'name': 'G3 inner'},
    {'pitch': 71, 'string': 1, 'fret': 7, 'name': 'B4 melody'},
    {'pitch': 59, 'string': 2, 'fret': 0, 'name': 'B3 inner'},
]

# SoloTab出力の最初の1小節 (t=1.90~3.90)
solotab_m1 = [n for n in notes if 1.85 <= n['start'] < 3.90]

print("=== 正解 m.1 ===")
for i, c in enumerate(correct_m1):
    print(f"  {i+1}. pitch={c['pitch']}({PC_NAMES[c['pitch']%12]}) s={c['string']} f={c['fret']}  [{c['name']}]")

print(f"\n=== SoloTab m.1 ({len(solotab_m1)} notes) ===")
for i, n in enumerate(solotab_m1):
    pc = PC_NAMES[n['pitch'] % 12]
    print(f"  {i+1}. t={n['start']:.2f} pitch={n['pitch']}({pc}) s={n['string']} f={n['fret']}")

# 差分分析
print("\n=== 差分分析 ===")
print(f"ノート数: 正解={len(correct_m1)}, SoloTab={len(solotab_m1)}")

# ピッチ比較
correct_pitches = sorted([c['pitch'] for c in correct_m1])
solotab_pitches = sorted([n['pitch'] for n in solotab_m1])
print(f"ピッチ集合 正解: {correct_pitches}")
print(f"ピッチ集合 出力: {solotab_pitches}")

# 弦/フレット比較
print("\n=== 弦割り当ての差異 ===")
for n in solotab_m1:
    p, s, f = n['pitch'], n['string'], n['fret']
    # 同じピッチの正解を探す
    for c in correct_m1:
        if c['pitch'] == p:
            if c['string'] != s or c['fret'] != f:
                print(f"  pitch={p}({PC_NAMES[p%12]}): "
                      f"正解=s{c['string']}/f{c['fret']} ← SoloTab=s{s}/f{f}")
            break
