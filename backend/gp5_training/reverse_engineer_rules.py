"""
chords-db 3,283ボイシングから運指ルールを逆算する
==================================================
frets + fingers の正解ペアから、人間の運指判断パターンを抽出
"""
import sys, json
from collections import Counter, defaultdict
sys.stdout.reconfigure(encoding='utf-8')

with open(r'D:\Music\nextchord-solotab\datasets\chords-db\lib\guitar.json') as f:
    data = json.load(f)

# 全ボイシングを展開
voicings = []
for key_chords in data['chords'].values():
    for chord in key_chords:
        for pos in chord.get('positions', []):
            voicings.append({
                'key': chord['key'],
                'suffix': chord['suffix'],
                'frets': pos['frets'],      # [6弦→1弦] -1=mute, 0=open
                'fingers': pos['fingers'],  # [6弦→1弦] 0=なし/開放
                'baseFret': pos['baseFret'],
                'barres': pos.get('barres', []),
            })

print(f"総ボイシング数: {len(voicings)}")

# ===== ルール逆算 =====

# 1. フレット→指のマッピング（ポジション相対）
fret_offset_to_finger = defaultdict(Counter)
for v in voicings:
    base = v['baseFret']
    for i in range(6):
        fret = v['frets'][i]
        finger = v['fingers'][i]
        if fret > 0 and finger > 0:
            actual_fret = fret + base - 1
            offset = fret - 1  # baseFretからの相対位置 (0-based)
            fret_offset_to_finger[offset][finger] += 1

print("\n=== R1: フレットオフセット → 指の対応（position相対）===")
for offset in sorted(fret_offset_to_finger.keys()):
    dist = fret_offset_to_finger[offset]
    total = sum(dist.values())
    print(f"  offset={offset}: ", end="")
    for f, c in dist.most_common():
        pct = c/total*100
        fname = {1:'index',2:'mid',3:'ring',4:'pinky'}[f]
        print(f"{fname}={pct:.0f}% ", end="")
    print()

# 2. バレー分析
barre_count = 0
barre_finger = Counter()
for v in voicings:
    if v['barres']:
        barre_count += 1
        # バレーフレットの指を確認
        for barre_fret in v['barres']:
            for i in range(6):
                if v['frets'][i] == barre_fret:
                    barre_finger[v['fingers'][i]] += 1
                    break

print(f"\n=== R2: バレーコード ===")
print(f"  バレー使用: {barre_count}/{len(voicings)} ({barre_count/len(voicings)*100:.0f}%)")
print(f"  バレー時の指: {dict(barre_finger.most_common())}")

# 3. 開放弦パターン
open_patterns = Counter()
for v in voicings:
    opens = tuple(i+1 for i in range(6) if v['frets'][i] == 0)
    if opens:
        open_patterns[opens] += 1

print(f"\n=== R3: 開放弦パターン TOP10 ===")
for pattern, count in open_patterns.most_common(10):
    strings = ','.join(f"{s}弦" for s in pattern)
    print(f"  {strings}: {count}回")

# 4. 指の使用頻度（弦別）
string_finger_usage = defaultdict(Counter)
for v in voicings:
    for i in range(6):
        string = 6 - i  # 6弦→1弦
        finger = v['fingers'][i]
        if finger > 0:
            string_finger_usage[string][finger] += 1

print(f"\n=== R4: 弦別の指使用頻度 ===")
for s in range(6, 0, -1):
    dist = string_finger_usage[s]
    total = sum(dist.values())
    if total == 0:
        continue
    print(f"  {s}弦: ", end="")
    for f, c in dist.most_common():
        pct = c/total*100
        fname = {1:'index',2:'mid',3:'ring',4:'pinky'}[f]
        print(f"{fname}={pct:.0f}% ", end="")
    print(f"  (n={total})")

# 5. 同時押弦時の指の組み合わせ
finger_combos = Counter()
for v in voicings:
    active = tuple(sorted(f for f in v['fingers'] if f > 0))
    if active:
        finger_combos[active] += 1

print(f"\n=== R5: 指の組み合わせ TOP15 ===")
for combo, count in finger_combos.most_common(15):
    names = '+'.join({1:'1',2:'2',3:'3',4:'4'}[f] for f in combo)
    print(f"  [{names}]: {count}回")

# 6. フレットスパン分析
span_dist = Counter()
for v in voicings:
    fretted = [v['frets'][i] for i in range(6) if v['frets'][i] > 0]
    if len(fretted) >= 2:
        span = max(fretted) - min(fretted)
        span_dist[span] += 1

print(f"\n=== R6: コード内フレットスパン分布 ===")
for span in sorted(span_dist.keys()):
    count = span_dist[span]
    print(f"  span={span}: {count}回 ({count/len(voicings)*100:.0f}%)")

# 7. 逆算ルール: offset→finger の最頻値マッピング
print(f"\n=== 逆算された基本ルール ===")
rules = {}
for offset in sorted(fret_offset_to_finger.keys()):
    best_finger = fret_offset_to_finger[offset].most_common(1)[0]
    rules[offset] = best_finger
    fname = {1:'index',2:'mid',3:'ring',4:'pinky'}[best_finger[0]]
    total = sum(fret_offset_to_finger[offset].values())
    pct = best_finger[1]/total*100
    print(f"  baseFret+{offset} → {fname} (確率{pct:.0f}%)")

# Save rules
output = {
    'fret_offset_rules': {str(k): dict(v) for k, v in fret_offset_to_finger.items()},
    'barre_stats': {'count': barre_count, 'total': len(voicings), 'finger_dist': dict(barre_finger)},
    'string_finger_usage': {str(k): dict(v) for k, v in string_finger_usage.items()},
    'span_distribution': dict(span_dist),
}
with open(r'D:\Music\nextchord-solotab\backend\derived_fingering_rules.json', 'w') as f:
    json.dump(output, f, indent=2)
print(f"\n保存: derived_fingering_rules.json")
