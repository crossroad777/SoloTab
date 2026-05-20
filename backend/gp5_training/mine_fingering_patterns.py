"""
GP5コーパス480万ノートからの運指パターンマイニング v2
====================================================
JONLフォーマット: context_strings/context_frets/target_string/target_fret
"""
import sys, os, json, time
from collections import Counter, defaultdict
sys.stdout.reconfigure(encoding='utf-8')

JSONL_PATH = r"D:\Music\nextchord-solotab\gp_training_data\fingering_train.jsonl"
OUTPUT = r"D:\Music\nextchord-solotab\gp_training_data\mined_fingering_patterns.json"

def derive_position(frets):
    """フレット列から最適ポジションを推定"""
    fretted = [f for f in frets if f > 0]
    if not fretted:
        return 0
    min_f = min(fretted)
    max_f = max(fretted)
    if max_f - min_f <= 3:
        return min_f
    for pos in range(max(1, min_f), max_f + 1):
        if all(0 <= f - pos <= 3 or f == 0 for f in frets):
            return pos
    return min_f

print("=" * 60)
print("  GP5 運指パターンマイニング v2")
print("=" * 60)

t0 = time.time()

# カウンター
chord_fret_patterns = Counter()   # 同時発音 (string,fret) タプル
scale_runs_2 = Counter()          # 同弦2連続 (string, fret1, fret2)
scale_runs_3 = Counter()          # 同弦3連続 (string, f1, f2, f3)
fret_string_freq = Counter()      # (string, fret) 単体頻度
position_usage = Counter()        # 推定ポジション使用頻度
cross_string_patterns = Counter() # 異弦遷移 (s1,f1, s2,f2)

note_count = 0
line_count = 0

with open(JSONL_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        line_count += 1
        if line_count % 500000 == 0:
            elapsed = time.time() - t0
            print(f"  ...{line_count:,} lines ({elapsed:.0f}s) "
                  f"chords={len(chord_fret_patterns)} runs2={len(scale_runs_2)}")
        
        try:
            d = json.loads(line.strip())
        except:
            continue
        
        note_count += 1
        ts = d.get('target_string', 0)
        tf = d.get('target_fret', 0)
        cs = d.get('context_strings', [])
        cf = d.get('context_frets', [])
        is_chord = d.get('is_chord', False)
        
        fret_string_freq[(ts, tf)] += 1
        
        if not cs or not cf or len(cs) != len(cf):
            continue
        
        # 直前のノート（context末尾）
        prev_s = cs[-1]
        prev_f = cf[-1]
        
        # 同弦2連続ラン
        if prev_s == ts:
            scale_runs_2[(ts, prev_f, tf)] += 1
        
        # 異弦遷移
        if prev_s != ts:
            cross_string_patterns[(prev_s, prev_f, ts, tf)] += 1
        
        # 同弦3連続ラン (context[-2], context[-1], target)
        if len(cs) >= 2 and cs[-2] == prev_s == ts:
            scale_runs_3[(ts, cf[-2], prev_f, tf)] += 1
        
        # 和音パターン（is_chord=Trueの場合、contextの最後とtargetが同時）
        if is_chord:
            chord_key = tuple(sorted([(prev_s, prev_f), (ts, tf)]))
            chord_fret_patterns[chord_key] += 1

        # ポジション推定
        frets_window = cf[-4:] + [tf]
        pos = derive_position(frets_window)
        if pos > 0:
            position_usage[pos] += 1

elapsed = time.time() - t0
print(f"\n  処理完了: {note_count:,} notes in {elapsed:.0f}s")

# === 結果集計 ===

# 1. 同弦2連続ラン → ポジションと指番号を導出
print(f"\n=== 同弦2連続ラン: {len(scale_runs_2):,} unique ===")
scale_fingerings = {}
for (string, f1, f2), count in scale_runs_2.most_common(2000):
    if count < 20:
        continue
    pos = derive_position([f1, f2])
    def fret_to_finger(f, p):
        if f == 0: return 0
        off = f - p
        if 0 <= off <= 3: return off + 1
        return -1
    fg1 = fret_to_finger(f1, pos)
    fg2 = fret_to_finger(f2, pos)
    if fg1 >= 0 and fg2 >= 0:
        key = f"{string}-{f1}-{f2}"
        scale_fingerings[key] = {
            'count': count, 'position': pos,
            'finger_from': fg1, 'finger_to': fg2,
        }

print(f"  指番号導出可能: {len(scale_fingerings)} patterns")
for k, v in list(scale_fingerings.items())[:10]:
    fn = {0:'O',1:'I',2:'M',3:'R',4:'P'}
    print(f"    {k}: {v['count']:>6}x pos={v['position']} "
          f"{fn.get(v['finger_from'],'?')}→{fn.get(v['finger_to'],'?')}")

# 2. 同弦3連続ラン
print(f"\n=== 同弦3連続ラン: {len(scale_runs_3):,} unique ===")
run3_fingerings = {}
for (string, f1, f2, f3), count in scale_runs_3.most_common(1000):
    if count < 10:
        continue
    pos = derive_position([f1, f2, f3])
    def fret_to_finger(f, p):
        if f == 0: return 0
        off = f - p
        if 0 <= off <= 3: return off + 1
        return -1
    fg1 = fret_to_finger(f1, pos)
    fg2 = fret_to_finger(f2, pos)
    fg3 = fret_to_finger(f3, pos)
    if fg1 >= 0 and fg2 >= 0 and fg3 >= 0:
        key = f"{string}-{f1}-{f2}-{f3}"
        run3_fingerings[key] = {
            'count': count, 'position': pos,
            'fingers': [fg1, fg2, fg3],
        }

print(f"  指番号導出可能: {len(run3_fingerings)} patterns")
for k, v in list(run3_fingerings.items())[:10]:
    fn = {0:'O',1:'I',2:'M',3:'R',4:'P'}
    fstr = '→'.join(fn.get(f,'?') for f in v['fingers'])
    print(f"    {k}: {v['count']:>5}x pos={v['position']} {fstr}")

# 3. 和音パターン
print(f"\n=== 和音パターン: {len(chord_fret_patterns):,} unique ===")
chord_fingerings = {}
for pattern, count in chord_fret_patterns.most_common(1000):
    if count < 10:
        continue
    frets = [f for _, f in pattern]
    pos = derive_position(frets)
    fingers = {}
    for s, f in pattern:
        if f == 0:
            fingers[f"{s}-{f}"] = 0
        else:
            off = f - pos
            if 0 <= off <= 3:
                fingers[f"{s}-{f}"] = off + 1
    if fingers:
        chord_fingerings[str(pattern)] = {
            'count': count, 'position': pos, 'fingers': fingers
        }

print(f"  指番号導出可能: {len(chord_fingerings)} patterns")
for k, v in list(chord_fingerings.items())[:5]:
    print(f"    {k}: {v['count']:>5}x pos={v['position']} {v['fingers']}")

# 4. ポジション使用頻度
print(f"\n=== ポジション使用頻度 TOP15 ===")
for pos, count in position_usage.most_common(15):
    pct = count / sum(position_usage.values()) * 100
    print(f"  pos={pos:2d}: {count:>8,} ({pct:.1f}%)")

# 5. (string, fret) 頻度
print(f"\n=== 弦フレット頻度 TOP20 ===")
for (s, f), count in fret_string_freq.most_common(20):
    print(f"  S{s} F{f:2d}: {count:>8,}")

# 保存
output = {
    'metadata': {
        'total_notes': note_count,
        'elapsed_seconds': round(elapsed, 1),
        'scale_run2_patterns': len(scale_fingerings),
        'scale_run3_patterns': len(run3_fingerings),
        'chord_patterns': len(chord_fingerings),
    },
    'scale_run2_fingerings': scale_fingerings,
    'scale_run3_fingerings': run3_fingerings,
    'chord_fingerings': chord_fingerings,
    'position_usage': {str(k): v for k, v in position_usage.most_common(24)},
    'top_fret_string': {f"{s}-{f}": c for (s, f), c in fret_string_freq.most_common(100)},
}

with open(OUTPUT, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=1, ensure_ascii=False)
print(f"\n保存: {OUTPUT}")
print(f"  2連続ラン: {len(scale_fingerings)}")
print(f"  3連続ラン: {len(run3_fingerings)}")
print(f"  和音: {len(chord_fingerings)}")
