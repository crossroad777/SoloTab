"""chord-collection 167K ボイシングを集計・変換"""
import sys, json
sys.stdout.reconfigure(encoding='utf-8')

with open(r'D:\Music\nextchord-solotab\datasets\chord-collection\chords.complete.json') as f:
    data = json.load(f)

total_voicings = 0
total_with_fingers = 0
total_chords = len(data)

# サンプル確認
samples = []
for chord_name, voicing_list in list(data.items())[:5]:
    for v in voicing_list[:2]:
        samples.append((chord_name, v))

for name, v in samples:
    print(f"{name}: pos={v['positions']} fingers={v['fingerings']}")

# 全体集計
for chord_name, voicing_list in data.items():
    for v in voicing_list:
        total_voicings += 1
        fingerings = v.get('fingerings', [])
        if fingerings and any(f != '0' and f != 'x' for fg in fingerings for f in fg):
            total_with_fingers += 1

print(f"\n合計: {total_chords:,} コード名, {total_voicings:,} ボイシング")
print(f"指番号あり: {total_with_fingers:,} ({total_with_fingers/total_voicings*100:.0f}%)")

# コード名分析
chord_types = {}
for name in data.keys():
    # "Cmaj7" → root="C", type="maj7"
    for r in ['C#','Db','D#','Eb','F#','Gb','G#','Ab','A#','Bb','C','D','E','F','G','A','B']:
        if name.startswith(r):
            ct = name[len(r):] or 'major'
            chord_types[ct] = chord_types.get(ct, 0) + 1
            break

print(f"\nコードタイプ: {len(chord_types)} unique")
for ct, count in sorted(chord_types.items(), key=lambda x: -x[1])[:20]:
    print(f"  {ct:20s}: {count}")
