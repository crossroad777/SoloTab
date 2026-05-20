"""UCIコードデータをパースして guitar_fingering_db.py 用のコードフォームDBに変換"""
import sys, json, csv, re
sys.stdout.reconfigure(encoding='utf-8')

INPUT = r"D:\Music\nextchord-solotab\datasets\uci_guitar_chords\chord-fingers.csv"
OUTPUT = r"D:\Music\nextchord-solotab\backend\uci_chord_fingers.json"

with open(INPUT, 'r') as f:
    content = f.read()

chords = {}
lines = content.strip().split('\n')[1:]  # skip header

for line in lines:
    # Parse: ROOT;TYPE;"STRUCTURE";FINGER_POS;NOTE_NAMES
    # STRUCTURE has semicolons inside quotes
    # Split on ; but respect quotes
    parts = []
    in_quote = False
    current = ""
    for ch in line:
        if ch == '"':
            in_quote = not in_quote
            current += ch
        elif ch == ';' and not in_quote:
            parts.append(current)
            current = ""
        else:
            current += ch
    parts.append(current)
    
    if len(parts) < 4:
        continue
    
    root = parts[0].strip()
    chord_type = parts[1].strip()
    finger_str = parts[3].strip()  # "x,1,0,2,3,4"
    
    fingers = finger_str.split(',')
    if len(fingers) != 6:
        continue
    
    # 弦6→弦1 の指番号
    chord_key = f"{root}_{chord_type}"
    finger_map = []
    for s_idx, f in enumerate(fingers):
        string = 6 - s_idx  # 6,5,4,3,2,1
        f = f.strip()
        if f == 'x':
            finger_map.append((string, -1))  # mute
        elif f == '0':
            finger_map.append((string, 0))   # open
        else:
            try:
                finger_map.append((string, int(f)))
            except:
                finger_map.append((string, -1))
    
    if chord_key not in chords:
        chords[chord_key] = []
    chords[chord_key].append(finger_map)

# 統計
total_chords = sum(len(v) for v in chords.values())
unique_names = len(chords)
print(f"パース完了: {total_chords} コードフォーム, {unique_names} ユニークコード名")

# コードタイプ別集計
type_count = {}
for k in chords:
    parts = k.split('_', 1)
    if len(parts) == 2:
        t = parts[1]
        type_count[t] = type_count.get(t, 0) + len(chords[k])

print(f"\nコードタイプTOP20:")
for t, c in sorted(type_count.items(), key=lambda x: -x[1])[:20]:
    print(f"  {t:20s}: {c}")

# JSON保存
with open(OUTPUT, 'w', encoding='utf-8') as f:
    # リストのリストをJSON化
    out = {}
    for k, variants in chords.items():
        out[k] = [
            {str(s): finger for s, finger in variant}
            for variant in variants
        ]
    json.dump(out, f, indent=1, ensure_ascii=False)

print(f"\n保存: {OUTPUT} ({unique_names} chords, {total_chords} variants)")
