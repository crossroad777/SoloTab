"""
Romance (Romanza / Forbidden Games) — 正解タブとの完全比較

ロマンス（禁じられた遊び）のEm部（前半）の正確なタブ譜：
3/4拍子、3連符アルペジオ
各拍: ベース音 + トレブル3音（又はトレブル3音のみ）

標準チューニング (EADGBE)
"""
import json, sys
from collections import Counter, defaultdict
sys.path.insert(0, r"D:\Music\nextchord-solotab\backend")

SESSION = r"D:\Music\nextchord-solotab\uploads\20260510-082310"

with open(f"{SESSION}/notes_assigned.json", "r", encoding="utf-8") as f:
    notes = json.load(f)
with open(f"{SESSION}/beats.json", "r", encoding="utf-8") as f:
    beat_data = json.load(f)
beats = beat_data.get("beats", [])

MIDI_TO_NOTE = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

# ===================================================================
# ロマンス Em部 正解タブ（小節1-16）
# 各小節は3拍、各拍は3連符（3音）
# フォーマット: [(string, fret), ...] per beat
# ベース音は拍頭に (string, fret) で追加
# ===================================================================
# 小節1: Em (B4-B3-G3 アルペジオ + E2ベース)
# 小節2: Em
# 小節3: Em → Am
# 小節4: Am / E
# 小節5-8: 同様のパターン
#
# 正解のMIDIピッチと弦フレット割り当て:
# E2=40: str6/f0
# B2=47: str5/f2
# A2=45: str5/f0
# G3=55: str3/f0
# G#3=56: str3/f1
# A3=57: str3/f2
# B3=59: str2/f0
# C4=60: str2/f1
# D4=62: str2/f3
# D#4=63: str2/f4
# E4=64: str1/f0
# F#4=66: str1/f2
# G4=67: str1/f3
# G#4=68: str1/f4
# A4=69: str1/f5
# B4=71: str1/f7
# C5=72: str1/f8
# C#5=73: str1/f9
# D5=74: str1/f10
# D#5=75: str1/f11
# E5=76: str1/f12

# ロマンスの正解ポジションマップ（Emセクション）
CORRECT_POSITIONS = {
    # ベース音（6弦・5弦）
    40: (6, 0),   # E2
    43: (6, 3),   # G2
    44: (6, 4),   # G#2
    45: (5, 0),   # A2
    47: (5, 2),   # B2
    # 中音域（4弦・3弦）
    52: (4, 2),   # E3
    54: (4, 4),   # F#3
    55: (3, 0),   # G3
    56: (3, 1),   # G#3
    57: (3, 2),   # A3
    # アルペジオ音（2弦・1弦）
    59: (2, 0),   # B3
    60: (2, 1),   # C4
    62: (2, 3),   # D4
    63: (2, 4),   # D#4
    64: (1, 0),   # E4
    66: (1, 2),   # F#4
    67: (1, 3),   # G4
    68: (1, 4),   # G#4
    69: (1, 5),   # A4
    71: (1, 7),   # B4
    72: (1, 8),   # C5
    73: (1, 9),   # C#5
    74: (1, 10),  # D5
    75: (1, 11),  # D#5
    76: (1, 12),  # E5
}

# --- 比較 ---
print("=" * 70)
print("ロマンス（禁じられた遊び）正解タブとの比較")
print("=" * 70)

total = 0
correct = 0
wrong = 0
wrong_details = defaultdict(list)

for i, n in enumerate(notes):
    pitch = n["pitch"]
    actual_s = n.get("string", -1)
    actual_f = n.get("fret", -1)
    
    if pitch in CORRECT_POSITIONS:
        expected_s, expected_f = CORRECT_POSITIONS[pitch]
        total += 1
        if actual_s == expected_s and actual_f == expected_f:
            correct += 1
        else:
            wrong += 1
            note_name = MIDI_TO_NOTE[pitch % 12] + str(pitch // 12 - 1)
            wrong_details[pitch].append({
                "idx": i, "time": n["start"],
                "actual": f"s{actual_s}/f{actual_f}",
                "expected": f"s{expected_s}/f{expected_f}",
            })
    else:
        # 正解マップにないピッチ（想定外の音 or E majorセクション固有）
        total += 1
        # 音域的に妥当かチェック
        if 40 <= pitch <= 76:
            note_name = MIDI_TO_NOTE[pitch % 12] + str(pitch // 12 - 1)
            # 推奨ポジションを計算（最低フレット優先）
            TUNING = [40, 45, 50, 55, 59, 64]
            best = None
            for s_idx, op in enumerate(TUNING):
                f = pitch - op
                if 0 <= f <= 12:
                    s_num = 6 - s_idx
                    if best is None or f < best[1]:
                        best = (s_num, f)
            if best and (actual_s != best[0] or actual_f != best[1]):
                wrong += 1
                wrong_details[pitch].append({
                    "idx": i, "time": n["start"],
                    "actual": f"s{actual_s}/f{actual_f}",
                    "expected": f"s{best[0]}/f{best[1]} (推定)",
                })
            else:
                correct += 1
        else:
            correct += 1  # 音域外は判定不能

accuracy = correct / total * 100 if total > 0 else 0
print(f"\n### 全体精度: {correct}/{total} = {accuracy:.1f}%")
print(f"    正解: {correct}, 不正解: {wrong}")

# ピッチ別の不正解サマリー
if wrong_details:
    print(f"\n### ピッチ別 不正解一覧 ({len(wrong_details)} ピッチ)")
    for pitch in sorted(wrong_details.keys()):
        items = wrong_details[pitch]
        note_name = MIDI_TO_NOTE[pitch % 12] + str(pitch // 12 - 1)
        expected = items[0]["expected"]
        actual_counts = Counter(item["actual"] for item in items)
        total_for_pitch = sum(1 for n in notes if n["pitch"] == pitch)
        correct_for_pitch = total_for_pitch - len(items)
        print(f"\n  {note_name} (MIDI {pitch}): {len(items)}/{total_for_pitch} 不正解")
        print(f"    正解: {expected}")
        print(f"    不正解パターン: {dict(actual_counts)}")
        # 最初の3つの不正解例を表示
        for item in items[:3]:
            print(f"      [{item['idx']}] t={item['time']:.3f}s: {item['actual']} (正解: {item['expected']})")
else:
    print("\n### 全ノート正解！")

# --- 小節ごとの表示 ---
print(f"\n{'='*70}")
print("小節ごとのタブ出力 vs 正解パターン")
print(f"{'='*70}")

# ビートから小節を構築（3/4拍子）
BEATS_PER_BAR = 3
measures = []
for bi in range(0, len(beats)-BEATS_PER_BAR, BEATS_PER_BAR):
    bar_start = beats[bi]
    bar_end = beats[bi + BEATS_PER_BAR] if bi + BEATS_PER_BAR < len(beats) else bar_start + 2.0
    bar_notes = [n for n in notes if bar_start <= n["start"] < bar_end]
    measures.append({"start": bar_start, "end": bar_end, "notes": bar_notes})

# 最初の12小節を表示
for mi, m in enumerate(measures[:12]):
    # 小節内のノートをビートごとに分割
    bar_beats = []
    for b in range(BEATS_PER_BAR):
        bi = mi * BEATS_PER_BAR + b
        if bi >= len(beats) - 1:
            break
        bt, nbt = beats[bi], beats[bi + 1]
        beat_notes = [n for n in m["notes"] if bt <= n["start"] < nbt]
        bar_beats.append(beat_notes)
    
    print(f"\n--- 小節 {mi+1} (t={m['start']:.2f}s) ---")
    tab_lines = {s: [] for s in range(1, 7)}
    
    for b_idx, beat_notes in enumerate(bar_beats):
        for s in range(1, 7):
            matching = [n for n in beat_notes if n.get("string") == s]
            if matching:
                frets = [str(n.get("fret", "?")) for n in matching]
                tab_lines[s].append("/".join(frets).center(6))
            else:
                tab_lines[s].append("---".center(6))
    
    for s in range(1, 7):
        line = "|".join(tab_lines[s])
        # 弦の正解チェック
        print(f"  str{s}: {line}")
    
    # 正解との比較
    issues = []
    for n in m["notes"]:
        pitch = n["pitch"]
        if pitch in CORRECT_POSITIONS:
            exp_s, exp_f = CORRECT_POSITIONS[pitch]
            act_s, act_f = n.get("string", -1), n.get("fret", -1)
            if act_s != exp_s or act_f != exp_f:
                nn = MIDI_TO_NOTE[pitch % 12] + str(pitch // 12 - 1)
                issues.append(f"{nn}: s{act_s}/f{act_f} → 正解 s{exp_s}/f{exp_f}")
    if issues:
        print(f"  ⚠ 不正解: {', '.join(issues[:3])}")
    else:
        print(f"  ✅ 全ノート正解")
