"""
analyze_idmt_fingering.py — IDMT人間運指 vs 理論運指の比較分析
================================================================
IDMTのXMLから人間の弦・フレット選択を読み取り、
理論的最低ポジション（lowest fret）との違いを分析する。

Usage:
    python backend/train/analyze_idmt_fingering.py
"""
import os, glob, xml.etree.ElementTree as ET
from collections import Counter, defaultdict

IDMT_ROOT = r"D:\Music\datasets\IDMT-SMT-GUITAR_V2\IDMT-SMT-GUITAR_V2"

# 標準チューニング (E2=40, A2=45, D3=50, G3=55, B3=59, E4=64)
# IDMTのstringNumber: 1=low E, 2=A, 3=D, 4=G, 5=B, 6=high E
STANDARD_TUNING = {1: 40, 2: 45, 3: 50, 4: 55, 5: 59, 6: 64}


def theoretical_lowest_fret(pitch):
    """理論上の最低フレットポジション（最も太い弦を優先）"""
    for string_num in range(1, 7):  # 1=low E から
        open_pitch = STANDARD_TUNING[string_num]
        fret = pitch - open_pitch
        if 0 <= fret <= 19:
            return string_num, fret
    return None, None


def all_possible_positions(pitch):
    """あるピッチを弾ける全ポジション"""
    positions = []
    for string_num in range(1, 7):
        open_pitch = STANDARD_TUNING[string_num]
        fret = pitch - open_pitch
        if 0 <= fret <= 19:
            positions.append((string_num, fret))
    return positions


def parse_all_xmls():
    """全XMLを解析してノートデータを収集"""
    xml_files = glob.glob(os.path.join(IDMT_ROOT, "**", "*.xml"), recursive=True)
    print(f"Found {len(xml_files)} XML files")

    all_notes = []
    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for event in root.iter("event"):
                pitch = int(event.findtext("pitch", "0"))
                string = int(event.findtext("stringNumber", "0"))
                fret = int(event.findtext("fretNumber", "0"))
                excitation = event.findtext("excitationStyle", "")
                expression = event.findtext("expressionStyle", "")

                if pitch > 0 and string > 0:
                    all_notes.append({
                        "pitch": pitch,
                        "string": string,
                        "fret": fret,
                        "excitation": excitation,
                        "expression": expression,
                        "file": os.path.basename(xml_path),
                    })
        except Exception as e:
            pass

    print(f"Total notes: {len(all_notes)}")
    return all_notes


def analyze(notes):
    """人間運指 vs 理論運指の比較"""

    # === 1. 基本統計 ===
    print(f"\n{'='*60}")
    print(f"  1. 基本統計")
    print(f"{'='*60}")

    pitch_range = [n["pitch"] for n in notes]
    print(f"  ピッチ範囲: {min(pitch_range)} - {max(pitch_range)} (MIDI)")
    print(f"  弦分布:")
    string_counts = Counter(n["string"] for n in notes)
    for s in range(1, 7):
        print(f"    弦{s} ({['','E2','A2','D3','G3','B3','E4'][s]}): {string_counts.get(s,0):5d} ({string_counts.get(s,0)/len(notes)*100:.1f}%)")

    print(f"  奏法分布:")
    for ex, cnt in Counter(n["excitation"] for n in notes).most_common():
        print(f"    {ex}: {cnt}")

    # === 2. 人間 vs 最低フレット ===
    print(f"\n{'='*60}")
    print(f"  2. 人間ポジション vs 理論最低フレット")
    print(f"{'='*60}")

    same_count = 0
    diff_count = 0
    higher_fret_count = 0
    higher_string_count = 0  # 高い弦（細い弦）を選んだケース
    fret_diffs = []
    interesting_cases = []

    for n in notes:
        theo_string, theo_fret = theoretical_lowest_fret(n["pitch"])
        if theo_string is None:
            continue

        if n["string"] == theo_string and n["fret"] == theo_fret:
            same_count += 1
        else:
            diff_count += 1
            fret_diff = n["fret"] - theo_fret
            fret_diffs.append(fret_diff)

            if n["fret"] > theo_fret:
                higher_fret_count += 1
            if n["string"] > theo_string:
                higher_string_count += 1

            if abs(fret_diff) >= 5:
                interesting_cases.append(n)

    total = same_count + diff_count
    print(f"  理論と一致: {same_count}/{total} ({same_count/total*100:.1f}%)")
    print(f"  理論と不一致: {diff_count}/{total} ({diff_count/total*100:.1f}%)")
    print(f"    うち、より高いフレット: {higher_fret_count} ({higher_fret_count/total*100:.1f}%)")
    print(f"    うち、より高い弦(細い弦): {higher_string_count} ({higher_string_count/total*100:.1f}%)")

    if fret_diffs:
        import statistics
        print(f"\n  フレット差分統計 (人間 - 理論):")
        print(f"    平均: {statistics.mean(fret_diffs):+.2f}")
        print(f"    中央値: {statistics.median(fret_diffs):+.1f}")
        print(f"    最大: {max(fret_diffs):+d}")
        print(f"    最小: {min(fret_diffs):+d}")

    # === 3. フレット差分のヒストグラム ===
    print(f"\n{'='*60}")
    print(f"  3. フレット差分分布 (人間フレット - 理論フレット)")
    print(f"{'='*60}")

    diff_hist = Counter(fret_diffs)
    for diff in sorted(diff_hist.keys()):
        bar = "#" * min(diff_hist[diff] // 5, 50)
        print(f"  {diff:+3d}: {diff_hist[diff]:5d} {bar}")

    # === 4. ポジション維持パターン ===
    print(f"\n{'='*60}")
    print(f"  4. 人間のポジション選択パターン")
    print(f"{'='*60}")

    # pitch → 人間が選んだ(string, fret)の分布
    pitch_positions = defaultdict(list)
    for n in notes:
        pitch_positions[n["pitch"]].append((n["string"], n["fret"]))

    multi_position_pitches = 0
    for pitch, positions in sorted(pitch_positions.items()):
        unique = set(positions)
        if len(unique) > 1:
            multi_position_pitches += 1

    print(f"  複数ポジションで弾かれたピッチ: {multi_position_pitches}/{len(pitch_positions)}")

    # 最も多様なポジション選択のピッチ Top 10
    print(f"\n  ピッチごとの人間ポジション選択 (多様性 Top 10):")
    diversity = []
    for pitch, positions in pitch_positions.items():
        unique = set(positions)
        if len(unique) > 1 and len(positions) >= 5:
            diversity.append((pitch, unique, len(positions)))
    diversity.sort(key=lambda x: len(x[1]), reverse=True)

    for pitch, unique_pos, total_count in diversity[:10]:
        theo_s, theo_f = theoretical_lowest_fret(pitch)
        pos_strs = [f"S{s}F{f}" for s, f in sorted(unique_pos)]
        print(f"    MIDI {pitch}: {len(unique_pos)} positions ({total_count} notes) -> {', '.join(pos_strs)}")
        print(f"      理論: S{theo_s}F{theo_f}")

    # === 5. 弦選択の偏り ===
    print(f"\n{'='*60}")
    print(f"  5. 人間の弦選択 vs 理論弦選択")
    print(f"{'='*60}")

    string_migration = Counter()
    for n in notes:
        theo_s, _ = theoretical_lowest_fret(n["pitch"])
        if theo_s:
            key = f"理論S{theo_s}→人間S{n['string']}"
            string_migration[key] += 1

    for key, cnt in string_migration.most_common(15):
        print(f"    {key}: {cnt}")

    # === 6. 興味深いケース ===
    if interesting_cases:
        print(f"\n{'='*60}")
        print(f"  6. 興味深いケース（フレット差 >= 5）")
        print(f"{'='*60}")

        for n in interesting_cases[:20]:
            theo_s, theo_f = theoretical_lowest_fret(n["pitch"])
            print(f"    MIDI={n['pitch']} 人間:S{n['string']}F{n['fret']} 理論:S{theo_s}F{theo_f} ({n['excitation']}/{n['expression']}) [{n['file']}]")


if __name__ == "__main__":
    notes = parse_all_xmls()
    analyze(notes)
