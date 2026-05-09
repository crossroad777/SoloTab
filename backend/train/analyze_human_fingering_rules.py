"""
analyze_human_fingering_rules.py — 人間運指ルールの深層分析
================================================================
IDMTの人間運指データから、ギタリストの暗黙のルールを抽出する。

分析項目:
  1. ポジション維持ルール（同一ポジション内での演奏傾向）
  2. 弦選択ルール（同じピッチの場合、どの弦を選ぶか）
  3. 開放弦使用パターン
  4. フレット範囲と弦の関係
  5. 連続音間のポジション遷移パターン
  6. ピッチ帯域別の弦選好

Usage:
    python backend/train/analyze_human_fingering_rules.py
"""
import os, glob, xml.etree.ElementTree as ET
from collections import Counter, defaultdict
import statistics

IDMT_ROOT = r"D:\Music\datasets\IDMT-SMT-GUITAR_V2\IDMT-SMT-GUITAR_V2"

# 標準チューニング (string_number → open pitch)
STANDARD_TUNING = {1: 40, 2: 45, 3: 50, 4: 55, 5: 59, 6: 64}
STRING_NAMES = {1: "6弦(E2)", 2: "5弦(A2)", 3: "4弦(D3)", 4: "3弦(G3)", 5: "2弦(B3)", 6: "1弦(E4)"}


def parse_all_xmls():
    """全XMLを解析。dataset2はフレーズ（連続音）を含む"""
    xml_files = glob.glob(os.path.join(IDMT_ROOT, "**", "*.xml"), recursive=True)

    # 単音ファイルとフレーズファイルを分離
    single_notes = []
    phrase_files = []  # [(filename, [notes])]

    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            notes = []
            for event in root.iter("event"):
                pitch = int(event.findtext("pitch", "0"))
                string = int(event.findtext("stringNumber", "0"))
                fret = int(event.findtext("fretNumber", "0"))
                onset = float(event.findtext("onsetSec", "0"))
                excitation = event.findtext("excitationStyle", "")
                expression = event.findtext("expressionStyle", "")

                if pitch > 0 and string > 0:
                    notes.append({
                        "pitch": pitch, "string": string, "fret": fret,
                        "onset": onset, "excitation": excitation,
                        "expression": expression,
                    })

            if len(notes) == 1:
                single_notes.append(notes[0])
            elif len(notes) > 1:
                notes.sort(key=lambda n: n["onset"])
                phrase_files.append((os.path.basename(xml_path), notes))
        except Exception:
            pass

    print(f"単音ファイル: {len(single_notes)} notes")
    print(f"フレーズファイル: {len(phrase_files)} files")
    total_phrase_notes = sum(len(pf[1]) for pf in phrase_files)
    print(f"フレーズ内ノート合計: {total_phrase_notes}")

    all_notes = single_notes + [n for _, notes in phrase_files for n in notes]
    return all_notes, single_notes, phrase_files


def rule1_position_preference(all_notes):
    """ルール1: ピッチ帯域別の弦・フレット選好"""
    print(f"\n{'='*70}")
    print(f"  ルール1: ピッチ帯域別の弦選好")
    print(f"{'='*70}")

    # ピッチを帯域に分類
    bands = {
        "低音 (E2-B2, MIDI 40-47)": (40, 47),
        "中低音 (C3-B3, MIDI 48-59)": (48, 59),
        "中高音 (C4-B4, MIDI 60-71)": (60, 71),
        "高音 (C5+, MIDI 72+)": (72, 92),
    }

    for band_name, (lo, hi) in bands.items():
        band_notes = [n for n in all_notes if lo <= n["pitch"] <= hi]
        if not band_notes:
            continue
        print(f"\n  {band_name}: {len(band_notes)} notes")
        string_dist = Counter(n["string"] for n in band_notes)
        frets = [n["fret"] for n in band_notes]
        print(f"    弦分布: {', '.join(f'S{s}={cnt}({cnt/len(band_notes)*100:.0f}%)' for s, cnt in sorted(string_dist.items()))}")
        print(f"    フレット: 平均={statistics.mean(frets):.1f}, 中央値={statistics.median(frets):.0f}, 範囲={min(frets)}-{max(frets)}")


def rule2_string_preference_per_pitch(all_notes):
    """ルール2: 同じピッチで複数ポジション可能な場合、人間はどの弦を選ぶか"""
    print(f"\n{'='*70}")
    print(f"  ルール2: ポジション選択の人間ルール")
    print(f"{'='*70}")

    # 複数ポジション可能なピッチ（MIDI 45-64）を分析
    pitch_choices = defaultdict(list)
    for n in all_notes:
        # このピッチが複数弦で弾けるか確認
        possible = []
        for s_num in range(1, 7):
            f = n["pitch"] - STANDARD_TUNING[s_num]
            if 0 <= f <= 19:
                possible.append(s_num)
        if len(possible) > 1:
            pitch_choices[n["pitch"]].append(n)

    print(f"\n  複数ポジション可能なピッチ: {len(pitch_choices)}")

    # ルール抽出: 人間はどの弦を選ぶ傾向があるか
    print(f"\n  【発見ルール: 弦選択の傾向】")

    lowest_string_count = 0   # 最も太い弦（最低フレット）を選んだ回数
    middle_string_count = 0   # 中間弦を選んだ回数
    highest_string_count = 0  # 最も細い弦（最高フレット）を選んだ回数
    low_fret_preference = 0   # フレット0-4を選んだ回数
    mid_fret_preference = 0   # フレット5-9を選んだ回数
    high_fret_preference = 0  # フレット10+を選んだ回数

    for pitch, notes in sorted(pitch_choices.items()):
        possible = []
        for s_num in range(1, 7):
            f = pitch - STANDARD_TUNING[s_num]
            if 0 <= f <= 19:
                possible.append((s_num, f))

        for n in notes:
            # この音のポジション順位
            pos_idx = next((i for i, (s, f) in enumerate(possible) if s == n["string"]), -1)
            if pos_idx == 0:
                lowest_string_count += 1
            elif pos_idx == len(possible) - 1:
                highest_string_count += 1
            else:
                middle_string_count += 1

            if n["fret"] <= 4:
                low_fret_preference += 1
            elif n["fret"] <= 9:
                mid_fret_preference += 1
            else:
                high_fret_preference += 1

    total = lowest_string_count + middle_string_count + highest_string_count
    print(f"    最低弦（最高フレット）選択: {lowest_string_count}/{total} ({lowest_string_count/total*100:.1f}%)")
    print(f"    中間弦選択: {middle_string_count}/{total} ({middle_string_count/total*100:.1f}%)")
    print(f"    最高弦（最低フレット）選択: {highest_string_count}/{total} ({highest_string_count/total*100:.1f}%)")

    total_fret = low_fret_preference + mid_fret_preference + high_fret_preference
    print(f"\n    フレット0-4選択: {low_fret_preference}/{total_fret} ({low_fret_preference/total_fret*100:.1f}%)")
    print(f"    フレット5-9選択: {mid_fret_preference}/{total_fret} ({mid_fret_preference/total_fret*100:.1f}%)")
    print(f"    フレット10+選択: {high_fret_preference}/{total_fret} ({high_fret_preference/total_fret*100:.1f}%)")


def rule3_open_string_usage(all_notes):
    """ルール3: 開放弦の使用パターン"""
    print(f"\n{'='*70}")
    print(f"  ルール3: 開放弦の使用パターン")
    print(f"{'='*70}")

    open_notes = [n for n in all_notes if n["fret"] == 0]
    total = len(all_notes)
    print(f"  開放弦使用率: {len(open_notes)}/{total} ({len(open_notes)/total*100:.1f}%)")

    if open_notes:
        open_string_dist = Counter(n["string"] for n in open_notes)
        print(f"  開放弦の弦分布:")
        for s in range(1, 7):
            cnt = open_string_dist.get(s, 0)
            print(f"    S{s} {STRING_NAMES[s]}: {cnt} ({cnt/len(open_notes)*100:.0f}%)")

    # 開放弦で弾ける音を押弦で弾いたケース
    non_open_for_open_pitch = 0
    open_pitches = set(STANDARD_TUNING.values())
    for n in all_notes:
        if n["pitch"] in open_pitches and n["fret"] > 0:
            non_open_for_open_pitch += 1

    open_pitch_total = sum(1 for n in all_notes if n["pitch"] in open_pitches)
    if open_pitch_total > 0:
        print(f"\n  開放弦音を押弦で弾いた割合: {non_open_for_open_pitch}/{open_pitch_total} ({non_open_for_open_pitch/open_pitch_total*100:.1f}%)")
        print(f"  → 人間は開放弦音でも{non_open_for_open_pitch/open_pitch_total*100:.0f}%の確率で押弦を選ぶ")


def rule4_phrase_transitions(phrase_files):
    """ルール4: フレーズ内のポジション遷移パターン"""
    print(f"\n{'='*70}")
    print(f"  ルール4: フレーズ内ポジション遷移")
    print(f"{'='*70}")

    if not phrase_files:
        print(f"  フレーズデータなし")
        return

    fret_jumps = []
    string_jumps = []
    same_string = 0
    same_position = 0  # フレット差 <= 4
    total_transitions = 0

    for _, notes in phrase_files:
        for i in range(1, len(notes)):
            prev = notes[i-1]
            curr = notes[i]

            fret_jump = abs(curr["fret"] - prev["fret"])
            string_jump = abs(curr["string"] - prev["string"])

            fret_jumps.append(fret_jump)
            string_jumps.append(string_jump)
            total_transitions += 1

            if string_jump == 0:
                same_string += 1
            if fret_jump <= 4:
                same_position += 1

    if total_transitions > 0:
        print(f"  総遷移数: {total_transitions}")
        print(f"\n  【ルール: ポジション維持】")
        print(f"    同一弦上の移動: {same_string}/{total_transitions} ({same_string/total_transitions*100:.1f}%)")
        print(f"    4フレット以内の移動: {same_position}/{total_transitions} ({same_position/total_transitions*100:.1f}%)")
        print(f"\n  フレットジャンプ統計:")
        print(f"    平均: {statistics.mean(fret_jumps):.1f}")
        print(f"    中央値: {statistics.median(fret_jumps):.0f}")
        print(f"    最大: {max(fret_jumps)}")

        print(f"\n  弦ジャンプ統計:")
        print(f"    平均: {statistics.mean(string_jumps):.1f}")
        print(f"    中央値: {statistics.median(string_jumps):.0f}")

        # フレットジャンプのヒストグラム
        print(f"\n  フレットジャンプ分布:")
        fj_hist = Counter(fret_jumps)
        for j in range(0, min(max(fret_jumps)+1, 15)):
            cnt = fj_hist.get(j, 0)
            bar = "#" * min(cnt, 50)
            print(f"    {j:2d}f: {cnt:4d} {bar}")


def rule5_fret_string_correlation(all_notes):
    """ルール5: フレット番号と弦番号の相関"""
    print(f"\n{'='*70}")
    print(f"  ルール5: フレット-弦の相関（人間の指板使用マップ）")
    print(f"{'='*70}")

    # 弦ごとのフレット使用分布
    for s in range(1, 7):
        s_notes = [n for n in all_notes if n["string"] == s]
        if not s_notes:
            continue
        frets = [n["fret"] for n in s_notes]
        fret_dist = Counter(frets)
        top3 = fret_dist.most_common(3)
        top3_str = ", ".join(f"F{f}={c}" for f, c in top3)
        print(f"  S{s} {STRING_NAMES[s]}: {len(s_notes)}音, 平均F{statistics.mean(frets):.1f}, Top: {top3_str}")


def rule6_summarize_rules(all_notes, phrase_files):
    """ルール6: 発見されたルールの要約"""
    print(f"\n{'='*70}")
    print(f"  ★ 人間運指ルール要約 ★")
    print(f"{'='*70}")

    # フレット分布
    frets = [n["fret"] for n in all_notes]
    low = sum(1 for f in frets if f <= 4)
    mid = sum(1 for f in frets if 5 <= f <= 9)
    high = sum(1 for f in frets if f >= 10)
    total = len(frets)

    print(f"""
  ルール1: ローポジション優先
    F0-4: {low/total*100:.0f}%, F5-9: {mid/total*100:.0f}%, F10+: {high/total*100:.0f}%
    → 人間はフレット0-9に{(low+mid)/total*100:.0f}%集中

  ルール2: 弦選択は「低フレットの弦」を好む
    同じピッチが複数弦で弾ける場合、フレットが低くなる弦を選ぶ傾向

  ルール3: 開放弦は活用するが必須ではない
    開放弦使用率: {sum(1 for n in all_notes if n['fret']==0)/total*100:.1f}%

  ルール4: ポジション維持が最優先
    フレーズ内の連続音はポジション内（4フレット以内）で移動""")

    # 遷移統計
    if phrase_files:
        fret_jumps = []
        for _, notes in phrase_files:
            for i in range(1, len(notes)):
                fret_jumps.append(abs(notes[i]["fret"] - notes[i-1]["fret"]))
        if fret_jumps:
            within4 = sum(1 for j in fret_jumps if j <= 4)
            print(f"    4f以内の移動: {within4/len(fret_jumps)*100:.0f}%")

    print(f"""
  ルール5: 弦ごとの使用フレット帯域
    太い弦(S1-2) → 低フレット中心
    細い弦(S5-6) → 低〜中フレット中心

  ★ 核心: 人間は「最低フレット」ではなく
    「手の移動が最小になるポジション」を選ぶ
""")


if __name__ == "__main__":
    all_notes, single_notes, phrase_files = parse_all_xmls()
    rule1_position_preference(all_notes)
    rule2_string_preference_per_pitch(all_notes)
    rule3_open_string_usage(all_notes)
    rule4_phrase_transitions(phrase_files)
    rule5_fret_string_correlation(all_notes)
    rule6_summarize_rules(all_notes, phrase_files)
