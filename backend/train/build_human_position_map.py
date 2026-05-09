"""
build_human_position_map.py — 人間運指選好マップの構築
================================================================
IDMTの人間運指データから、ピッチごとの(弦, フレット)選好頻度を
JSONマップとして出力する。

出力: human_position_preference.json
  {
    "40": {"1_0": 50, "2_5": 3, ...},   // MIDI 40 → S1F0が50回選ばれた
    "45": {"2_0": 48, "1_5": 12, ...},
    ...
  }

Usage:
    python backend/train/build_human_position_map.py
"""
import os, glob, json, xml.etree.ElementTree as ET
from collections import defaultdict, Counter

IDMT_ROOT = r"D:\Music\datasets\IDMT-SMT-GUITAR_V2\IDMT-SMT-GUITAR_V2"
OUTPUT_PATH = r"D:\Music\nextchord-solotab\backend\human_position_preference.json"

# 標準チューニング (string_number → open pitch)
STANDARD_TUNING = {1: 40, 2: 45, 3: 50, 4: 55, 5: 59, 6: 64}


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
                if pitch > 0 and string > 0:
                    all_notes.append({"pitch": pitch, "string": string, "fret": fret})
        except Exception:
            pass

    print(f"Total notes: {len(all_notes)}")
    return all_notes


def build_preference_map(notes):
    """ピッチごとの(弦, フレット)選好頻度マップを構築"""
    # pitch → Counter{(string, fret): count}
    pitch_freq = defaultdict(Counter)
    for n in notes:
        key = f"{n['string']}_{n['fret']}"
        pitch_freq[n["pitch"]][key] += 1

    # JSON用に変換
    result = {}
    for pitch in sorted(pitch_freq.keys()):
        freq = pitch_freq[pitch]
        total = sum(freq.values())
        # 頻度を正規化して確率にする
        probs = {}
        for pos_key, count in freq.most_common():
            probs[pos_key] = round(count / total, 4)
        result[str(pitch)] = {
            "freq": dict(freq.most_common()),
            "prob": probs,
            "total": total,
        }

    return result


def main():
    notes = parse_all_xmls()
    pref_map = build_preference_map(notes)

    # 保存
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(pref_map, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to: {OUTPUT_PATH}")
    print(f"Pitches covered: {len(pref_map)}")

    # サマリー表示
    print(f"\n=== 人間ポジション選好マップ (Top examples) ===")
    for pitch_str in list(pref_map.keys())[:10]:
        data = pref_map[pitch_str]
        top3 = list(data["prob"].items())[:3]
        top_str = ", ".join([f"S{k.split('_')[0]}F{k.split('_')[1]}={v:.0%}" for k, v in top3])
        print(f"  MIDI {pitch_str} ({data['total']}件): {top_str}")

    # 人間が最も好むフレット範囲
    all_frets = [n["fret"] for n in notes]
    fret_counts = Counter(all_frets)
    print(f"\n=== 人間のフレット使用頻度 ===")
    for fret in range(0, 20):
        cnt = fret_counts.get(fret, 0)
        bar = "#" * (cnt // 20)
        print(f"  F{fret:2d}: {cnt:5d} {bar}")


if __name__ == "__main__":
    main()
