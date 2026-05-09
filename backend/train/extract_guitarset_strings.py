"""
extract_guitarset_strings.py — GuitarSetヘキサフォニックデータから弦/ピッチ/フレット抽出
=================================================================================
GuitarSetのJAMSファイルは、ヘキサフォニックピックアップで6弦を物理的に分離記録。
annotation[i].annotation_metadata.data_source = "0"~"5" が6弦~1弦に対応。
namespace="note_midi"にMIDIピッチ値が含まれる。

ピッチ + 弦番号 → フレット番号を逆算し、人間選好マップに追加する。

Usage:
    python backend/train/extract_guitarset_strings.py
"""
import json, glob, os
from collections import defaultdict, Counter

GUITARSET_DIR = r"D:\Music\datasets\GuitarSet\annotation"
PREF_MAP_PATH = r"D:\Music\nextchord-solotab\backend\human_position_preference.json"

# 標準チューニング: data_source index → 開放弦MIDIピッチ
# GuitarSet: data_source "0"=6弦(E2=40), "1"=5弦(A2=45), ... "5"=1弦(E4=64)
STRING_TUNING = {0: 40, 1: 45, 2: 50, 3: 55, 4: 59, 5: 64}
# data_source index → string_number (IDMT形式: 1=6弦, 6=1弦)
DS_TO_STRING = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}


def extract_all():
    jams_files = sorted(glob.glob(os.path.join(GUITARSET_DIR, "*.jams")))
    print(f"GuitarSet JAMS files: {len(jams_files)}")

    all_notes = []
    for jf in jams_files:
        with open(jf, "r") as f:
            data = json.load(f)

        for ann in data.get("annotations", []):
            ns = ann.get("namespace", "")
            if ns != "note_midi":
                continue

            ann_meta = ann.get("annotation_metadata", {})
            ds = ann_meta.get("data_source", "")
            try:
                ds_idx = int(ds)
            except (ValueError, TypeError):
                continue

            if ds_idx not in STRING_TUNING:
                continue

            string_num = DS_TO_STRING[ds_idx]  # 1-6
            open_pitch = STRING_TUNING[ds_idx]

            for obs in ann.get("data", []):
                midi_pitch = obs.get("value", 0)
                if not midi_pitch or midi_pitch <= 0:
                    continue

                pitch = round(midi_pitch)
                fret = pitch - open_pitch

                if fret < 0 or fret > 22:
                    continue

                all_notes.append({
                    "pitch": pitch,
                    "string": string_num,
                    "fret": fret,
                    "file": os.path.basename(jf),
                })

    print(f"Total notes extracted: {len(all_notes)}")
    return all_notes


def merge_with_idmt(guitarset_notes):
    """GuitarSetデータをIDMTの選好マップにマージ"""
    # 既存マップを読み込み
    if os.path.exists(PREF_MAP_PATH):
        with open(PREF_MAP_PATH, "r", encoding="utf-8") as f:
            pref_map = json.load(f)
        print(f"Existing map: {len(pref_map)} pitches")
    else:
        pref_map = {}

    # GuitarSetのデータを集計
    gs_freq = defaultdict(Counter)
    for n in guitarset_notes:
        key = f"{n['string']}_{n['fret']}"
        gs_freq[n["pitch"]][key] += 1

    # マージ
    new_pitches = 0
    updated_pitches = 0
    for pitch, freq in gs_freq.items():
        pitch_str = str(pitch)
        if pitch_str in pref_map:
            # 既存データにGuitarSetを追加
            existing_freq = pref_map[pitch_str].get("freq", {})
            for pos_key, count in freq.items():
                existing_freq[pos_key] = existing_freq.get(pos_key, 0) + count
            updated_pitches += 1
        else:
            existing_freq = dict(freq)
            new_pitches += 1

        # 確率を再計算
        total = sum(existing_freq.values())
        sorted_freq = dict(sorted(existing_freq.items(), key=lambda x: -x[1]))
        probs = {k: round(v / total, 4) for k, v in sorted_freq.items()}

        pref_map[pitch_str] = {
            "freq": sorted_freq,
            "prob": probs,
            "total": total,
        }

    # ソートして保存
    sorted_map = dict(sorted(pref_map.items(), key=lambda x: int(x[0])))

    with open(PREF_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted_map, f, indent=2, ensure_ascii=False)

    print(f"\nMerge complete:")
    print(f"  New pitches: {new_pitches}")
    print(f"  Updated pitches: {updated_pitches}")
    print(f"  Total pitches in map: {len(sorted_map)}")

    # サマリー
    total_notes = sum(d["total"] for d in sorted_map.values())
    print(f"  Total notes in map: {total_notes}")


def show_stats(notes):
    """GuitarSetの統計"""
    print(f"\n=== GuitarSet 弦データ統計 ===")
    string_counts = Counter(n["string"] for n in notes)
    for s in range(1, 7):
        cnt = string_counts.get(s, 0)
        print(f"  S{s}: {cnt:5d} notes")

    frets = [n["fret"] for n in notes]
    print(f"\n  フレット範囲: {min(frets)}-{max(frets)}")
    print(f"  フレット分布:")
    fret_counts = Counter(frets)
    for f in range(0, 20):
        cnt = fret_counts.get(f, 0)
        bar = "#" * (cnt // 20)
        print(f"    F{f:2d}: {cnt:5d} {bar}")

    pitches = [n["pitch"] for n in notes]
    print(f"\n  ピッチ範囲: MIDI {min(pitches)}-{max(pitches)}")
    print(f"  ユニークピッチ: {len(set(pitches))}")


if __name__ == "__main__":
    notes = extract_all()
    show_stats(notes)
    merge_with_idmt(notes)
