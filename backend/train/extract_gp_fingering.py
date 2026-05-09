"""
extract_gp_fingering.py — GuitarProファイルから運指データを抽出し選好マップに追加
================================================================================
任意のGuitarProファイル(.gp3/.gp4/.gp5/.gpx)からstring/fret/pitchを抽出し、
human_position_preference.jsonにマージする。

大量のGPファイルを投入すれば、人間運指パターンの精度が向上する。

Usage:
    python backend/train/extract_gp_fingering.py <gpファイルまたはディレクトリ>
    python backend/train/extract_gp_fingering.py D:\\Music\\datasets\\GOAT-Dataset-main
"""
import json, glob, os, sys
from collections import defaultdict, Counter

try:
    import guitarpro
except ImportError:
    print("ERROR: pip install PyGuitarPro")
    sys.exit(1)

PREF_MAP_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "human_position_preference.json")

# 標準チューニング（比較用）
STANDARD_TUNING_VALUES = [64, 59, 55, 50, 45, 40]  # PyGuitarPro形式: 1弦→6弦


def extract_from_gp(gp_path):
    """1つのGPファイルからノートデータを抽出"""
    try:
        song = guitarpro.parse(gp_path)
    except Exception as e:
        return []

    notes = []
    for track in song.tracks:
        if track.isPercussionTrack:
            continue

        # 6弦ギターのみ
        if len(track.strings) != 6:
            continue

        tuning = [s.value for s in track.strings]

        # 標準チューニングかチェック（半音上下まで許容）
        is_standard = all(abs(tuning[i] - STANDARD_TUNING_VALUES[i]) <= 2
                          for i in range(6))

        for measure in track.measures:
            for voice in measure.voices:
                for beat in voice.beats:
                    for note in beat.notes:
                        gp_string = note.string  # 1=1弦(E4), 6=6弦(E2)
                        fret = note.value

                        if fret < 0 or fret > 24:
                            continue

                        # IDMT/GuitarSet形式に変換: 1=6弦(E2), 6=1弦(E4)
                        idmt_string = 7 - gp_string

                        # ピッチ計算
                        open_pitch = tuning[gp_string - 1]
                        pitch = open_pitch + fret

                        if pitch < 30 or pitch > 96:
                            continue

                        # 非標準チューニングの場合は標準に変換
                        if not is_standard:
                            std_open = STANDARD_TUNING_VALUES[gp_string - 1]
                            std_fret = pitch - std_open
                            if std_fret < 0 or std_fret > 24:
                                continue
                            idmt_string_for_std = 7 - gp_string
                            fret = std_fret

                        notes.append({
                            "pitch": pitch,
                            "string": idmt_string,
                            "fret": fret,
                        })

    return notes


def find_gp_files(path):
    """指定パスからGPファイルを再帰検索"""
    if os.path.isfile(path):
        return [path]

    extensions = ["*.gp3", "*.gp4", "*.gp5", "*.gpx"]
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(path, "**", ext), recursive=True))
    return sorted(set(files))


def merge_to_preference_map(all_notes, source_name="GuitarPro"):
    """選好マップにマージ"""
    if os.path.exists(PREF_MAP_PATH):
        with open(PREF_MAP_PATH, "r", encoding="utf-8") as f:
            pref_map = json.load(f)
        existing_total = sum(d["total"] for d in pref_map.values())
        print(f"既存マップ: {len(pref_map)} pitches, {existing_total} notes")
    else:
        pref_map = {}

    # 新規データを集計
    new_freq = defaultdict(Counter)
    for n in all_notes:
        key = f"{n['string']}_{n['fret']}"
        new_freq[n["pitch"]][key] += 1

    new_pitches = 0
    updated_pitches = 0
    for pitch, freq in new_freq.items():
        pitch_str = str(pitch)
        if pitch_str in pref_map:
            existing_freq = pref_map[pitch_str].get("freq", {})
            for pos_key, count in freq.items():
                existing_freq[pos_key] = existing_freq.get(pos_key, 0) + count
            updated_pitches += 1
        else:
            existing_freq = dict(freq)
            new_pitches += 1

        total = sum(existing_freq.values())
        sorted_freq = dict(sorted(existing_freq.items(), key=lambda x: -x[1]))
        probs = {k: round(v / total, 4) for k, v in sorted_freq.items()}
        pref_map[pitch_str] = {"freq": sorted_freq, "prob": probs, "total": total}

    sorted_map = dict(sorted(pref_map.items(), key=lambda x: int(x[0])))
    with open(PREF_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted_map, f, indent=2, ensure_ascii=False)

    final_total = sum(d["total"] for d in sorted_map.values())
    print(f"\nマージ完了 ({source_name}):")
    print(f"  追加ノート: {len(all_notes)}")
    print(f"  新規ピッチ: {new_pitches}")
    print(f"  更新ピッチ: {updated_pitches}")
    print(f"  マップ合計: {len(sorted_map)} pitches, {final_total} notes")


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_gp_fingering.py <path_to_gp_files>")
        sys.exit(1)

    target = sys.argv[1]
    gp_files = find_gp_files(target)
    print(f"GPファイル検出: {len(gp_files)}")

    all_notes = []
    success = 0
    fail = 0
    for gf in gp_files:
        notes = extract_from_gp(gf)
        if notes:
            all_notes.extend(notes)
            success += 1
            print(f"  OK: {os.path.basename(gf)} -> {len(notes)} notes")
        else:
            fail += 1

    print(f"\n結果: {success} files OK, {fail} files failed")
    print(f"抽出ノート合計: {len(all_notes)}")

    if all_notes:
        merge_to_preference_map(all_notes)

        # 簡易統計
        frets = [n["fret"] for n in all_notes]
        print(f"\nフレット分布:")
        fret_counts = Counter(frets)
        for f in range(0, min(max(frets)+1, 20)):
            cnt = fret_counts.get(f, 0)
            bar = "#" * min(cnt // 5, 50)
            print(f"  F{f:2d}: {cnt:5d} {bar}")


if __name__ == "__main__":
    main()
