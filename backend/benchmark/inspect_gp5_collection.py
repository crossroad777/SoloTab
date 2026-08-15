"""
inspect_gp5_collection.py
=========================
ローカルディスク上の .gp5 ファイルを発掘調査し、
SoloTabの教師データ（宝）としての価値を定量評価するスクリプト。
"""

import os
import sys
import time
from pathlib import Path
from collections import Counter, defaultdict
import json

# Ensure guitarpro is available
try:
    import guitarpro
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "guitarpro"])
    import guitarpro

SEARCH_ROOTS = [
    Path("D:/"),
    Path("C:/Users/kotan/"),
]

print("=== Step 1: 在庫調査 (Searching for *.gp5) ===")
gp5_files = []

for root in SEARCH_ROOTS:
    if not root.exists():
        continue
    print(f"Scanning {root} ...")
    try:
        for p in root.rglob("*.gp5"):
            # Exclude recycle bin / temp dirs if any
            if "$RECYCLE.BIN" in str(p) or "System Volume Information" in str(p):
                continue
            gp5_files.append(p)
    except Exception as e:
        print(f"Error scanning {root}: {e}")

# Deduplicate
gp5_files = sorted(list(set(gp5_files)))
total_files = len(gp5_files)
total_bytes = sum(f.stat().st_size for f in gp5_files if f.is_file())

print(f"\n[在庫調査結果]")
print(f"総ファイル数: {total_files} 件")
print(f"総容量: {total_bytes / (1024*1024):.2f} MB ({total_bytes} bytes)")

# フォルダ別集計
folder_counts = Counter(f.parent for f in gp5_files)
print(f"\n保存先フォルダTOP 10:")
for fld, count in folder_counts.most_common(10):
    print(f"  - {fld}: {count} 件")

# ──────────────────────────────────────────────────────────
# Step 2 & Step 3: 解析可行性 & 内容分析（宝判定）
# ──────────────────────────────────────────────────────────
print("\n=== Step 2 & 3: 解析可行性 & 内容分析（宝判定） ===")

success_count = 0
fail_count = 0
error_types = Counter()

# 集計項目
tuning_counter = Counter()
polyphony_files = 0
tuplet_files = 0
total_notes = 0
fingered_notes = 0

tech_counter = Counter()
genre_counter = Counter()

file_evaluations = []

STD_TUNING = [64, 59, 55, 50, 45, 40]  # E4, B3, G3, D3, A2, E2
DADGAD = [62, 57, 55, 50, 45, 38]      # D4, A3, G3, D3, A2, D2
DROP_D = [64, 59, 55, 50, 45, 38]      # E4, B3, G3, D3, A2, D2

KEYWORDS_CLASSICAL = ["romance", "bach", "tarrega", "sor", "giuliani", "villa-lobos", "classical", "estudio", "prelude", "canon"]
KEYWORDS_FINGERSTYLE = ["kotaro", "oshio", "sungha", "tommy", "emmanuel", "depapepe", "masaaki", "kishibe", "acoustic", "fingerstyle", "solo"]
KEYWORDS_ROCK_POP = ["rock", "pop", "metal", "anime", "beatles", "clapton", "hendrix"]

for idx, gp_path in enumerate(gp5_files):
    fname_lower = gp_path.name.lower()
    
    # ジャンル推定
    genre = "Other"
    if any(k in fname_lower for k in KEYWORDS_CLASSICAL):
        genre = "Classical"
    elif any(k in fname_lower for k in KEYWORDS_FINGERSTYLE):
        genre = "Fingerstyle"
    elif any(k in fname_lower for k in KEYWORDS_ROCK_POP):
        genre = "Rock/Pop"
    genre_counter[genre] += 1

    try:
        song = guitarpro.parse(str(gp_path))
        success_count += 1

        file_has_polyphony = False
        file_has_tuplet = False
        file_notes = 0
        file_fingered = 0
        file_techs = Counter()

        # チューニング判定 (Track 1)
        if song.tracks:
            t1 = song.tracks[0]
            t_vals = [s.value for s in t1.strings] if t1.strings else []
            if len(t_vals) == 6:
                if t_vals == STD_TUNING:
                    tuning_name = "Standard"
                elif t_vals == DADGAD:
                    tuning_name = "DADGAD"
                elif t_vals == DROP_D:
                    tuning_name = "Drop D"
                elif t_vals[0] == 62 and t_vals[1] == 59 and t_vals[2] == 55 and t_vals[3] == 50 and t_vals[4] == 43 and t_vals[5] == 38:
                    tuning_name = "Open G"
                elif t_vals[0] == 64 and t_vals[1] == 59 and t_vals[2] == 56 and t_vals[3] == 52 and t_vals[4] == 47 and t_vals[5] == 40:
                    tuning_name = "Open E"
                elif t_vals[0] == 62 and t_vals[1] == 57 and t_vals[2] == 54 and t_vals[3] == 50 and t_vals[4] == 45 and t_vals[5] == 38:
                    tuning_name = "Open D"
                else:
                    tuning_name = f"Custom ({'-'.join(str(v) for v in t_vals)})"
            else:
                tuning_name = f"{len(t_vals)}-String"
        else:
            tuning_name = "No Tracks"

        tuning_counter[tuning_name] += 1

        # トラック・小節の走査
        for track in song.tracks:
            for m in track.measures:
                # ポリフォニー判定 (Voice 1 と Voice 2 の両方にノートが存在するか)
                if len(m.voices) >= 2:
                    v0_has_notes = any(len(b.notes) > 0 for b in m.voices[0].beats if not b.status.isRest)
                    v1_has_notes = any(len(b.notes) > 0 for b in m.voices[1].beats if not b.status.isRest)
                    if v0_has_notes and v1_has_notes:
                        file_has_polyphony = True

                for voice in m.voices:
                    for beat in voice.beats:
                        if beat.status.isRest:
                            continue
                        if beat.duration.tuplet.enters != 1 or beat.duration.tuplet.times != 1:
                            file_has_tuplet = True

                        for note in beat.notes:
                            file_notes += 1
                            # 運指データ判定
                            if note.effect.leftHandFinger != guitarpro.Fingering.none or note.effect.rightHandFinger != guitarpro.Fingering.none:
                                file_fingered += 1

                            # 奏法記号判定
                            if note.effect.hammer:
                                file_techs["hammer_pull"] += 1
                            if note.effect.slides:
                                file_techs["slide"] += 1
                            if note.effect.bend:
                                file_techs["bend"] += 1
                            if note.effect.harmonic:
                                file_techs["harmonic"] += 1
                            if note.effect.vibrato:
                                file_techs["vibrato"] += 1
                            if note.effect.ghostNote:
                                file_techs["ghost_note"] += 1

        if file_has_polyphony:
            polyphony_files += 1
        if file_has_tuplet:
            tuplet_files += 1

        total_notes += file_notes
        fingered_notes += file_fingered
        for k, v in file_techs.items():
            tech_counter[k] += v

        # 宝スコア（教師データ適格性スコア）算出
        # スコア基準:
        # - 運指率 (0~100) * 3
        # - ポリフォニーあり (+50)
        # - 3連符あり (+30)
        # - クラシック/フィンガースタイル (+40)
        # - 奏法充実度 (min(50, total_techs))
        finger_ratio = (file_fingered / file_notes) if file_notes > 0 else 0
        score = 0
        score += finger_ratio * 300
        if file_has_polyphony: score += 50
        if file_has_tuplet: score += 30
        if genre in ("Classical", "Fingerstyle"): score += 40
        score += min(50, sum(file_techs.values()))

        is_romance_like = ("romance" in fname_lower or "tarrega" in fname_lower or "sor" in fname_lower or (file_has_polyphony and file_has_tuplet and genre in ("Classical", "Fingerstyle")))

        file_evaluations.append({
            "path": str(gp_path),
            "name": gp_path.name,
            "genre": genre,
            "tuning": tuning_name,
            "notes": file_notes,
            "fingered_notes": file_fingered,
            "finger_ratio": finger_ratio,
            "polyphony": file_has_polyphony,
            "tuplet": file_has_tuplet,
            "techs": sum(file_techs.values()),
            "score": score,
            "is_romance_like": is_romance_like,
        })

    except Exception as e:
        fail_count += 1
        err_name = type(e).__name__
        error_types[err_name] += 1

print(f"\n[解析結果]")
print(f"成功: {success_count} / {total_files} ({success_count/total_files*100:.1f}%)" if total_files else "0")
print(f"失敗: {fail_count} / {total_files}")
print(f"エラー内訳: {dict(error_types)}")

# 結果保存
output_data = {
    "total_files": total_files,
    "total_bytes": total_bytes,
    "success_count": success_count,
    "fail_count": fail_count,
    "error_types": dict(error_types),
    "tuning_dist": dict(tuning_counter.most_common(15)),
    "genre_dist": dict(genre_counter),
    "polyphony_rate": (polyphony_files / success_count) if success_count else 0,
    "tuplet_rate": (tuplet_files / success_count) if success_count else 0,
    "total_notes": total_notes,
    "fingered_notes": fingered_notes,
    "finger_rate": (fingered_notes / total_notes) if total_notes else 0,
    "tech_counts": dict(tech_counter),
    "files": file_evaluations,
}

report_path = Path("backend/benchmark/gp5_inventory_report.json")
report_path.parent.mkdir(parents=True, exist_ok=True)
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"\nレポートJSON出力完了: {report_path}")
