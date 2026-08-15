"""
inspect_gp5_collection_fast.py
==============================
ローカルディスク上の .gp5 / .gp* コレクションを高精度かつ高速に発掘調査し、
SoloTab 教師データ（宝）としての価値を定量評価する。
"""

import os
import sys
import time
from pathlib import Path
from collections import Counter, defaultdict
import json
import random

try:
    import guitarpro
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "guitarpro"])
    import guitarpro

SEARCH_DIRS = [
    Path("D:/Music/chordlink-solotab/datasets"),
    Path("D:/Music"),
    Path("C:/Users/kotan/Downloads"),
    Path("C:/Users/kotan/Desktop"),
    Path("C:/Users/kotan/Documents"),
]

print("=== Step 1: 在庫調査 (Searching for GP Files) ===")
start_t = time.time()
all_gp_files = []

for s_dir in SEARCH_DIRS:
    if not s_dir.exists():
        continue
    print(f"Searching in {s_dir} ...")
    # walk directly for max speed
    for root, dirs, files in os.walk(s_dir):
        # skip node_modules / .git / venv / build dirs
        if "node_modules" in root or ".git" in root or "venv" in root or "dist" in root:
            continue
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in (".gp5", ".gp4", ".gp3", ".gpx", ".gp"):
                full_p = Path(root) / f
                all_gp_files.append(full_p)

# Deduplicate
all_gp_files = sorted(list(set(all_gp_files)))
gp5_only = [f for f in all_gp_files if f.suffix.lower() == ".gp5"]

total_files = len(all_gp_files)
total_gp5 = len(gp5_only)
total_bytes = sum(f.stat().st_size for f in all_gp_files)

print(f"\n[在庫調査完了] ({time.time() - start_t:.1f}s)")
print(f"総GPファイル数: {total_files} 件 (うち .gp5: {total_gp5} 件)")
print(f"総容量: {total_bytes / (1024*1024):.2f} MB ({total_bytes:,} bytes)")

ext_counts = Counter(f.suffix.lower() for f in all_gp_files)
print(f"拡張子内訳: {dict(ext_counts)}")

# フォルダ別集計
folder_counts = Counter(str(f.parent) for f in all_gp_files)
print(f"\n保存先フォルダ TOP 10:")
for fld, count in folder_counts.most_common(10):
    print(f"  - {fld}: {count} 件")

# ──────────────────────────────────────────────────────────
# Step 2: 解析可行性（サンプリングテスト）
# ──────────────────────────────────────────────────────────
print("\n=== Step 2: 解析可行性テスト (150ファイルサンプリング) ===")
# クラシック/フィンガースタイルおよび一般から150件を厳選
classical_candidates = [f for f in gp5_only if any(k in str(f).lower() for k in ["classical", "brouwer", "villa-lobos", "tarrega", "sor", "bach", "romance", "giuliani"])]
fingerstyle_candidates = [f for f in gp5_only if any(k in str(f).lower() for k in ["kotaro", "oshio", "sungha", "tommy", "acoustic", "fingerstyle"])]
other_candidates = [f for f in gp5_only if f not in classical_candidates and f not in fingerstyle_candidates]

random.seed(42)
test_sample = (
    classical_candidates[:60] +
    fingerstyle_candidates[:40] +
    random.sample(other_candidates, min(50, len(other_candidates)))
)
# サンプルが150未満なら補完
if len(test_sample) < 150 and len(gp5_only) > len(test_sample):
    remaining = [f for f in gp5_only if f not in test_sample]
    test_sample.extend(random.sample(remaining, min(150 - len(test_sample), len(remaining))))

print(f"テスト対象: {len(test_sample)} ファイル")

success_count = 0
fail_count = 0
error_types = Counter()

# ──────────────────────────────────────────────────────────
# Step 3: 内容分析（宝判定）
# ──────────────────────────────────────────────────────────
print("\n=== Step 3: 内容分析（宝判定） ===")

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

KEYWORDS_CLASSICAL = ["romance", "bach", "tarrega", "sor", "giuliani", "villa-lobos", "brouwer", "classical", "estudio", "prelude", "canon"]
KEYWORDS_FINGERSTYLE = ["kotaro", "oshio", "sungha", "tommy", "emmanuel", "depapepe", "masaaki", "kishibe", "acoustic", "fingerstyle", "solo", "sakura"]

for gp_path in test_sample:
    fname_lower = gp_path.name.lower()
    path_lower = str(gp_path).lower()

    genre = "Rock/Pop/Other"
    if any(k in path_lower for k in KEYWORDS_CLASSICAL):
        genre = "Classical"
    elif any(k in path_lower for k in KEYWORDS_FINGERSTYLE):
        genre = "Fingerstyle"
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
        tuning_name = "Standard"
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
                elif t_vals == [62, 59, 55, 50, 43, 38]:
                    tuning_name = "Open G"
                elif t_vals == [64, 59, 56, 52, 47, 40]:
                    tuning_name = "Open E"
                elif t_vals == [62, 57, 54, 50, 45, 38]:
                    tuning_name = "Open D"
                else:
                    tuning_name = "Custom Tuning"
            else:
                tuning_name = f"{len(t_vals)}-String"
        tuning_counter[tuning_name] += 1

        # トラック・小節の走査
        for track in song.tracks:
            for m in track.measures:
                # ポリフォニー判定
                if len(m.voices) >= 2:
                    v0_has_notes = any(len(b.notes) > 0 for b in m.voices[0].beats if not b.status.isRest)
                    v1_has_notes = any(len(b.notes) > 0 for b in m.voices[1].beats if not b.status.isRest)
                    if v0_has_notes and v1_has_notes:
                        file_has_polyphony = True

                for voice in m.voices:
                    for beat in voice.beats:
                        if beat.status.isRest:
                            continue
                        if hasattr(beat.duration, "tuplet") and (beat.duration.tuplet.enters != 1 or beat.duration.tuplet.times != 1):
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

        # 宝スコア算出
        finger_ratio = (file_fingered / file_notes) if file_notes > 0 else 0
        score = 0
        score += finger_ratio * 250
        if file_has_polyphony: score += 60
        if file_has_tuplet: score += 40
        if genre in ("Classical", "Fingerstyle"): score += 50
        score += min(50, sum(file_techs.values()))

        is_romance_like = ("romance" in path_lower or "tarrega" in path_lower or "sor" in path_lower or "brouwer" in path_lower or "villa-lobos" in path_lower or (file_has_polyphony and file_has_tuplet and genre in ("Classical", "Fingerstyle")))

        file_evaluations.append({
            "path": str(gp_path),
            "name": gp_path.name,
            "genre": genre,
            "tuning": tuning_name,
            "notes": file_notes,
            "fingered_notes": file_fingered,
            "finger_ratio": round(finger_ratio * 100, 1),
            "polyphony": file_has_polyphony,
            "tuplet": file_has_tuplet,
            "techs": sum(file_techs.values()),
            "score": round(score, 1),
            "is_romance_like": is_romance_like,
        })

    except Exception as e:
        fail_count += 1
        err_name = type(e).__name__
        error_types[err_name] += 1

print(f"\n[解析成功率]")
print(f"成功: {success_count} / {len(test_sample)} ({success_count/len(test_sample)*100:.1f}%)")
print(f"失敗: {fail_count} / {len(test_sample)}")
if error_types:
    print(f"エラー内訳: {dict(error_types)}")

# 宝判定
treasure_count = sum(1 for e in file_evaluations if e["score"] >= 100 or (e["polyphony"] and e["notes"] >= 50))
treasure_ratio = (treasure_count / success_count * 100) if success_count else 0
print(f"\n[宝判定 (教師データ適格)]")
print(f"宝データ数 (サンプル内): {treasure_count} / {success_count} ({treasure_ratio:.1f}%)")
print(f"推計全体宝データ数: 約 {int(total_files * (treasure_ratio / 100)):,} 件")

# Top 10 Romance-like
romance_top = sorted([e for e in file_evaluations if e["is_romance_like"]], key=lambda x: -x["score"])[:10]

report = {
    "inventory": {
        "total_files": total_files,
        "total_gp5": total_gp5,
        "total_bytes_mb": round(total_bytes / (1024*1024), 2),
        "extensions": dict(ext_counts),
        "top_folders": dict(folder_counts.most_common(10)),
    },
    "feasibility": {
        "sampled": len(test_sample),
        "success": success_count,
        "failed": fail_count,
        "success_rate": round(success_count / len(test_sample) * 100, 1),
        "error_types": dict(error_types),
    },
    "analysis": {
        "tuning_dist": dict(tuning_counter.most_common(10)),
        "genre_dist": dict(genre_counter),
        "polyphony_rate": round(polyphony_files / success_count * 100, 1) if success_count else 0,
        "tuplet_rate": round(tuplet_files / success_count * 100, 1) if success_count else 0,
        "total_notes": total_notes,
        "fingered_notes": fingered_notes,
        "finger_rate": round(fingered_notes / total_notes * 100, 1) if total_notes else 0,
        "tech_counts": dict(tech_counter),
        "treasure_rate": round(treasure_ratio, 1),
        "estimated_total_treasures": int(total_files * (treasure_ratio / 100)),
    },
    "top_10_romance": romance_top,
}

out_json = Path("backend/benchmark/gp5_treasure_report.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\n集計レポート出力: {out_json}")
