"""
analyze_gp_treasures_v4.py
==========================
各ファイルの安全な走査と高精度集計。
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os
from pathlib import Path
from collections import Counter
import json
import random
import guitarpro

TARGET_DIRS = [
    Path("D:/Music/chordlink-solotab/datasets/gprotab_downloads"),
    Path("D:/Music/chordlink-solotab/datasets/gp-classical-guitar"),
    Path("C:/Users/kotan/Downloads"),
    Path("C:/Users/kotan/Desktop"),
]

all_files = []
for d in TARGET_DIRS:
    if not d.exists():
        continue
    for p in d.rglob("*.gp*"):
        ext = p.suffix.lower()
        if ext in (".gp5", ".gp4", ".gp3", ".gpx", ".gp"):
            all_files.append(p)

all_files = sorted(list(set(all_files)))
total_files = len(all_files)
total_bytes = sum(f.stat().st_size for f in all_files)
ext_counter = Counter(f.suffix.lower() for f in all_files)
dir_counter = Counter(str(f.parent.name) for f in all_files)

classical_files = [f for f in all_files if any(k in str(f).lower() for k in ["classical", "brouwer", "villa-lobos", "tarrega", "sor", "bach", "romance", "giuliani", "estudio", "prelude"])]
fingerstyle_files = [f for f in all_files if any(k in str(f).lower() for k in ["kotaro", "oshio", "sungha", "tommy", "acoustic", "fingerstyle", "kishibe", "depapepe", "sakura"])]
other_files = [f for f in all_files if f not in classical_files and f not in fingerstyle_files]

random.seed(42)
sample = (
    classical_files[:60] +
    fingerstyle_files[:40] +
    random.sample(other_files, min(100, len(other_files)))
)

success = 0
failed = 0
errors = Counter()

tuning_counts = Counter()
genre_counts = Counter()
polyphony_count = 0
tuplet_count = 0
total_notes = 0
fingered_notes = 0
tech_counts = Counter()
evaluations = []

STD_TUNING = [64, 59, 55, 50, 45, 40]
DADGAD = [62, 57, 55, 50, 45, 38]
DROP_D = [64, 59, 55, 50, 45, 38]

for p in sample:
    p_str = str(p).lower()
    genre = "Classical" if any(k in p_str for k in ["classical", "brouwer", "villa-lobos", "tarrega", "sor", "bach", "romance", "giuliani"]) else ("Fingerstyle" if any(k in p_str for k in ["kotaro", "oshio", "sungha", "tommy", "acoustic", "fingerstyle", "sakura"]) else "Rock/Pop/Other")
    genre_counts[genre] += 1

    try:
        song = guitarpro.parse(str(p))
        
        has_poly = False
        has_tup = False
        file_n = 0
        file_fingered = 0
        file_techs = Counter()

        # チューニング
        tuning_str = "Standard"
        try:
            if song.tracks:
                t1 = song.tracks[0]
                if hasattr(t1, "strings") and t1.strings:
                    t_vals = [s.value for s in t1.strings]
                    if len(t_vals) == 6:
                        if t_vals == STD_TUNING: tuning_str = "Standard"
                        elif t_vals == DADGAD: tuning_str = "DADGAD"
                        elif t_vals == DROP_D: tuning_str = "Drop D"
                        elif t_vals == [62, 59, 55, 50, 43, 38]: tuning_str = "Open G"
                        elif t_vals == [64, 59, 56, 52, 47, 40]: tuning_str = "Open E"
                        elif t_vals == [62, 57, 54, 50, 45, 38]: tuning_str = "Open D"
                        else: tuning_str = "Custom"
                    else:
                        tuning_str = f"{len(t_vals)}-String"
        except Exception:
            pass
        tuning_counts[tuning_str] += 1

        # トラック走査
        try:
            for track in song.tracks:
                for m in track.measures:
                    # ポリフォニー判定
                    if len(m.voices) >= 2:
                        v0 = any(len(b.notes) > 0 for b in m.voices[0].beats if hasattr(b, "notes") and b.notes)
                        v1 = any(len(b.notes) > 0 for b in m.voices[1].beats if hasattr(b, "notes") and b.notes)
                        if v0 and v1:
                            has_poly = True

                    for voice in m.voices:
                        for beat in voice.beats:
                            if hasattr(beat, "duration") and hasattr(beat.duration, "tuplet") and beat.duration.tuplet:
                                tup = beat.duration.tuplet
                                if getattr(tup, "enters", 1) != 1 or getattr(tup, "times", 1) != 1:
                                    has_tup = True

                            if hasattr(beat, "notes"):
                                for note in beat.notes:
                                    file_n += 1
                                    if hasattr(note, "effect") and note.effect:
                                        eff = note.effect
                                        lh = getattr(eff, "leftHandFinger", None)
                                        rh = getattr(eff, "rightHandFinger", None)
                                        if (lh and str(lh) != "Fingering.none") or (rh and str(rh) != "Fingering.none"):
                                            file_fingered += 1

                                        if getattr(eff, "hammer", False): file_techs["hammer_pull"] += 1
                                        if getattr(eff, "slides", None): file_techs["slide"] += 1
                                        if getattr(eff, "bend", None): file_techs["bend"] += 1
                                        if getattr(eff, "harmonic", None): file_techs["harmonic"] += 1
                                        if getattr(eff, "vibrato", False): file_techs["vibrato"] += 1
                                        if getattr(eff, "ghostNote", False): file_techs["ghost_note"] += 1
        except Exception as e:
            # 走査エラーがあってもパース自体は成功
            pass

        success += 1
        if has_poly: polyphony_count += 1
        if has_tup: tuplet_count += 1
        total_notes += file_n
        fingered_notes += file_fingered
        for k, v in file_techs.items():
            tech_counts[k] += v

        # スコアリング
        f_ratio = (file_fingered / file_n) if file_n > 0 else 0
        score = f_ratio * 300 + (60 if has_poly else 0) + (40 if has_tup else 0) + (50 if genre in ("Classical", "Fingerstyle") else 0) + min(50, sum(file_techs.values()))

        is_romance_like = ("romance" in p_str or "tarrega" in p_str or "sor" in p_str or "brouwer" in p_str or "villa-lobos" in p_str or (has_poly and has_tup and genre == "Classical"))

        evaluations.append({
            "name": p.name,
            "path": str(p),
            "genre": genre,
            "tuning": tuning_str,
            "notes": file_n,
            "finger_ratio": round(f_ratio * 100, 1),
            "polyphony": has_poly,
            "tuplet": has_tup,
            "techs": sum(file_techs.values()),
            "score": round(score, 1),
            "is_romance_like": is_romance_like,
        })

    except Exception as e:
        failed += 1
        errors[type(e).__name__] += 1

# 宝判定
treasures = [e for e in evaluations if e["score"] >= 80 or (e["polyphony"] and e["notes"] >= 50)]
treasure_rate = len(treasures) / success * 100 if success else 0
est_total_treasures = int(total_files * (treasure_rate / 100))

# Romance Top 10
romance_top = sorted([e for e in evaluations if e["is_romance_like"]], key=lambda x: -x["score"])[:10]

report = {
    "inventory": {
        "total_files": total_files,
        "total_gp5": ext_counter.get(".gp5", 0),
        "total_bytes_mb": round(total_bytes / (1024*1024), 2),
        "extensions": dict(ext_counter),
        "top_folders": dict(dir_counter.most_common(10)),
    },
    "feasibility": {
        "sampled": len(sample),
        "success": success,
        "failed": failed,
        "success_rate": round(success / len(sample) * 100, 1) if sample else 0,
        "error_types": dict(errors),
    },
    "analysis": {
        "tuning_dist": dict(tuning_counts.most_common(6)),
        "genre_dist": dict(genre_counts),
        "polyphony_rate": round(polyphony_count / success * 100, 1) if success else 0,
        "tuplet_rate": round(tuplet_count / success * 100, 1) if success else 0,
        "total_notes": total_notes,
        "fingered_notes": fingered_notes,
        "finger_rate": round(fingered_notes / total_notes * 100, 1) if total_notes else 0,
        "tech_counts": dict(tech_counts),
        "treasure_rate": round(treasure_rate, 1),
        "estimated_treasures": est_total_treasures,
    },
    "top_10_romance": romance_top,
}

out_p = Path("backend/benchmark/gp5_treasure_final.json")
with open(out_p, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print("\n" + "="*60)
print(f"総GPファイル数: {total_files:,} 件 / 解析成功率: {success/len(sample)*100:.1f}% ({success}/{len(sample)})")
print(f"ポリフォニー率: {polyphony_count/success*100:.1f}% / 3連符率: {tuplet_count/success*100:.1f}%")
print(f"運指付与率: {fingered_notes/total_notes*100:.1f}% / 宝データ率: {treasure_rate:.1f}% (推計約 {est_total_treasures:,} 件)")
print("="*60)
