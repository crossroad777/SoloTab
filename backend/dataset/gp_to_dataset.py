"""
gp_to_dataset.py — SoloTab-26K 3層分類キュレーション＆データ変換パイプライン
========================================================================
26,092件のGPコレクションを全走査し、
1. 3層分類 (Trash / Gold / Silver / Bronze)
2. 同一曲の複数採譜 (Variants) を分布保持
3. シンボリックデータ (meta.json + notes.json) を datasets/solotab26k/symbolic/ へ出力
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os
import re
import time
from pathlib import Path
from collections import Counter, defaultdict
import json
import guitarpro

TARGET_DIRS = [
    Path("D:/Music/chordlink-solotab/datasets/gprotab_downloads"),
    Path("D:/Music/chordlink-solotab/datasets/gp-classical-guitar"),
    Path("C:/Users/kotan/Downloads"),
    Path("C:/Users/kotan/Desktop"),
]

OUT_DIR = Path("D:/Music/chordlink-solotab/datasets/solotab26k/symbolic")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=== Step 1: 全GPファイル走査開始 ===")
t0 = time.time()
all_gp_paths = []

for d in TARGET_DIRS:
    if not d.exists():
        continue
    for p in d.rglob("*.gp*"):
        ext = p.suffix.lower()
        if ext in (".gp5", ".gp4", ".gp3", ".gpx", ".gp"):
            all_gp_paths.append(p)

all_gp_paths = sorted(list(set(all_gp_paths)))
total_files = len(all_gp_paths)
print(f"発見ファイル総数: {total_files:,} 件")

# 集計カウンタ
tier_counts = Counter()
trash_reasons = Counter()
genre_counts = Counter()
tuning_counts = Counter()
technique_counts = Counter()

variants_groups = defaultdict(list)
curated_records = []

STD_TUNING = [64, 59, 55, 50, 45, 40]
DADGAD = [62, 57, 55, 50, 45, 38]
DROP_D = [64, 59, 55, 50, 45, 38]

# 処理制限（テスト時は全件、高速処理）
print("\n=== 3層分類キュレーション処理中... ===")

for idx, gp_path in enumerate(all_gp_paths):
    fname = gp_path.name
    fname_lower = fname.lower()
    norm_song_key = re.sub(r'[-\s_0-9\(\)]+', '', os.path.splitext(fname_lower)[0])

    try:
        song = guitarpro.parse(str(gp_path))
    except Exception as e:
        tier_counts["Trash"] += 1
        trash_reasons["parse_error"] += 1
        continue

    # テンポ/拍子チェック
    tempo = getattr(song, "tempo", 120)
    if not tempo or tempo <= 0:
        tempo = 120

    if not getattr(song, "tracks", None):
        tier_counts["Trash"] += 1
        trash_reasons["no_tracks"] += 1
        continue

    # アコースティック/クラシック/ソロギタートラックの選別
    guitar_tracks = []
    for t_idx, track in enumerate(song.tracks):
        t_name = getattr(track, "name", "").lower()
        # ドラム/ベース/ボーカル除外
        if any(skip in t_name for skip in ["drum", "bass", "vocal", "piano", "organ", "synth", "key"]):
            continue
        # 6弦ギターを優先
        s_count = len(getattr(track, "strings", []))
        if s_count == 6 or s_count == 7:
            guitar_tracks.append((t_idx, track))

    if not guitar_tracks:
        # トラック名にマッチしない場合、第1トラックを採用
        guitar_tracks = [(0, song.tracks[0])]

    # 最適トラックの抽出
    best_t_idx, best_track = guitar_tracks[0]
    strings = getattr(best_track, "strings", [])
    tuning_vals = [s.value for s in strings] if strings else STD_TUNING

    # ノート情報の抽出
    notes_list = []
    has_fingering = False
    has_out_of_range_fret = False
    has_out_of_range_pitch = False
    polyphony_measures = 0
    tuplet_count = 0

    measure_count = len(getattr(best_track, "measures", []))
    current_time_ms = 0.0

    for m_idx, m in enumerate(getattr(best_track, "measures", [])):
        ts_num = 4
        ts_den = 4
        if hasattr(m, "header") and hasattr(m.header, "timeSignature"):
            ts_num = getattr(m.header.timeSignature, "numerator", 4)
            if hasattr(m.header.timeSignature, "denominator"):
                ts_den = getattr(m.header.timeSignature.denominator, "value", 4)

        bar_poly = False
        if len(getattr(m, "voices", [])) >= 2:
            v0_notes = any(len(getattr(b, "notes", [])) > 0 for b in m.voices[0].beats)
            v1_notes = any(len(getattr(b, "notes", [])) > 0 for b in m.voices[1].beats)
            if v0_notes and v1_notes:
                bar_poly = True
                polyphony_measures += 1

        for v_idx, voice in enumerate(getattr(m, "voices", [])):
            for beat in getattr(voice, "beats", []):
                beat_notes = getattr(beat, "notes", [])
                if not beat_notes:
                    continue

                is_tup = False
                dur_el = getattr(beat, "duration", None)
                if dur_el and getattr(dur_el, "tuplet", None):
                    tup = dur_el.tuplet
                    if getattr(tup, "enters", 1) != 1 or getattr(tup, "times", 1) != 1:
                        is_tup = True
                        tuplet_count += 1

                for n in beat_notes:
                    val = getattr(n, "value", 0)
                    string_num = getattr(n, "string", 1)
                    
                    # フレット・ピッチ算出
                    fret = int(val)
                    if fret > 24:
                        has_out_of_range_fret = True

                    # 弦ピッチ
                    open_p = tuning_vals[6 - string_num] if (6 - string_num) < len(tuning_vals) else 40
                    pitch = open_p + fret

                    if pitch < 38 or pitch > 92:
                        has_out_of_range_pitch = True

                    # 運指
                    lh_finger = 0
                    eff = getattr(n, "effect", None)
                    techs = []
                    if eff:
                        lh = getattr(eff, "leftHandFinger", None)
                        if lh and str(lh) != "Fingering.none":
                            has_fingering = True
                            try:
                                lh_finger = int(lh.value) if hasattr(lh, "value") else int(str(lh).split(".")[-1])
                            except Exception:
                                lh_finger = 1

                        if getattr(eff, "hammer", False): techs.append("hammer_pull"); technique_counts["hammer_pull"] += 1
                        if getattr(eff, "slides", None): techs.append("slide"); technique_counts["slide"] += 1
                        if getattr(eff, "bend", None): techs.append("bend"); technique_counts["bend"] += 1
                        if getattr(eff, "harmonic", None): techs.append("harmonic"); technique_counts["harmonic"] += 1
                        if getattr(eff, "vibrato", False): techs.append("vibrato"); technique_counts["vibrato"] += 1
                        if getattr(eff, "ghostNote", False): techs.append("ghost_note"); technique_counts["ghost_note"] += 1

                    note_obj = {
                        "bar": m_idx,
                        "voice": v_idx + 1,
                        "pitch": pitch,
                        "string": string_num,
                        "fret": fret,
                        "finger": lh_finger,
                        "is_triplet": is_tup,
                        "techniques": techs,
                    }
                    notes_list.append(note_obj)

    note_count = len(notes_list)

    # ──────────────────────────────────────────────────────────
    # 3層分類判定
    # ──────────────────────────────────────────────────────────
    if note_count < 30:
        tier = "Trash"
        trash_reasons["notes_under_30"] += 1
    elif has_out_of_range_fret:
        tier = "Trash"
        trash_reasons["fret_over_24"] += 1
    elif has_out_of_range_pitch:
        tier = "Trash"
        trash_reasons["pitch_out_of_range"] += 1
    elif has_fingering and note_count >= 100 and measure_count >= 16:
        tier = "Gold"
        weight = 1.0
    elif note_count >= 100 and measure_count >= 16:
        tier = "Silver"
        weight = 0.8
    else:
        tier = "Bronze"
        weight = 0.5

    tier_counts[tier] += 1

    if tier != "Trash":
        rec = {
            "id": f"solotab_{idx:05d}",
            "filename": fname,
            "path": str(gp_path),
            "tier": tier,
            "weight": weight,
            "notes_count": note_count,
            "measures_count": measure_count,
            "has_fingering": has_fingering,
            "polyphony_measures": polyphony_measures,
            "tuplet_notes": tuplet_count,
            "tuning": tuning_vals,
            "tempo": tempo,
            "norm_key": norm_song_key,
        }
        curated_records.append(rec)
        variants_groups[norm_song_key].append(rec["id"])

        # シンボリックデータの保存 (上位またはサンプリング)
        if idx < 500 or tier == "Gold":
            item_dir = OUT_DIR / rec["id"]
            item_dir.mkdir(parents=True, exist_ok=True)
            with open(item_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=2)
            with open(item_dir / "notes.json", "w", encoding="utf-8") as f:
                json.dump(notes_list, f, ensure_ascii=False, indent=2)

# Variants 統計
multi_variant_songs = sum(1 for k, v in variants_groups.items() if len(v) >= 2)
total_variant_tracks = sum(len(v) for k, v in variants_groups.items() if len(v) >= 2)

report_summary = {
    "total_scanned": total_files,
    "tier_distribution": dict(tier_counts),
    "trash_breakdown": dict(trash_reasons),
    "curated_total": len(curated_records),
    "gold_tier": tier_counts.get("Gold", 0),
    "silver_tier": tier_counts.get("Silver", 0),
    "bronze_tier": tier_counts.get("Bronze", 0),
    "variants": {
        "unique_song_keys": len(variants_groups),
        "multi_variant_songs": multi_variant_songs,
        "total_variant_tracks": total_variant_tracks,
    },
    "techniques_mined": dict(technique_counts),
}

rep_path = Path("backend/benchmark/solotab26k_curation_report.json")
with open(rep_path, "w", encoding="utf-8") as f:
    json.dump(report_summary, f, ensure_ascii=False, indent=2)

print("\n" + "="*60)
print("SOLOTAB-26K 3層分類キュレーション完了レポート")
print("="*60)
print(f"総走査ファイル: {total_files:,} 件")
print(f"  ・Gold   (重み1.0 / 運指・完奏): {tier_counts.get('Gold', 0):,} 件")
print(f"  ・Silver (重み0.8 / TAB完奏):    {tier_counts.get('Silver', 0):,} 件")
print(f"  ・Bronze (重み0.5 / 短尺):       {tier_counts.get('Bronze', 0):,} 件")
print(f"  ・Trash  (除外 / 演奏不可等):   {tier_counts.get('Trash', 0):,} 件")
print(f"活かすデータ総数: {len(curated_records):,} 件 (全コレクションの {len(curated_records)/total_files*100:.1f}%)")
print(f"複数採譜バリアント曲数: {multi_variant_songs:,} 曲 ({total_variant_tracks:,} トラックを分布保持)")
print("="*60)
