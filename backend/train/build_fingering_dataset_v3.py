"""
build_fingering_dataset_v3.py — V3: 特徴量強化版データセット
==========================================================
V2からの改善:
  1. duration（音価）を追加 — 速いパッセージ vs アルペジオの判別
  2. interval（前ノートとのピッチ差）を追加 — 跳躍 vs 順次進行の判別
  3. position_context（直近のフレット中央値）を追加 — ポジション意識
  4. GP5ファイルの増加分を反映（スクレイピング継続中）

【提供物の状態】
- 構文レベル: 未検証
- 実行結果: 未検証
"""
import os, sys, json, random, math
from pathlib import Path
from collections import Counter, defaultdict

try:
    import guitarpro
except ImportError:
    print("ERROR: pip install pyguitarpro")
    sys.exit(1)

DOWNLOAD_DIR = Path(__file__).parent.parent.parent / "gprotab_downloads"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "gp_training_data" / "v3"
STANDARD_TUNING = [64, 59, 55, 50, 45, 40]  # string 1-6 open pitches
OPEN_PITCH = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}
CONTEXT_LEN = 16
MAX_FRET = 24


def classify_tuning(tuning):
    if all(v == 0 for v in tuning):
        return "zero_tuning"
    if len(tuning) != 6:
        return "invalid"
    standard = [64, 59, 55, 50, 45, 40]
    if tuning == standard:
        return "standard"
    diff = [t - s for t, s in zip(tuning, standard)]
    if all(d == diff[0] for d in diff):
        if diff[0] == -1:
            return "half_step_down"
        return "uniform_shift"
    if tuning[4] == 43 and all(t == s for t, s in zip(tuning[:4], standard[:4])) and tuning[5] == standard[5]:
        return "drop_d"
    for t in tuning:
        if t < 20 or t > 90:
            return "invalid"
    return "custom_valid"


def extract_notes_from_gp(gp_file):
    """GP5からノート列を抽出（duration情報付き）"""
    song = guitarpro.parse(str(gp_file))
    
    # テンポ
    tempo = song.tempo.value if hasattr(song.tempo, 'value') else song.tempo
    
    all_tracks = []
    
    for track in song.tracks:
        if len(track.strings) != 6:
            continue
        
        tuning = [s.value for s in track.strings]
        cls = classify_tuning(tuning)
        if cls in ("zero_tuning", "invalid"):
            continue
        
        notes = []
        beat_time = 0.0  # 相対時間（拍単位）
        
        for measure_idx, measure in enumerate(track.measures):
            # テンポ変更の反映（簡略化: 最初のテンポを使用）
            for voice in measure.voices:
                if voice is None:
                    continue
                for beat in voice.beats:
                    if beat is None:
                        continue
                    
                    # duration を秒に変換
                    dur_value = beat.duration.value  # 1=whole, 2=half, 4=quarter, etc.
                    dur_beats = 4.0 / dur_value  # 四分音符=1拍
                    dur_seconds = dur_beats * 60.0 / tempo
                    
                    for note in beat.notes:
                        midi_pitch = tuning[note.string - 1] + note.value
                        fret = note.value
                        
                        if fret < 0 or fret > MAX_FRET:
                            continue
                        
                        notes.append({
                            "time": beat_time,
                            "pitch": midi_pitch,
                            "string": note.string,  # 1-indexed
                            "fret": fret,
                            "duration": dur_seconds,
                            "measure": measure_idx + 1,
                        })
                    
                    beat_time += dur_seconds
        
        if len(notes) >= CONTEXT_LEN + 1:
            all_tracks.append({
                "tuning": tuning,
                "notes": notes,
                "file": gp_file.stem,
            })
    
    return all_tracks


def build_samples(notes, tuning):
    """ノート列からV3特徴量付きサンプルを構築"""
    samples = []
    
    for i in range(CONTEXT_LEN, len(notes)):
        ctx = notes[i - CONTEXT_LEN:i]
        target = notes[i]
        
        # 基本特徴（V2と同じ）
        ctx_pitches = [n["pitch"] for n in ctx]
        ctx_strings = [n["string"] for n in ctx]
        ctx_frets = [min(n["fret"], MAX_FRET) for n in ctx]
        
        # V3追加特徴
        # 1. duration（秒、量子化: 0-31にクリップ）
        ctx_durations = [min(31, int(n["duration"] * 10)) for n in ctx]
        target_duration = min(31, int(target["duration"] * 10))
        
        # 2. interval（前ノートとのピッチ差、-24〜+24にクリップ→0-48にシフト）
        ctx_intervals = [0]  # 最初のノートはinterval=0
        for j in range(1, len(ctx)):
            interval = ctx[j]["pitch"] - ctx[j-1]["pitch"]
            interval = max(-24, min(24, interval)) + 24  # 0-48
            ctx_intervals.append(interval)
        target_interval = max(-24, min(24, target["pitch"] - ctx[-1]["pitch"])) + 24
        
        # 3. position_context（直近8ノートのフレット中央値）
        recent_frets = [n["fret"] for n in ctx[-8:] if n["fret"] > 0]
        pos_ctx = int(sum(recent_frets) / len(recent_frets)) if recent_frets else 0
        pos_ctx = min(24, pos_ctx)
        
        # 候補数
        candidates = sum(
            1 for s in range(1, 7) 
            if 0 <= target["pitch"] - OPEN_PITCH[s] <= MAX_FRET
        )
        
        # 人間工学チェック（直近4ノートとのストレッチ）
        recent_4 = [n for n in ctx[-4:] if n["fret"] > 0]
        ergo_ok = True
        if recent_4 and target["fret"] > 0:
            for rn in recent_4:
                if abs(target["fret"] - rn["fret"]) > 6:
                    ergo_ok = False
                    break
        
        samples.append({
            "context_pitches": ctx_pitches,
            "context_strings": ctx_strings,
            "context_frets": ctx_frets,
            "context_durations": ctx_durations,
            "context_intervals": ctx_intervals,
            "position_context": pos_ctx,
            "target_pitch": target["pitch"],
            "target_string": target["string"],
            "target_duration": target_duration,
            "target_interval": target_interval,
            "num_candidates": candidates,
            "is_ambiguous": candidates > 1,
        })
    
    return samples


def main():
    print("=== Build V3 Fingering Dataset ===")
    
    gp_files = list(DOWNLOAD_DIR.rglob("*.gp5"))
    gp_files += list(DOWNLOAD_DIR.rglob("*.gp4"))
    gp_files += list(DOWNLOAD_DIR.rglob("*.gp3"))
    print(f"GP files: {len(gp_files)}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 曲単位でシャッフル → 分割
    random.seed(42)
    random.shuffle(gp_files)
    
    all_songs = []  # [(song_id, samples)]
    stats = {
        "total_files": len(gp_files), "parsed_ok": 0, "parse_errors": 0,
        "total_tracks": 0, "total_notes": 0, "total_samples": 0,
    }
    
    for fi, gp_file in enumerate(gp_files):
        if fi % 200 == 0:
            print(f"  [{fi}/{len(gp_files)}] tracks={stats['total_tracks']}, "
                  f"samples={stats['total_samples']:,}")
        
        try:
            tracks = extract_notes_from_gp(gp_file)
        except Exception:
            stats["parse_errors"] += 1
            continue
        
        if not tracks:
            continue
        
        stats["parsed_ok"] += 1
        
        for track in tracks:
            stats["total_tracks"] += 1
            stats["total_notes"] += len(track["notes"])
            
            samples = build_samples(track["notes"], track["tuning"])
            if samples:
                song_id = f"{gp_file.stem}_{track['tuning'][0]}"
                all_songs.append((song_id, samples))
                stats["total_samples"] += len(samples)
    
    print(f"\n=== Extraction Complete ===")
    print(f"Parsed: {stats['parsed_ok']}/{stats['total_files']}")
    print(f"Tracks: {stats['total_tracks']}")
    print(f"Notes:  {stats['total_notes']:,}")
    print(f"Samples: {stats['total_samples']:,}")
    print(f"Songs (track units): {len(all_songs)}")
    
    # 曲単位で分割 (80/10/10)
    random.shuffle(all_songs)
    n = len(all_songs)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    
    splits = {
        "train": all_songs[:train_end],
        "val": all_songs[train_end:val_end],
        "test": all_songs[val_end:],
    }
    
    for split_name, songs in splits.items():
        out_file = OUTPUT_DIR / f"fingering_{split_name}.jsonl"
        count = 0
        with open(out_file, "w", encoding="utf-8") as f:
            for song_id, samples in songs:
                for s in samples:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
                    count += 1
        print(f"  {split_name}: {count:,} samples -> {out_file.name}")
    
    stats["splits"] = {k: sum(len(s) for _, s in v) for k, v in splits.items()}
    stats["num_songs"] = len(all_songs)
    
    with open(OUTPUT_DIR / "dataset_v3_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nDone! Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
