"""
GP5 Fingering Data Pipeline — 統合パイプライン v1.0
====================================================
増分パース + ワンコマンド再学習

Usage:
  python pipeline_fingering.py parse          # 新規GP5ファイルのみ増分パース
  python pipeline_fingering.py train-dnn      # DNN弦分類器の再学習
  python pipeline_fingering.py train-lstm     # Context LSTM の再学習
  python pipeline_fingering.py mine           # 運指パターンマイニング
  python pipeline_fingering.py all            # 全ステップ実行
  python pipeline_fingering.py status         # 現在のデータ・モデル状態表示
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import os
import json
import time
import hashlib
from pathlib import Path
from fractions import Fraction

# === Paths ===
ROOT = Path(r"D:\Music\nextchord-solotab")
GP5_DIRS = [
    ROOT / "datasets" / "gprotab_downloads",
    ROOT / "gprotab_downloads",
]
DATA_DIR = ROOT / "backend" / "gp5_training" / "data"
MODEL_DIR = ROOT / "backend" / "gp5_training" / "models"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

NOTES_FILE = DATA_DIR / "notes_dataset.jsonl"
CHORDS_FILE = DATA_DIR / "chords_dataset.jsonl"
PARSED_INDEX = DATA_DIR / "parsed_files_index.json"
STATS_FILE = DATA_DIR / "dataset_stats.json"

STANDARD_TUNING = [64, 59, 55, 50, 45, 40]


class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Fraction):
            return float(obj)
        return super().default(obj)


# ============================================================
# Step 1: 増分パース
# ============================================================

def collect_gp_files():
    """全GP5ファイルパスを収集"""
    extensions = {'.gp3', '.gp4', '.gp5', '.gpx', '.gp'}
    files = []
    for d in GP5_DIRS:
        if not d.exists():
            continue
        for f in d.rglob("*"):
            if f.suffix.lower() in extensions and f.is_file():
                files.append(str(f))
    return sorted(files)


def load_parsed_index():
    """パース済みファイルのインデックスを読み込み"""
    if PARSED_INDEX.exists():
        try:
            return json.load(open(PARSED_INDEX, 'r', encoding='utf-8'))
        except Exception:
            pass
    return {"files": {}, "total_notes": 0, "total_chords": 0, "version": "v1.0"}


def save_parsed_index(index):
    with open(PARSED_INDEX, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


def file_hash(filepath):
    """ファイルサイズ+更新日時でファスト識別子を生成（MD5より高速）"""
    stat = os.stat(filepath)
    return f"{stat.st_size}_{int(stat.st_mtime)}"


def is_real_guitar_track(track):
    """ギタートラック判定"""
    name = (track.name or "").lower()
    n_strings = len(track.strings)
    if n_strings < 6 or n_strings > 8:
        return False
    exclude_keywords = ['drum', 'perc', 'vocal', 'vox', 'voice', 'sing',
                        'bass', 'key', 'piano', 'organ', 'synth', 'pad',
                        'string', 'violin', 'cello', 'brass', 'horn',
                        'flute', 'sax', 'trumpet']
    if any(k in name for k in exclude_keywords):
        return False
    if hasattr(track, 'channel') and track.channel:
        ch = track.channel
        if hasattr(ch, 'channel') and ch.channel == 9:
            return False
    return True


def parse_track_notes(track, song):
    """1トラックからノート・コード情報を抽出"""
    import guitarpro as gp
    notes = []
    chords = []
    tuning = [s.value for s in track.strings]
    n_strings = len(tuning)
    tempo = song.tempo
    tick_pos = 0
    ticks_per_beat = 960

    for measure_idx, measure in enumerate(track.measures):
        for voice in measure.voices:
            for beat in voice.beats:
                if beat.status != gp.BeatStatus.normal:
                    tick_pos += getattr(beat.duration, 'time', 960)
                    continue
                beat_duration_ticks = getattr(beat.duration, 'time', 960)
                beat_time_sec = tick_pos / ticks_per_beat * (60.0 / max(tempo, 1))
                beat_notes = []
                for note in beat.notes:
                    string_num = note.string
                    fret = note.value
                    if string_num < 1 or string_num > n_strings:
                        continue
                    if fret < 0 or fret > 24:
                        continue
                    open_pitch = tuning[string_num - 1]
                    pitch = open_pitch + fret
                    if pitch < 30 or pitch > 100:
                        continue
                    techniques = []
                    try:
                        eff = note.effect
                        if eff:
                            if getattr(eff, 'hammer', False): techniques.append("hammer")
                            if getattr(eff, 'slide', None): techniques.append("slide")
                            if getattr(eff, 'bend', None): techniques.append("bend")
                            if getattr(eff, 'harmonic', None): techniques.append("harmonic")
                            if getattr(eff, 'palmMute', False) or getattr(eff, 'isPalmMute', False):
                                techniques.append("palm_mute")
                    except Exception:
                        pass
                    record = {
                        "pitch": pitch, "string": string_num, "fret": fret,
                        "duration_ticks": beat_duration_ticks,
                        "time_sec": round(beat_time_sec, 4),
                        "measure": measure_idx, "tempo": tempo,
                        "tuning": tuning[:6], "n_strings": n_strings,
                        "techniques": techniques, "velocity": note.velocity,
                    }
                    notes.append(record)
                    beat_notes.append(record)
                if len(beat_notes) >= 2:
                    chords.append({
                        "notes": [{"pitch": n["pitch"], "string": n["string"], "fret": n["fret"]}
                                  for n in beat_notes],
                        "time_sec": round(beat_time_sec, 4),
                        "measure": measure_idx, "tuning": tuning[:6], "n_strings": n_strings,
                    })
                tick_pos += beat_duration_ticks
    return notes, chords


def cmd_parse():
    """増分パース: 新規/変更ファイルのみ処理してappend"""
    import guitarpro as gp

    print("=" * 60)
    print("  Step 1: 増分パース (Incremental Parse)")
    print("=" * 60)

    print("\n[1/3] GP5ファイル収集...")
    all_files = collect_gp_files()
    print(f"  全GP5ファイル: {len(all_files):,}")

    index = load_parsed_index()
    parsed = index["files"]

    # 新規・変更ファイルを特定
    new_files = []
    for fp in all_files:
        fh = file_hash(fp)
        if fp not in parsed or parsed[fp].get("hash") != fh:
            new_files.append(fp)

    print(f"  パース済み: {len(parsed):,}")
    print(f"  新規/変更: {len(new_files):,}")

    if not new_files:
        print("  新規ファイルなし。スキップ。")
        return

    print(f"\n[2/3] {len(new_files):,}ファイルをパース中...")

    # Append mode
    notes_f = open(NOTES_FILE, 'a', encoding='utf-8')
    chords_f = open(CHORDS_FILE, 'a', encoding='utf-8')

    total_new_notes = 0
    total_new_chords = 0
    success = 0
    failed = 0
    start_time = time.time()

    for i, filepath in enumerate(new_files):
        if (i + 1) % 200 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(new_files) - i) / rate / 60 if rate > 0 else 0
            print(f"  [{i+1}/{len(new_files)}] +{total_new_notes:,} notes, "
                  f"{success} ok, {failed} fail ({rate:.1f} f/s, ETA {eta:.0f}m)")

        fh = file_hash(filepath)
        try:
            song = gp.parse(filepath)
        except UnicodeDecodeError:
            song = None
            for enc in ['latin1', 'cp1252', 'utf-8']:
                try:
                    song = gp.parse(filepath, encoding=enc)
                    break
                except Exception:
                    continue
            if song is None:
                parsed[filepath] = {"hash": fh, "notes": 0, "status": "parse_error"}
                failed += 1
                continue
        except Exception:
            parsed[filepath] = {"hash": fh, "notes": 0, "status": "parse_error"}
            failed += 1
            continue

        file_notes = 0
        file_chords = 0
        for track in song.tracks:
            if not is_real_guitar_track(track):
                continue
            try:
                notes, chords = parse_track_notes(track, song)
            except Exception:
                continue
            for note in notes:
                notes_f.write(json.dumps(note, ensure_ascii=False, cls=SafeEncoder) + '\n')
                file_notes += 1
            for chord in chords:
                chords_f.write(json.dumps(chord, ensure_ascii=False, cls=SafeEncoder) + '\n')
                file_chords += 1

        total_new_notes += file_notes
        total_new_chords += file_chords
        parsed[filepath] = {"hash": fh, "notes": file_notes, "status": "ok" if file_notes > 0 else "no_notes"}

        if file_notes > 0:
            success += 1
        else:
            failed += 1

        # 100ファイルごとにインデックス保存
        if (i + 1) % 100 == 0:
            index["files"] = parsed
            index["total_notes"] = index.get("total_notes", 0) + total_new_notes
            save_parsed_index(index)

    notes_f.close()
    chords_f.close()

    index["files"] = parsed
    index["total_notes"] = sum(v.get("notes", 0) for v in parsed.values())
    index["last_parse"] = time.strftime('%Y-%m-%dT%H:%M:%S')
    save_parsed_index(index)

    elapsed = time.time() - start_time
    print(f"\n[3/3] 完了")
    print(f"  新規パース: {success} ok, {failed} fail")
    print(f"  追加ノート: +{total_new_notes:,}")
    print(f"  追加コード: +{total_new_chords:,}")
    print(f"  総ノート数: {index['total_notes']:,}")
    print(f"  所要時間: {elapsed:.0f}s")


# ============================================================
# Step 2: DNN弦分類器の学習
# ============================================================

def cmd_train_dnn():
    """DNN弦分類器の再学習（train_string_classifier_v2.py呼び出し）"""
    print("=" * 60)
    print("  Step 2: DNN弦分類器 再学習")
    print("=" * 60)

    script = ROOT / "backend" / "gp5_training" / "train_string_classifier_v2.py"
    if not script.exists():
        print(f"  ERROR: {script} not found")
        return

    print(f"  実行: {script}")
    os.system(f'python "{script}"')


# ============================================================
# Step 3: Context LSTMの学習
# ============================================================

def cmd_train_lstm():
    """Context LSTMの再学習（train_context_lstm_v2.py呼び出し）"""
    print("=" * 60)
    print("  Step 3: Context LSTM 再学習")
    print("=" * 60)

    script = ROOT / "backend" / "gp5_training" / "train_context_lstm_v2.py"
    if not script.exists():
        print(f"  ERROR: {script} not found")
        return

    print(f"  実行: {script}")
    os.system(f'python "{script}"')


# ============================================================
# Step 4: 運指パターンマイニング
# ============================================================

def cmd_mine():
    """運指パターンマイニング（mine_fingering_patterns.py呼び出し）"""
    print("=" * 60)
    print("  Step 4: 運指パターンマイニング")
    print("=" * 60)

    script = ROOT / "backend" / "gp5_training" / "mine_fingering_patterns.py"
    if not script.exists():
        print(f"  ERROR: {script} not found")
        return

    print(f"  実行: {script}")
    os.system(f'python "{script}"')


# ============================================================
# Status: 現在のデータ・モデル状態
# ============================================================

def cmd_status():
    """現在のデータ・モデル状態を表示"""
    print("=" * 60)
    print("  GP5 Fingering Pipeline - Status")
    print("=" * 60)

    # GP5ファイル数
    all_files = collect_gp_files()
    print(f"\n--- GP5ファイル ---")
    for d in GP5_DIRS:
        if d.exists():
            count = sum(1 for _ in d.rglob("*") if _.suffix.lower() in {'.gp3','.gp4','.gp5','.gpx','.gp'})
            print(f"  {d}: {count:,}")
    print(f"  合計: {len(all_files):,}")

    # パース状況
    index = load_parsed_index()
    parsed = index.get("files", {})
    ok = sum(1 for v in parsed.values() if v.get("status") == "ok")
    err = sum(1 for v in parsed.values() if v.get("status") == "parse_error")
    no_notes = sum(1 for v in parsed.values() if v.get("status") == "no_notes")
    new_count = len(all_files) - len(parsed)

    print(f"\n--- パース状況 ---")
    print(f"  パース済み: {len(parsed):,} (ok={ok}, err={err}, no_notes={no_notes})")
    print(f"  未パース: {new_count:,}")
    print(f"  総ノート数: {index.get('total_notes', 0):,}")
    print(f"  最終パース: {index.get('last_parse', 'N/A')}")

    # データファイル
    print(f"\n--- データファイル ---")
    for f in [NOTES_FILE, CHORDS_FILE]:
        if f.exists():
            size = f.stat().st_size / 1024 / 1024
            print(f"  {f.name}: {size:.1f} MB")
        else:
            print(f"  {f.name}: NOT FOUND")

    # モデル
    print(f"\n--- 学習済みモデル ---")
    for f in sorted(MODEL_DIR.glob("*.pth")):
        size = f.stat().st_size / 1024 / 1024
        mtime = time.strftime('%Y-%m-%d %H:%M', time.localtime(f.stat().st_mtime))
        print(f"  {f.name}: {size:.1f} MB ({mtime})")

    # パターン
    pf = ROOT / "gp_training_data" / "mined_fingering_patterns.json"
    if pf.exists():
        p = json.load(open(pf, 'r', encoding='utf-8'))
        meta = p.get('metadata', {})
        print(f"\n--- マイニング済みパターン ---")
        print(f"  2音ラン: {meta.get('scale_run2_patterns', '?')}")
        print(f"  3音ラン: {meta.get('scale_run3_patterns', '?')}")
        print(f"  コード: {meta.get('chord_patterns', '?')}")


# ============================================================
# Main
# ============================================================

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1].lower()

    if cmd == "parse":
        cmd_parse()
    elif cmd == "train-dnn":
        cmd_train_dnn()
    elif cmd == "train-lstm":
        cmd_train_lstm()
    elif cmd == "mine":
        cmd_mine()
    elif cmd == "status":
        cmd_status()
    elif cmd == "all":
        cmd_parse()
        cmd_train_dnn()
        cmd_train_lstm()
        cmd_mine()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
