"""
download_gaps_audio.py — GAPS音声ダウンロードスクリプト
=====================================================
GAPSメタデータCSVからYouTube IDを抽出し、
yt-dlpで音声をダウンロードする。

Usage:
    python download_gaps_audio.py [--max N]
"""

import os
import sys
import csv
import subprocess
import argparse
from pathlib import Path
import json


# パス設定
SCRIPT_DIR = Path(__file__).parent
GAPS_DIR = SCRIPT_DIR.parent / "datasets" / "gaps" / "gaps_v1"
AUDIO_DIR = GAPS_DIR / "audio"
MIDI_DIR = GAPS_DIR / "midi"
METADATA_CSV = GAPS_DIR / "gaps_v1_metadata.csv"

# yt-dlp パス
def get_ytdlp_path():
    # venv312のyt-dlpを優先
    venv_ytdlp = Path(r"D:\Music\nextchord\venv312\Scripts\yt-dlp.exe")
    if venv_ytdlp.exists():
        return str(venv_ytdlp)
    # フォールバック
    return "yt-dlp"


def parse_metadata():
    """GAPSメタデータCSVからYouTube IDと情報を抽出"""
    tracks = []
    
    with open(METADATA_CSV, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    
    # CSVが複雑（改行含むフィールド）なので、yt_id列を正規表現で抽出
    import re
    
    # MIDIファイル一覧から曲名を取得 -> メタデータと照合
    midi_files = sorted(MIDI_DIR.glob("*.mid"))
    
    # メタデータからyt_idを抽出 (yt_id列は29列目 = index 28)
    # CSVが複雑なので、MIDIファイル名とメタデータのgpx_name列で照合する
    lines = content.split("\n")
    header = lines[0]
    
    # yt_idを含む行を見つける
    yt_ids_found = set()
    for line in lines[1:]:
        # YouTube ID パターン (11文字)
        yt_matches = re.findall(r',([a-zA-Z0-9_-]{11}),', line)
        for yt_id in yt_matches:
            # 有効なYouTube IDかチェック
            if not yt_id.startswith("www") and not yt_id.startswith("http"):
                yt_ids_found.add(yt_id)
    
    # MIDIファイルのベース名からyt_idを推測
    # GAPSのMIDIファイル名は "scorehash-fine-aligned.mid" 形式
    # メタデータのscorerhash列と照合
    print(f"MIDI files: {len(midi_files)}")
    print(f"YouTube IDs found in metadata: {len(yt_ids_found)}")
    
    # より確実: メタデータの各行からyt_idとscorehashを抽出
    for midi_file in midi_files:
        base = midi_file.stem.replace("-fine-aligned", "")
        
        # メタデータでscorehashを検索
        for line in lines[1:]:
            if f",{base}," in line:
                # この行からyt_idを取得
                yt_matches = re.findall(r',([a-zA-Z0-9_-]{11}),\d+,', line)
                if yt_matches:
                    yt_id = yt_matches[0]
                    # cropped_durationも取得
                    dur_matches = re.findall(r',(\d+),', line)
                    duration = int(dur_matches[-2]) if len(dur_matches) >= 2 else 0
                    
                    tracks.append({
                        "midi_file": str(midi_file),
                        "scorehash": base,
                        "yt_id": yt_id,
                        "duration": duration,
                    })
                    break
    
    return tracks


def download_audio(tracks, max_tracks=None, ytdlp_path="yt-dlp"):
    """YouTube IDから音声をダウンロード (webm -> WAV 22050Hz mono)"""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    
    downloaded = 0
    skipped = 0
    errors = 0
    
    if max_tracks:
        tracks = tracks[:max_tracks]
    
    total = len(tracks)
    print(f"\n{'='*50}")
    print(f"Downloading {total} tracks from YouTube")
    print(f"{'='*50}\n")
    
    for i, track in enumerate(tracks):
        yt_id = track["yt_id"]
        wav_file = AUDIO_DIR / f"{track['scorehash']}.wav"
        
        if wav_file.exists():
            skipped += 1
            continue
        
        url = f"https://www.youtube.com/watch?v={yt_id}"
        print(f"[{i+1}/{total}] {yt_id} -> {track['scorehash']}.wav")
        
        try:
            # Step 1: yt-dlpでダウンロード (any format)
            tmp_file = AUDIO_DIR / f"{track['scorehash']}_tmp"
            cmd = [
                ytdlp_path,
                "-x",                        # 音声のみ
                "-o", str(tmp_file) + ".%(ext)s",  # 拡張子は自動
                "--no-playlist",
                "--quiet",
                url
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                errors += 1
                print(f"  [FAIL] yt-dlp failed: {result.stderr[:100]}")
                continue
            
            # Step 2: ダウンロードされたファイルを見つける
            import glob
            tmp_files = glob.glob(str(tmp_file) + ".*")
            if not tmp_files:
                errors += 1
                print(f"  [FAIL] No file downloaded")
                continue
            
            dl_file = tmp_files[0]
            
            # Step 3: ffmpeg で WAV 変換 (22050Hz mono)
            try:
                import imageio_ffmpeg
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                ffmpeg_exe = "ffmpeg"
            
            conv_result = subprocess.run([
                ffmpeg_exe, "-y", "-i", dl_file,
                "-ar", "22050", "-ac", "1",  # 22050Hz mono
                "-f", "wav", str(wav_file)
            ], capture_output=True, text=True, timeout=60)
            
            # tmp削除
            if os.path.exists(dl_file):
                os.remove(dl_file)
            
            if conv_result.returncode == 0 and wav_file.exists():
                downloaded += 1
                size_mb = wav_file.stat().st_size / (1024*1024)
                print(f"  [OK] OK ({size_mb:.1f}MB)")
            else:
                errors += 1
                print(f"  [FAIL] WAV conversion failed")
            
        except subprocess.TimeoutExpired:
            errors += 1
            print(f"  [FAIL] Timeout")
        except Exception as e:
            errors += 1
            print(f"  [FAIL] Error: {e}")
    
    print(f"\n{'='*50}")
    print(f"Results: downloaded={downloaded}, skipped={skipped}, errors={errors}")
    print(f"Audio dir: {AUDIO_DIR}")
    print(f"{'='*50}")
    
    return downloaded


def main():
    parser = argparse.ArgumentParser(description="Download GAPS audio from YouTube")
    parser.add_argument("--max", type=int, default=None, help="Max tracks to download")
    parser.add_argument("--list", action="store_true", help="List tracks without downloading")
    args = parser.parse_args()
    
    print("Parsing GAPS metadata...")
    tracks = parse_metadata()
    print(f"Found {len(tracks)} tracks with YouTube IDs")
    
    if args.list:
        for t in tracks:
            print(f"  {t['yt_id']} -> {t['scorehash']} ({t['duration']}s)")
        return
    
    ytdlp = get_ytdlp_path()
    print(f"Using yt-dlp: {ytdlp}")
    
    download_audio(tracks, max_tracks=args.max, ytdlp_path=ytdlp)
    
    # 結果をJSONに保存
    manifest_path = GAPS_DIR / "audio_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(tracks, f, indent=2)
    print(f"\nManifest saved: {manifest_path}")


if __name__ == "__main__":
    main()
