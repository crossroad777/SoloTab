# pyre-ignore-all-errors
# type: ignore
from __future__ import annotations
"""
pipeline.py — SoloTab 解析パイプライン
======================================
音声ファイルを受け取り、以下のステップを順次実行する:
  1. ビート検出 (madmom)
  2. ノート検出 (アンサンブル優先 → Basic Pitch フォールバック)
  3. TAB用MusicXML生成
"""

import json
import time
import sys
from pathlib import Path
from typing import Callable, Optional
import numpy as np  # type: ignore

from beat_detector import detect_beats  # type: ignore


from solotab_utils import _to_native, STANDARD_TUNING, TUNINGS  # type: ignore
from tab_renderer import notes_to_tab_musicxml  # type: ignore


def _get_open_string_pitches(tuning: list) -> dict:
    """チューニングリストを弦番号→MIDI辞書に変換する。"""
    return {i: pitch for i, pitch in enumerate(tuning)}


def _run_demucs_separation(wav_path: Path, session_dir: Path, report) -> tuple:
    """
    Demucsによるギタートラック分離を実行する。
    
    Returns:
        (guitar_wav_path: str, is_solo_guitar: bool)
    """
    import sys, subprocess
    import soundfile as sf  # type: ignore
    import numpy as np  # type: ignore

    guitar_wav = str(wav_path)
    is_solo_guitar = False
    demucs_model = "htdemucs_6s"
    report("demucs", f"ギタートラック分離中 (Demucs {demucs_model})...")
    t0 = time.time()

    song_name = wav_path.stem
    stems_dir = session_dir / demucs_model / song_name
    guitar_path = stems_dir / "guitar.wav"

    # Demucs分離を実行（キャッシュがなければ）
    if not guitar_path.exists():
        cmd = [
            sys.executable, "-m", "demucs.separate",
            "-o", str(session_dir),
            "-n", demucs_model,
            str(wav_path)
        ]
        subprocess.run(
            cmd, check=True, capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            env={"PYTHONIOENCODING": "utf-8", **__import__('os').environ}  # type: ignore
        )
        # 出力ディレクトリを検索
        if not stems_dir.exists():
            model_dir = session_dir / demucs_model
            if model_dir.exists():
                candidates = [d for d in model_dir.iterdir() if d.is_dir()]
                if candidates:
                    stems_dir = candidates[-1]
                    guitar_path = stems_dir / "guitar.wav"

    if not guitar_path.exists():
        report("demucs", f"❌ 分離失敗、元音声を使用 ({time.time()-t0:.1f}s)")
        return guitar_wav, is_solo_guitar

    # ソロギター判定
    guitar_audio, sr_g = sf.read(str(guitar_path))
    guitar_energy = np.mean(np.abs(guitar_audio))

    # bass.wav のエネルギー
    bass_path = stems_dir / "bass.wav"
    bass_energy = 0.0
    bass_audio = None
    if bass_path.exists():
        bass_audio, _ = sf.read(str(bass_path))
        bass_energy = np.mean(np.abs(bass_audio))

    # other.wav のエネルギー
    other_path = stems_dir / "other.wav"
    other_energy = 0.0
    if other_path.exists():
        other_audio, _ = sf.read(str(other_path))
        other_energy = np.mean(np.abs(other_audio))

    # 非ギタートラック: vocals/drums/piano のみ
    non_guitar_energy = 0.0
    for track_name in ["vocals.wav", "drums.wav", "piano.wav"]:
        track_path = stems_dir / track_name
        if track_path.exists():
            track_audio, _ = sf.read(str(track_path))
            non_guitar_energy += np.mean(np.abs(track_audio))  # type: ignore

    guitar_related_energy = guitar_energy + bass_energy + other_energy  # type: ignore
    total_energy = guitar_related_energy + non_guitar_energy
    guitar_ratio = guitar_related_energy / max(total_energy, 1e-8)

    if guitar_ratio > 0.65:
        is_solo_guitar = True
        if bass_audio is not None and guitar_audio is not None and bass_energy > guitar_energy * 0.05:
            min_len = min(len(guitar_audio), len(bass_audio))
            mixed = guitar_audio[:min_len] + bass_audio[:min_len]  # type: ignore
            peak = np.max(np.abs(mixed))
            if peak > 1.0:
                mixed = mixed / peak * 0.95
            mixed_path = stems_dir / "guitar_full.wav"
            sf.write(str(mixed_path), mixed, sr_g)
            guitar_wav = str(mixed_path)
            report("demucs", f"🎸 ソロギター検出 (guitar_ratio={guitar_ratio:.0%}) "
                   f"→ guitar+bassミックス使用（低音域保護） ({time.time()-t0:.1f}s)")
        else:
            guitar_wav = str(wav_path)
            report("demucs", f"🎸 ソロギター検出 (guitar_ratio={guitar_ratio:.0%}) "
                   f"→ 元音声を使用 ({time.time()-t0:.1f}s)")
    else:
        if bass_audio is not None and guitar_audio is not None:
            min_len = min(len(guitar_audio), len(bass_audio))
            mixed = guitar_audio[:min_len] + bass_audio[:min_len]  # type: ignore
            peak = np.max(np.abs(mixed))
            if peak > 1.0:
                mixed = mixed / peak * 0.95
            mixed_path = stems_dir / "guitar_full.wav"
            sf.write(str(mixed_path), mixed, sr_g)
            guitar_wav = str(mixed_path)
            report("demucs", f"🎵 バンド曲 → guitar+bass合成 ({time.time()-t0:.1f}s)")
        else:
            guitar_wav = str(guitar_path)
            report("demucs", f"✅ ギター分離完了 ({time.time()-t0:.1f}s)")

    return guitar_wav, is_solo_guitar


def run_pipeline(session_id: str, session_dir: Path, wav_path: Path, *,
                 tuning_name: str = "standard",
                 title: Optional[str] = None,
                 progress_cb: Optional[Callable] = None):
    def report(step: str, msg: str):
        if progress_cb:
            progress_cb(step, msg)
        print(f"[{session_id}] [{step}] {msg}")

    print(f"DEBUG: run_pipeline started. session_id: {session_id}, session_dir: {session_dir}")
    tuning = TUNINGS.get(tuning_name, STANDARD_TUNING)
    tuning_pitches = _get_open_string_pitches(tuning)

    # --- Step 1: Beat Detection ---
    report("beats", "ビート検出中...")
    t0 = time.time()
    try:
        beat_result = detect_beats(str(wav_path))
    except Exception as e:
        report("beats", f"❌ ビート検出致命的エラー: {e}")
        # フォールバック: 空の値を返して続行を試みるか、あるいは明示的に例外を投げる
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Beat detection failed: {e}. Please check if ffmpeg is installed and in PATH.")

    beats = beat_result["beats"]
    bpm = beat_result["bpm"]
    time_signature = beat_result.get("time_signature", "4/4")
    downbeats = beat_result.get("downbeats", [])
    report("beats", f"完了: {len(beats)} beats, BPM={bpm}, 拍子={time_signature} ({time.time()-t0:.1f}s)")

    with open(session_dir / "beats.json", "w", encoding="utf-8") as f:
        json.dump(_to_native(beat_result), f, ensure_ascii=False)

    # --- Step 1.5: Key Detection ---
    detected_key = None
    initial_position = 0.0
    key_confidence = 0.5
    try:
        from key_analyzer import detect_key  # type: ignore
        report("key", "キー検出中...")
        t0 = time.time()
        key_result = detect_key(str(wav_path))
        detected_key = key_result["key"]
        initial_position = float(key_result["position"])
        key_confidence = key_result.get("confidence", 0.5)
        report("key", f"キー: {detected_key} (確信度: {key_confidence:.2f}), 推奨ポジション: {initial_position} ({time.time()-t0:.1f}s)")

        with open(session_dir / "key.json", "w", encoding="utf-8") as f:
            json.dump(_to_native(key_result), f, ensure_ascii=False, indent=2)
    except Exception as e:
        report("key", f"キー検出スキップ: {e}")

    # --- Step 1.6: カポ推定 ---
    capo_result = {"capo": 0, "effective_key": detected_key or "C", "confidence": 0}
    try:
        from capo_detector import detect_capo  # type: ignore
        if detected_key:
            capo_result = detect_capo(detected_key, confidence=key_confidence)
            if capo_result["capo"] > 0:
                report("capo", f"カポ推定: {capo_result['capo']}フレット (実質キー: {capo_result['effective_key']})")
            else:
                report("capo", "カポ不要")

            with open(session_dir / "capo.json", "w", encoding="utf-8") as f:
                json.dump(_to_native(capo_result), f, ensure_ascii=False, indent=2)
    except Exception as e:
        report("capo", f"カポ推定スキップ: {e}")

    # --- Step 1.7: Demucs ギタートラック分離 ---
    guitar_wav = str(wav_path)
    is_solo_guitar = False
    try:
        guitar_wav, is_solo_guitar = _run_demucs_separation(wav_path, session_dir, report)
    except Exception as e:
        report("demucs", f"分離スキップ (元音声を使用): {e}")

    # --- Step 2: 音声前処理 (Domain Adaptation) ---
    if is_solo_guitar:
        # ソロギター: 前処理EQをスキップ（HPFがE2=82Hzを減衰、lowシェルフがさらに低音カット）
        transcription_wav = guitar_wav
        report("preprocess", f"🎸 ソロギター → 前処理スキップ（元音声を直接使用）")
    else:
        report("preprocess", "音声前処理中 (ラウドネス正規化・ノイズ除去)...")
        t0 = time.time()
        preprocessed_path = session_dir / "preprocessed.wav"
        try:
            from ensemble_transcriber import preprocess_audio_for_transcription
            preprocess_audio_for_transcription(guitar_wav, str(preprocessed_path))
            transcription_wav = str(preprocessed_path)
            report("preprocess", f"前処理適用（メロディ強調・ベース抑制） ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"[pipeline] Preprocessing failed, using original: {e}")
            transcription_wav = guitar_wav

    # --- Step 3: Note Detection (純粋な MoE アンサンブル) ---
    ensemble_success = False
    notes = []
    method = "none"
    model_stats = {}

    try:
        from pure_moe_transcriber import transcribe_pure_moe
        report("notes", "純粋なMoE推論中 (6モデル統合のみ、他フィルタ排除)...")
        t0 = time.time()
        
        # 閾値を 3/6 票、確率は最大値(np.max)でピーク保持、閾値0.5で検出
        notes = transcribe_pure_moe(
            str(transcription_wav), 
            vote_threshold=3, 
            onset_threshold=0.5
        )
        
        method = "pure_moe"
        model_stats = {
            "ensemble_notes": len(notes)
        }
        
        report("notes", f"MoE抽出完了: {len(notes)} notes ({time.time()-t0:.1f}s)")

        with open(session_dir / "notes.json", "w", encoding="utf-8") as f:
            json.dump(_to_native({
                "notes": notes, "total_count": len(notes), "tuning": tuning_name,
                "method": method, "model_stats": model_stats,
            }), f, ensure_ascii=False, indent=2)
        ensemble_success = True
    except Exception as e:
        import traceback
        traceback.print_exc()
        report("notes", f"MoE失敗: {e}, Basic Pitchにフォールバック")

    # 不要な後処理（Basic Pitchフォールバック、ノイズフィルタ、DP弦割り当て、リズム量子化）は
    # Pure MoE の精度を阻害することが判明したため、V2.0 本番デフォルトとして完全に廃止しました。

    # --- Step 2.4: コード検出 (musical_filterの前に実行) ---
    chords = []
    try:
        from chord_detector import detect_chords  # type: ignore
        report("chords", "コード検出中...")
        t0 = time.time()
        chords = detect_chords(str(transcription_wav), beats=beats)
        report("chords", f"コード検出完了: {len(chords)}区間 ({time.time()-t0:.1f}s)")
        with open(session_dir / "chords.json", "w", encoding="utf-8") as f:
            json.dump(_to_native(chords), f, ensure_ascii=False, indent=2)
    except Exception as e:
        report("chords", f"コード検出スキップ: {e}")

    report("technique", "テクニック検出中 (h/p/slide/bend)...")

    try:
        from technique_detector import detect_techniques, add_techniques_to_musicxml_notes  # type: ignore
        t0 = time.time()
        notes = detect_techniques(notes, bpm=bpm)
        notes = add_techniques_to_musicxml_notes(notes)
        tech_count = sum(1 for n in notes if n.get("technique"))
        report("technique", f"テクニック検出完了: {tech_count}件 ({time.time()-t0:.1f}s)")
    except Exception as e:
        report("technique", f"テクニック検出スキップ: {e}")

    # --- Palm Mute / Harmonic 検出 (テクニック分類CNN) ---
    report("technique_pm", "PM/NH検出中...")
    try:
        from ensemble_transcriber import _annotate_techniques  # type: ignore
        t0 = time.time()
        # 既存のh/p/slideを保持し、technique=Noneのノートのみ対象
        pre_techs = {i: n.get("technique") for i, n in enumerate(notes) if n.get("technique")}
        notes = _annotate_techniques(str(guitar_wav), notes, report=lambda msg: report("technique_pm", msg))
        # h/p/slideが上書きされた場合は復元
        for i, tech in pre_techs.items():
            if i < len(notes) and tech in ("h", "p", "/", "\\"):
                notes[i]["technique"] = tech
        pm_count = sum(1 for n in notes if n.get("technique") == "palm_mute")
        nh_count = sum(1 for n in notes if n.get("technique") == "harmonic")
        report("technique_pm", f"PM/NH検出完了: PM={pm_count}, NH={nh_count} ({time.time()-t0:.1f}s)")
    except Exception as e:
        report("technique_pm", f"PM/NH検出スキップ: {e}")

    # --- Step 2.9: チューニング推定 ---
    tuning_suggestion = {"tuning": tuning_name, "confidence": 0}
    try:
        from tuning_detector import detect_tuning  # type: ignore
        tuning_suggestion = detect_tuning(notes, detected_key=detected_key)
        if tuning_suggestion["tuning"] != tuning_name:
            report("tuning_detect", f"推定チューニング: {tuning_suggestion.get('label', tuning_suggestion['tuning'])} "
                   f"(確信度: {tuning_suggestion['confidence']:.2f})")
        else:
            report("tuning_detect", f"チューニング確認: {tuning_name}")
    except Exception as e:
        report("tuning_detect", f"チューニング推定スキップ: {e}")

    # Save assigned notes
    with open(session_dir / "notes_assigned.json", "w", encoding="utf-8") as f:
        json.dump(_to_native(notes), f, ensure_ascii=False, indent=2)

    # --- Step 3: TAB MusicXML Generation ---
    report("musicxml", "TAB譜生成中...")
    t0 = time.time()

    title = title or session_dir.name
    xml_content, tech_map = notes_to_tab_musicxml(
        notes,
        beats=beats,
        bpm=bpm,
        title=title,
        tuning=tuning,
        chords=chords,
        time_signature=time_signature,
    )

    musicxml_path = session_dir / "tab.musicxml"
    with open(musicxml_path, "w", encoding="utf-8") as f:
        f.write(xml_content)

    # テクニックマップ保存 (AlphaTab API操作用)
    tech_path = session_dir / "techniques.json"
    with open(session_dir / "techniques.json", "w") as f:
        json.dump(_to_native(tech_map), f)

    report("musicxml", f"TAB譜生成完了 ({time.time()-t0:.1f}s)")

    return {
        "bpm": bpm,
        "time_signature": time_signature,
        "total_beats": len(beats),
        "total_notes": len(notes),
        "tuning": tuning_name,
        "method": method,
        "musicxml_path": str(musicxml_path),
        "key": detected_key,
        "capo": capo_result.get("capo", 0),
        "effective_key": capo_result.get("effective_key", detected_key),
        "suggested_tuning": tuning_suggestion.get("tuning", tuning_name),
        "tuning_confidence": tuning_suggestion.get("confidence", 0),
    }

if __name__ == "__main__":
    import argparse
    import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("wav_path", help="Path to input WAV file")
    parser.add_argument("--tuning", default="standard", help="Guitar tuning name")
    parser.add_argument("--session_id", help="Optional session ID")
    parser.add_argument("--ensemble", type=str, default="true", help="Use ensemble (ignored for now as it is hardcoded)")
    
    args = parser.parse_args()
    
    wav_path = Path(args.wav_path)
    if not wav_path.exists():
        print(f"Error: File not found: {wav_path}")
        sys.exit(1)
        
    session_id = args.session_id or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    session_dir = Path("uploads") / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Starting pipeline: {session_id} ---")
    try:
        result = run_pipeline(
            session_id=session_id,
            session_dir=session_dir,
            wav_path=wav_path,
            tuning_name=args.tuning
        )
        print(f"--- Pipeline completed successfully ---")
        print(f"Result: {result['total_notes']} notes, MusicXML: {result['musicxml_path']}")
    except Exception as e:
        print(f"--- Pipeline failed ---")
        import traceback
        traceback.print_exc()
        sys.exit(1)
