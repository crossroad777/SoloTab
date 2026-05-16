# pyre-ignore-all-errors
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
import os
import time
import sys
import numpy as np
from pathlib import Path
from typing import Callable, Optional

from beat_detector import detect_beats
from solotab_utils import _to_native, STANDARD_TUNING, TUNINGS
from tab_renderer import notes_to_tab_musicxml
from gp_renderer import notes_to_gp5



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
        report("demucs", f"[FAIL] 分離失敗、元音声を使用 ({time.time()-t0:.1f}s)")
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
            report("demucs", f"[SOLO] ソロギター検出 (guitar_ratio={guitar_ratio:.0%}) "
                   f"→ guitar+bassミックス使用（低音域保護） ({time.time()-t0:.1f}s)")
        else:
            guitar_wav = str(wav_path)
            report("demucs", f"[SOLO] ソロギター検出 (guitar_ratio={guitar_ratio:.0%}) "
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
            report("demucs", f"[BAND] バンド曲 → guitar+bass合成 ({time.time()-t0:.1f}s)")
        else:
            guitar_wav = str(guitar_path)
            report("demucs", f"[OK] ギター分離完了 ({time.time()-t0:.1f}s)")

    return guitar_wav, is_solo_guitar


def run_pipeline(session_id: str, session_dir: Path, wav_path: Path, *,
                 tuning_name: str = "standard",
                 title: Optional[str] = None,
                 progress_cb: Optional[Callable] = None,
                 skip_demucs: bool = False,
                 fast_moe: bool = True):
    def report(step: str, msg: str):
        if progress_cb:
            progress_cb(step, msg)
        try:
            print(f"[{session_id}] [{step}] {msg}")
        except UnicodeEncodeError:
            print(f"[{session_id}] [{step}] {msg.encode('ascii', 'replace').decode()}")

    print(f"DEBUG: run_pipeline started. session_id: {session_id}, session_dir: {session_dir}")
    tuning = TUNINGS.get(tuning_name, STANDARD_TUNING)
    tuning_pitches = _get_open_string_pitches(tuning)

    # --- Step 1: Beat Detection ---
    report("beats", "ビート検出中...")
    t0 = time.time()
    try:
        beat_result = detect_beats(str(wav_path))
    except Exception as e:
        report("beats", f"[FAIL] ビート検出致命的エラー: {e}")
        # フォールバック: 空の値を返して続行を試みるか、あるいは明示的に例外を投げる
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Beat detection failed: {e}. Please check if ffmpeg is installed and in PATH.")

    beats = beat_result["beats"]
    bpm = beat_result["bpm"]
    time_signature = beat_result.get("time_signature", "4/4")
    downbeats = beat_result.get("downbeats", [])
    report("beats", f"完了: {len(beats)} beats, BPM={bpm}, 拍子={time_signature} ({time.time()-t0:.1f}s)")

    # --- ビートグリッド整合性チェック ---
    # madmomがBPMとは異なる粒度のパルスを検出する場合がある
    # (例: 89 BPM報告だがビート配列は132 BPM間隔)
    # BPMから期待される4分音符間隔と、実際のビート間隔を比較し、
    # 乖離が大きい場合はBPMベースで等間隔ビートグリッドを再構築する
    if len(beats) >= 4 and bpm > 0:
        expected_interval = 60.0 / bpm
        actual_intervals = [beats[i+1] - beats[i] for i in range(min(20, len(beats)-1))]
        actual_avg = sum(actual_intervals) / len(actual_intervals)
        ratio = expected_interval / actual_avg if actual_avg > 0 else 1.0

        if ratio > 1.3 or ratio < 0.7:
            # ビート間隔がBPMと乖離 → BPMベースで再構築
            first_beat = beats[0]
            last_beat = beats[-1]
            num_true_beats = int(round((last_beat - first_beat) / expected_interval)) + 1
            new_beats = [first_beat + i * expected_interval for i in range(num_true_beats)]
            new_beats = [b for b in new_beats if b <= last_beat + expected_interval * 0.5]
            report("beats", f"ビートグリッド補正: {len(beats)}→{len(new_beats)} beats "
                   f"(検出間隔={actual_avg:.3f}s → BPM基準={expected_interval:.3f}s, ratio={ratio:.2f})")
            beats = new_beats
            downbeats = [beats[i] for i in range(0, len(beats), 3 if time_signature == "3/4" else 4)]

    with open(session_dir / "beats.json", "w", encoding="utf-8") as f:
        json.dump(_to_native({
            "beats": beats, "bpm": bpm,
            "time_signature": time_signature, "downbeats": downbeats,
        }), f, ensure_ascii=False)

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
    if skip_demucs:
        is_solo_guitar = True
        report("demucs", "スキップ（ソロギターモード）")
    else:
        try:
            guitar_wav, is_solo_guitar = _run_demucs_separation(wav_path, session_dir, report)
        except Exception as e:
            report("demucs", f"分離スキップ (元音声を使用): {e}")

    # --- Step 2: 音声前処理 (Domain Adaptation) ---
    if is_solo_guitar:
        # ソロギター: 前処理EQをスキップ（HPFがE2=82Hzを減衰、lowシェルフがさらに低音カット）
        transcription_wav = guitar_wav
        report("preprocess", f"[SOLO] ソロギター → 前処理スキップ（元音声を直接使用）")
    else:
        report("preprocess", "音声前処理中 (ラウドネス正規化・ノイズ除去)...")
        t0 = time.time()
        preprocessed_path = session_dir / "preprocessed.wav"
        try:
            from audio_preprocessor import preprocess_audio_for_transcription
            preprocess_audio_for_transcription(guitar_wav, str(preprocessed_path))
            transcription_wav = str(preprocessed_path)
            report("preprocess", f"前処理適用（メロディ強調・ベース抑制） ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"[pipeline] Preprocessing failed, using original: {e}")
            transcription_wav = guitar_wav

    # --- Step 3: Note Detection (Priority: CRNN → MoE → Basic Pitch) ---
    # CRNN: guitar_transcriber.py — ギター専用モデル（弦/フレット付き直接出力）
    # MoE: 35モデルアンサンブル（高精度だがモジュール未インストールの場合あり）
    # Basic Pitch: 汎用ピアノモデル（最終フォールバック）
    ensemble_success = False
    notes = []
    method = "none"
    model_stats = {}

    # --- 3a-0: CRNN ギター専用モデル ---
    crnn_notes_list = []
    try:
        from guitar_transcriber import transcribe_guitar, is_model_available
        if is_model_available():
            report("notes", "CRNN推論中（ギター専用モデル）...")
            t0 = time.time()
            crnn_result = transcribe_guitar(
                str(transcription_wav),
                onset_threshold=0.5,
            )
            crnn_notes_list = crnn_result.get("notes", [])
            report("notes", f"CRNN抽出完了: {len(crnn_notes_list)} notes ({time.time()-t0:.1f}s)")
    except Exception as e:
        report("notes", f"CRNN失敗: {e}")

    # --- 3a: MoEアンサンブル ---
    moe_notes_list = []
    try:
        from pure_moe_transcriber import transcribe_pure_moe
        report("notes", "MoEアンサンブル推論中...")
        t0 = time.time()
        moe_notes_list = transcribe_pure_moe(
            str(transcription_wav),
            vote_threshold=21,      # ベンチマーク最適値 (F1=0.8916, 35モデル60%合議)
            onset_threshold=0.5,    # ベンチマーク最適値
            vote_prob_threshold=0.5, # 偽陽性抑制 (0.4→0.5): G3共鳴音等の除去
            fast_mode=False,        # フルモード（利用可能なモデルを全て使用）
        )
        report("notes", f"MoE抽出完了: {len(moe_notes_list)} notes ({time.time()-t0:.1f}s)")
    except Exception as e:
        report("notes", f"MoE失敗: {e}")

    # --- 3b: Basic Pitch ---
    bp_notes_list = []
    try:
        from basic_pitch.inference import predict as bp_predict, Model as BPModel
        import basic_pitch
        # TF 2.20とbasic-pitch 0.4.0のSavedModel非互換を回避:
        # ONNXモデル(nmp.onnx)を明示的に指定してTFバイパス
        onnx_model_path = os.path.join(
            os.path.dirname(basic_pitch.__file__),
            'saved_models', 'icassp_2022', 'nmp.onnx'
        )
        if os.path.exists(onnx_model_path):
            bp_model = BPModel(onnx_model_path)
            report("notes", "Basic Pitch推論中 (ONNXバックエンド)...")
        else:
            bp_model = None
            report("notes", "Basic Pitch推論中 (デフォルトバックエンド)...")
        t0 = time.time()
        _, midi_data, _ = bp_predict(str(transcription_wav), model_or_model_path=bp_model or basic_pitch.ICASSP_2022_MODEL_PATH)
        for inst in midi_data.instruments:
            for note in inst.notes:
                bp_notes_list.append({
                    "start": float(note.start),
                    "end": float(note.end),
                    "pitch": int(note.pitch),
                })
        report("notes", f"BasicPitch抽出完了: {len(bp_notes_list)} notes ({time.time()-t0:.1f}s)")
    except Exception as e:
        report("notes", f"BasicPitch失敗（MoE単独モードに切替）: {e}")

    # --- 3c: 融合 (優先順位: CRNN > MoE融合 > MoE単独 > BasicPitch) ---
    # CRNN: ギター専用モデル、クラシックギターでより多くの音を検出 (529 vs MoE 423)
    # MoE F1=0.89 は GuitarSet (Blues/Jazz/Rock) で計測、クラシックギターでは未検証
    # 弦/フレットはCNN分類器(92%)+Viterbi DPが担当
    MATCH_ONSET_TOL = 0.10   # 100ms (緩和: 速い曲で取りこぼし防止)
    MATCH_PITCH_TOL = 1      # ±1 semitone

    if crnn_notes_list:
        # CRNN: ギター専用モデル → 最優先（弦/フレットはCNN分類器+Viterbi DPで再割り当て）
        notes = crnn_notes_list
        method = "crnn_guitar"
        model_stats = {"crnn_notes": len(notes)}
        report("notes", f"CRNNモード: {len(notes)} notes（弦/フレットはCNN分類器+ViterbiDP）")
    elif bp_notes_list and moe_notes_list:
        # MoE融合フォールバック
        fused_notes = []
        used_moe = set()
        for bp_n in bp_notes_list:
            for j, moe_n in enumerate(moe_notes_list):
                if j in used_moe:
                    continue
                onset_diff = abs(bp_n["start"] - moe_n["start"])
                pitch_diff = abs(bp_n["pitch"] - moe_n["pitch"])
                if onset_diff < MATCH_ONSET_TOL and pitch_diff <= MATCH_PITCH_TOL:
                    boosted = dict(moe_n)
                    boosted["velocity"] = min(1.0, float(moe_n.get("velocity", 0.8)) * 1.2)
                    fused_notes.append(boosted)
                    used_moe.add(j)
                    break

        moe_only_added = 0
        for j, moe_n in enumerate(moe_notes_list):
            if j not in used_moe:
                vel = float(moe_n.get("velocity", 0))
                if vel >= 0.75:
                    downgraded = dict(moe_n)
                    downgraded["velocity"] = vel * 0.85
                    fused_notes.append(downgraded)
                    moe_only_added += 1

        fused_notes.sort(key=lambda n: (n["start"], n["pitch"]))

        notes = fused_notes
        method = "fusion_bp_moe"
        model_stats = {
            "bp_notes": len(bp_notes_list),
            "moe_notes": len(moe_notes_list),
            "fused_notes": len(notes),
            "moe_only_added": moe_only_added,
        }
        report("notes", f"融合完了: BP={len(bp_notes_list)} + MoE={len(moe_notes_list)} → {len(notes)} notes (一致={len(notes)-moe_only_added}, MoE独自={moe_only_added})")
    elif moe_notes_list:
        notes = moe_notes_list
        method = "pure_moe"
        model_stats = {"ensemble_notes": len(notes)}
        report("notes", f"MoE単独モード: {len(notes)} notes")
    elif bp_notes_list:
        notes = bp_notes_list
        method = "basic_pitch"
        model_stats = {"bp_notes": len(notes)}
        report("notes", f"BasicPitch単独モード: {len(notes)} notes（フレット情報なし）")
    else:
        report("notes", "全モデル失敗: ノート検出不可")

    ensemble_success = len(notes) > 0

    with open(session_dir / "notes.json", "w", encoding="utf-8") as f:
        json.dump(_to_native({
            "notes": notes, "total_count": len(notes), "tuning": tuning_name,
            "method": method, "model_stats": model_stats,
        }), f, ensure_ascii=False, indent=2)

    # --- Step 2.35: ビートグリッド位相調整（ノート情報ベース） ---
    # 最初のベース音（最低音域）に最も近いビートをdownbeat（小節1拍目）にする
    beats_per_bar = 3 if time_signature == "3/4" else 4
    if len(notes) > 10 and len(beats) > beats_per_bar * 2:
        all_pitches = sorted(set(int(n.get("pitch", 60)) for n in notes))
        bass_threshold = all_pitches[max(1, len(all_pitches) // 10)]
        bass_onsets = sorted([float(n["start"]) for n in notes if int(n.get("pitch", 60)) <= bass_threshold])
        
        if len(bass_onsets) >= 2:
            first_bass = bass_onsets[0]
            # first_bassに最も近いビートを探す
            dists = [abs(b - first_bass) for b in beats]
            snap_idx = dists.index(min(dists))
            
            # snap_idxをdownbeat（小節頭）にする → それ以前のビートを除去
            if snap_idx > 0:
                report("beats", f"ビート位相調整: first_bass={first_bass:.3f}s, "
                       f"nearest_beat[{snap_idx}]={beats[snap_idx]:.3f}s, "
                       f"trimming {snap_idx} beats")
                beats = beats[snap_idx:]
                downbeats = [beats[i] for i in range(0, len(beats), beats_per_bar)]
                with open(session_dir / "beats.json", "w", encoding="utf-8") as f:
                    json.dump(_to_native({
                        "beats": beats, "bpm": bpm,
                        "time_signature": time_signature, "downbeats": downbeats,
                    }), f, ensure_ascii=False, indent=2)

    # --- Step 2.4: コード検出 ---
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

    # --- テクニック検出 (h/p/slide/bend) ---
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

    # --- Palm Mute / Harmonic 検出 ---
    report("technique_pm", "PM/NH検出中...")
    try:
        from technique_classifier_cnn import annotate_techniques
        t0 = time.time()
        pre_techs = {i: n.get("technique") for i, n in enumerate(notes) if n.get("technique")}
        notes = annotate_techniques(str(guitar_wav), notes, report=lambda msg: report("technique_pm", msg))
        for i, tech in pre_techs.items():
            if i < len(notes) and tech in ("h", "p", "/", "\\"):
                notes[i]["technique"] = tech
        pm_count = sum(1 for n in notes if n.get("technique") == "palm_mute")
        nh_count = sum(1 for n in notes if n.get("technique") == "harmonic")
        report("technique_pm", f"PM/NH検出完了: PM={pm_count}, NH={nh_count} ({time.time()-t0:.1f}s)")
    except Exception as e:
        report("technique_pm", f"PM/NH検出スキップ: {e}")

    # --- チューニング推定 ---
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

    # --- Step: 音楽理論解析 ---
    report("theory", "音楽理論解析中...")
    t0 = time.time()
    rhythm_info = {'subdivision': 'straight', 'triplet_ratio': 0.0}
    detected_key_sig = detected_key or "C"
    try:
        from music_theory import detect_rhythm_pattern, detect_key_signature
        rhythm_info = detect_rhythm_pattern(notes, beats)
        # MIDIベースのキー推定: オーディオベースの確信度が高い場合は上書きしない
        # オーディオキー検出 (crepe/librosa) > MIDIノート分布推定
        midi_key = detect_key_signature(notes)
        if key_confidence < 0.6 or detected_key is None:
            detected_key_sig = midi_key
        else:
            # オーディオ検出が高確信度 → そちらを優先
            # ただし相対調（EmとD等）の場合はオーディオ側を採用
            detected_key_sig = detected_key
            if midi_key != detected_key:
                print(f"[theory] キー競合: audio={detected_key}(conf={key_confidence:.2f}) vs midi={midi_key} → audio採用")
        
        # 3/4拍子のアルペジオ3連符パターン補正
        # ロマンス等: onset fraction分析では検出できないが、
        # 1拍あたり3音のパターンが支配的なら3連符と判定
        if time_signature == "3/4" and rhythm_info["subdivision"] in ("straight", "mixed"):
            import numpy as np
            beats_arr = np.array(beats)
            notes_per_beat = []
            for bi in range(min(len(beats)-1, 60)):
                bt, nbt = beats[bi], beats[bi+1]
                count = sum(1 for n in notes if bt <= float(n["start"]) < nbt)
                if count > 0:
                    notes_per_beat.append(count)
            if notes_per_beat:
                avg_npb = np.mean(notes_per_beat)
                three_ratio = sum(1 for c in notes_per_beat if c == 3) / len(notes_per_beat)
                two_or_three = sum(1 for c in notes_per_beat if 2 <= c <= 4) / len(notes_per_beat)
                print(f"[theory] 3/4 arpeggio check: avg_npb={avg_npb:.1f}, 2-4_ratio={two_or_three:.2f}, beats_checked={len(notes_per_beat)}")
                if avg_npb >= 2.0 and two_or_three >= 0.7:
                    rhythm_info["subdivision"] = "triplet"
                    rhythm_info["triplet_ratio"] = three_ratio
                    report("theory", f"3/4アルペジオ3連符パターン検出 "
                           f"(avg={avg_npb:.1f} notes/beat, 2or3-note ratio={two_or_three:.0%})")
        
        report("theory", f"音楽理論解析完了: rhythm={rhythm_info['subdivision']} "
               f"(triplet_ratio={rhythm_info.get('triplet_ratio', 0):.2f}), "
               f"key={detected_key_sig} ({time.time()-t0:.1f}s)")
    except Exception as e:
        report("theory", f"音楽理論解析スキップ: {e}")

    # --- Step: 弦/フレット最適化 (Viterbi DP) ---
    # Conformer出力のpitchは正確だが、string/fret割り当ては
    # 弦正解率63%（ベンチマーク検証済み）のため、Viterbi DPに任せる。
    # CNN弦分類器 (Val acc 92.66%) + Viterbi DP が最適な弦割り当てを行う。
    if method == "crnn_guitar":
        report("assign", f"CRNNハイブリッドモード: ピッチはCRNN, 弦/フレットはCNN分類器+Viterbi DP: {len(notes)} notes")
        # CRNN弦予測を除去し、CNN弦分類器+Viterbi DPに弦割り当てを任せる
        for n in notes:
            n.pop("string", None)
            n.pop("fret", None)
            n.pop("cnn_string_probs", None)  # CNN分類器が新たに注入する

    report("assign", "運指最適化中 (Viterbi DP)...")
    t0 = time.time()
    try:
        from string_assigner import assign_strings_dp  # type: ignore

        # カポ適用チューニング
        capo = capo_result.get("capo", 0)
        if capo > 0:
            dp_tuning = [p + capo for p in tuning]
        else:
            dp_tuning = tuning

        notes = assign_strings_dp(
            notes,
            tuning=dp_tuning,
            initial_position=initial_position,
            chords=chords,  # 音楽理論エンジン: コード情報をViterbi DPに渡す
            audio_path=str(wav_path),  # CNN弦分類器: 音声特徴量から弦推定
        )
        report("assign", f"運指最適化完了: {len(notes)} notes ({time.time()-t0:.1f}s)")
    except Exception as e:
        report("assign", f"運指最適化スキップ（元出力をそのまま使用）: {e}")

    # --- Step: LSTM運指リファインメント ---
    # 無効化: Viterbi DPのみの方がfret 12偏重が発生せず
    # TAB表示が正常になることを確認済み (2026-05-11)
    # 再有効化する場合はLSTMの学習データを見直す必要あり
    # try:
    #     from fingering_model import predict_strings as lstm_predict_strings
    #     report("assign", "LSTM運指リファインメント中...")
    #     t0_lstm = time.time()
    #     notes = lstm_predict_strings(notes, tuning=dp_tuning)
    #     report("assign", f"LSTM運指リファインメント完了 ({time.time()-t0_lstm:.1f}s)")
    # except Exception as e:
    #     report("assign", f"LSTMリファインメントスキップ: {e}")

    # --- フレットクランプ: ハイポジション過多を抑制 ---
    # Viterbiは15fまでで最適化するが、出力は12fまでにクランプ
    MAX_FRET = 12
    clamp_count = 0
    for n in notes:
        if n.get("fret", 0) > MAX_FRET:
            pitch = n.get("pitch", 60)
            best_str, best_fret = None, 99
            for s_idx, open_pitch in enumerate(tuning):
                s_num = 6 - s_idx
                f = pitch - open_pitch
                if 0 <= f <= MAX_FRET and (best_str is None or f < best_fret):
                    best_str, best_fret = s_num, f
            if best_str is not None:
                n["string"] = best_str
                n["fret"] = best_fret
                clamp_count += 1
    if clamp_count > 0:
        report("assign", f"フレットクランプ: {clamp_count}ノートを0-{MAX_FRET}に修正")

    # --- Step: 左手指番号割り当て (finger_assigner.py) ---
    try:
        from finger_assigner import assign_fingers
        t0_finger = time.time()
        notes = assign_fingers(notes)
        report("assign", f"指番号割り当て完了: {len(notes)} notes ({time.time()-t0_finger:.1f}s)")
    except Exception as e:
        report("assign", f"指番号割り当てスキップ: {e}")

    # --- 後処理1: ノート重複除去 ---
    # Pass 1: 完全重複 — 同一ピッチが短い時間窓内（<0.08秒）で重複検出される場合
    notes.sort(key=lambda n: (float(n.get("start", 0)), int(n.get("pitch", 0))))
    DEDUP_WINDOW = 0.08  # 秒
    dedup_count = 0
    i = 0
    while i < len(notes) - 1:
        n1 = notes[i]
        n2 = notes[i + 1]
        if (int(n1.get("pitch", 0)) == int(n2.get("pitch", 0)) and
            abs(float(n1.get("start", 0)) - float(n2.get("start", 0))) < DEDUP_WINDOW):
            # velocity の低い方を除去
            if float(n1.get("velocity", 0)) >= float(n2.get("velocity", 0)):
                notes.pop(i + 1)
            else:
                notes.pop(i)
            dedup_count += 1
        else:
            i += 1

    # Pass 2 は無効化: CRNNの0.2秒間隔重複はtriplet-eighth(0.224秒)と区別できないため、
    # タイムウィンドウベースの除去は正当な音を消すリスクが高い。
    # 代わりにtab_renderer.pyのtriplet再割り当てで同一位置のノートを統合する。

    if dedup_count > 0:
        report("assign", f"ノート重複除去: {dedup_count}ノート統合")

    # --- 後処理1.5: 共鳴音(sympathetic resonance)フィルタ ---
    # ギターの開放弦(E2=40,A2=45,D3=50,G3=55,B3=59,E4=64)は
    # 他の弦を弾いた時に共鳴で鳴ることがある。
    # 特にG3(55)は3弦開放で、アルペジオ中に誤検出されやすい。
    # 判定基準: 開放弦pitchのノートが、前後のノートより有意にvelocityが低い場合は共鳴音。
    OPEN_PITCHES = {40, 45, 50, 55, 59, 64}  # standard tuning open strings
    SYMPA_VEL_RATIO = 0.6   # 周囲の平均velocityの60%未満 → 共鳴音と判定
    SYMPA_WINDOW = 0.3      # 前後0.3秒のノートを参照
    sympa_removed = 0
    if len(notes) > 10:
        sympa_keep = []
        for ni, n in enumerate(notes):
            pitch = int(n.get('pitch', 0))
            vel = float(n.get('velocity', 0.5))
            if vel > 1.0:
                vel /= 127.0
            t = float(n.get('start', 0))

            if pitch not in OPEN_PITCHES:
                sympa_keep.append(n)
                continue

            # 周囲のノートのvelocity平均を計算
            neighbors = []
            for nj in range(max(0, ni - 5), min(len(notes), ni + 6)):
                if nj == ni:
                    continue
                nt = float(notes[nj].get('start', 0))
                if abs(nt - t) <= SYMPA_WINDOW:
                    nv = float(notes[nj].get('velocity', 0.5))
                    if nv > 1.0:
                        nv /= 127.0
                    neighbors.append(nv)

            if not neighbors:
                sympa_keep.append(n)
                continue

            avg_vel = sum(neighbors) / len(neighbors)
            if vel < avg_vel * SYMPA_VEL_RATIO:
                # 共鳴音と判定 → 除去
                sympa_removed += 1
            else:
                sympa_keep.append(n)

        if sympa_removed > 0:
            notes = sympa_keep
            report("assign", f"共鳴音フィルタ: {sympa_removed}ノート除去")

    # --- 後処理2: キー制約フィルタ ---
    # 無効化: キー制約フィルタはピッチ検出結果を破壊する可能性があるため無効化
    # Em楽曲がDキーで補正される問題を根本的に回避
    # MoEのピッチ検出精度が十分高い場合、スケール外の音は
    # 装飾音・経過音の可能性が高く、「最近隣補正」は適切でない
    key_fix_count = 0
    print(f"[DEBUG] key_filter: DISABLED (detected_key_sig={detected_key_sig}, key_confidence={key_confidence})", flush=True)

    # Save assigned notes
    with open(session_dir / "notes_assigned.json", "w", encoding="utf-8") as f:
        json.dump(_to_native(notes), f, ensure_ascii=False, indent=2)

    # --- Step 3: TAB譜生成 (GP5 + MusicXML) ---
    report("musicxml", "TAB譜生成中...")
    t0 = time.time()

    title = title or session_dir.name
    # GP5 binary format requires Latin-1 compatible title
    try:
        title.encode('latin-1')
    except (UnicodeEncodeError, UnicodeDecodeError):
        import re
        title = re.sub(r'[^\x20-\x7E]', '', title).strip() or session_dir.name

    # GP5生成 (AlphaTab ネイティブ形式 — メイン出力)
    try:
        gp5_bytes = notes_to_gp5(
            notes,
            beats=beats,
            bpm=bpm,
            title=title,
            tuning=tuning,
            time_signature=time_signature,
            rhythm_info=rhythm_info,
            key_signature=detected_key_sig,
            noise_gate=0.20,
        )
        gp5_path = session_dir / "tab.gp5"
        with open(gp5_path, "wb") as f:
            f.write(gp5_bytes)
        report("musicxml", f"GP5生成完了: {len(gp5_bytes)} bytes")
    except Exception as e:
        report("musicxml", f"GP5生成失敗: {e}")
        import traceback; traceback.print_exc()

    # MusicXML生成 (フォールバック + PDF用)
    xml_content, tech_map = notes_to_tab_musicxml(
        notes,
        beats=beats,
        bpm=bpm,
        title=title,
        tuning=tuning,
        chords=chords,
        time_signature=time_signature,
        rhythm_info=rhythm_info,
        key_signature=detected_key_sig,
        noise_gate=0.15,
    )

    musicxml_path = session_dir / "tab.musicxml"
    with open(musicxml_path, "w", encoding="utf-8") as f:
        f.write(xml_content)

    # テクニックマップ保存 (AlphaTab API操作用)
    tech_path = session_dir / "techniques.json"
    with open(session_dir / "techniques.json", "w") as f:
        json.dump(_to_native(tech_map), f)

    report("musicxml", f"TAB譜生成完了 ({time.time()-t0:.1f}s)")

    # --- Step: MuseScoreによる五線譜+TAB譜PDF生成 ---
    try:
        report("pdf", "五線譜+TAB譜PDF生成中 (MuseScore)...")
        t0 = time.time()
        pdf_path = session_dir / "tab.pdf"

        from musescore_renderer import MUSESCORE_EXE
        import subprocess as sp
        if not Path(MUSESCORE_EXE).exists():
            raise FileNotFoundError(f"MuseScore not found: {MUSESCORE_EXE}")
        sp.run(
            [MUSESCORE_EXE, "-o", str(pdf_path), str(gp5_path)],
            capture_output=True, text=True, timeout=120,
            encoding="utf-8", errors="replace"
        )
        if pdf_path.exists():
            report("pdf", f"PDF生成完了 ({time.time()-t0:.1f}s)")
        else:
            raise RuntimeError("PDF file was not created")
    except Exception as e:
        report("pdf", f"MuseScore PDF生成スキップ: {e}")
        # フォールバック: 旧reportlabレンダラー
        try:
            from pdf_renderer import musicxml_to_pdf
            pdf_path = session_dir / "tab.pdf"
            musicxml_to_pdf(str(musicxml_path), str(pdf_path), title=title or "Guitar TAB")
            report("pdf", f"PDF生成完了 (reportlab fallback)")
        except Exception as e2:
            report("pdf", f"PDF生成失敗: {e2}")

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
