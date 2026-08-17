from __future__ import annotations
# pyre-ignore-all-errors
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
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
        try:
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
        except Exception as e:
            report("demucs", f"[WARN] Demucs separation failed: {e}. Using original wav as fallback.")
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
                 fast_moe: bool = True,
                 bp_onset_threshold: float = 0.8,
                 bp_minimum_note_length: float = 100.0,
                 moe_vote_threshold: int = -1,
                 moe_vote_prob_threshold: float = 0.5,
                 bp_only_threshold: float = 0.05,
                 guitar_type: str = "auto",
                 transcription_profile: str = "standard",
                 enable_technique_gp5: bool = True,
                 enable_technique_overlay: bool = True,
                 enable_technique_fingers: bool = True,
                 noise_gate: Optional[float] = None,
                 midi_path: Optional[Path] = None):
    def report(step: str, msg: str):
        if progress_cb:
            progress_cb(step, msg)
        try:
            print(f"[{session_id}] [{step}] {msg}")
        except UnicodeEncodeError:
            print(f"[{session_id}] [{step}] {msg.encode('ascii', 'replace').decode()}")

    tuning = TUNINGS.get(tuning_name, STANDARD_TUNING)
    tuning_pitches = _get_open_string_pitches(tuning)
    is_classic_profile = transcription_profile.lower() in ("classic", "arpeggio")

    # --- MIDI BYPASS: 外部MIDI入力時はAMTをスキップして直接パース ---
    is_midi_bypass = False
    if midi_path is not None and Path(midi_path).exists():
        from amt_basic_pitch import parse_midi_to_notes
        report("notes", f"MIDIバイパス有効: {Path(midi_path).name} からノートを直接取得")
        notes = parse_midi_to_notes(midi_path)
        is_midi_bypass = True

    # --- PARALLEL PHASE: Beat/Key + Note Detection ---
    # Beat detection and note detection are independent.
    # We run them concurrently to save ~15-20 seconds.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    # Shared results (thread-safe via GIL for simple assignments)
    _beat_result = {}
    _key_result_holder = [None, 0.0, 0.5]  # [detected_key, initial_position, key_confidence]
    _capo_result_holder = [{"capo": 0, "effective_key": "C", "confidence": 0}]
    _crnn_notes = []
    _moe_notes = []
    _bp_notes = []
    _guitar_type_detected = [guitar_type]  # mutable for thread
    _moe_vote_threshold = [21]

    def _do_beats_and_key():
        """Thread 1: Beat detection + Key detection + Capo estimation"""
        nonlocal beats, bpm, time_signature, downbeats
        report("beats", "ビート検出中...")
        t0 = time.time()
        try:
            result = detect_beats(str(wav_path))
        except Exception as e:
            report("beats", f"[FAIL] ビート検出致命的エラー: {e}")
            import traceback; traceback.print_exc()
            raise RuntimeError(f"Beat detection failed: {e}")
        _beat_result.update(result)
        report("beats", f"完了: {len(result['beats'])} beats, BPM={result['bpm']}, "
               f"拍子={result.get('time_signature', '4/4')} ({time.time()-t0:.1f}s)")

        # Key detection
        try:
            from key_analyzer import detect_key
            report("key", "キー検出中...")
            t0 = time.time()
            kr = detect_key(str(wav_path))
            _key_result_holder[0] = kr["key"]
            _key_result_holder[1] = float(kr["position"])
            _key_result_holder[2] = kr.get("confidence", 0.5)
            report("key", f"キー: {kr['key']} (確信度: {_key_result_holder[2]:.2f}) ({time.time()-t0:.1f}s)")
            with open(session_dir / "key.json", "w", encoding="utf-8") as f:
                json.dump(_to_native(kr), f, ensure_ascii=False, indent=2)
        except Exception as e:
            report("key", f"キー検出スキップ: {e}")

        # Capo estimation
        try:
            from capo_detector import detect_capo
            dk = _key_result_holder[0]
            if dk:
                cr = detect_capo(dk, confidence=_key_result_holder[2])
                _capo_result_holder[0] = cr
                if cr["capo"] > 0:
                    report("capo", f"カポ推定: {cr['capo']}フレット")
                else:
                    report("capo", "カポ不要")
                with open(session_dir / "capo.json", "w", encoding="utf-8") as f:
                    json.dump(_to_native(cr), f, ensure_ascii=False, indent=2)
        except Exception as e:
            report("capo", f"カポ推定スキップ: {e}")

    def _do_note_detection(transcription_wav_path: str):
        """Thread 2: CRNN + MoE + BasicPitch (sequential within thread, uses GPU)"""
        # Guitar type detection (spectral analysis)
        moe_vt = 21
        gt = guitar_type
        try:
            import librosa as _lr
            _y, _sr = _lr.load(str(wav_path), sr=22050, duration=30)
            _S = np.abs(_lr.stft(_y))
            _freqs = _lr.fft_frequencies(sr=_sr)
            _total = np.sum(_S) + 1e-10
            _hf4k = float(np.sum(_S[_freqs > 4000, :]) / _total)
            _hf6k = float(np.sum(_S[_freqs > 6000, :]) / _total)
            _bw = float(np.mean(_lr.feature.spectral_bandwidth(S=_S, sr=_sr)))
            _votes = (1 if _hf4k < 0.057 else 0) + (1 if _hf6k < 0.057 else 0) + (1 if _bw < 1386 else 0)
            is_nylon = _votes >= 2
            if gt == "nylon" or (gt == "auto" and is_nylon):
                moe_vt = 6
                gt = "nylon"
                report("notes", f"弦種: ナイロン弦 -> vote_threshold={moe_vt}/35")
            else:
                moe_vt = 6
                report("notes", f"弦種: スチール弦 -> vote_threshold={moe_vt}/35")
        except Exception as e:
            report("notes", f"弦種検出スキップ: {e}")

        env_vt = os.environ.get("SOLOTAB_MOE_VOTE_THRESHOLD")
        if env_vt is not None:
            moe_vt = int(env_vt)
        elif gt == "nylon":
            moe_vt = 6
        elif gt == "steel":
            moe_vt = 6
        _guitar_type_detected[0] = gt
        _moe_vote_threshold[0] = moe_vt

        # MoE (primary — 35モデルの合議制推論, F1=0.89)
        try:
            from pure_moe_transcriber import transcribe_pure_moe
            report("notes", "MoEアンサンブル推論中...")
            t0 = time.time()
            mn, moe_meta = transcribe_pure_moe(
                str(transcription_wav_path),
                vote_threshold=moe_vt if moe_vote_threshold == -1 else moe_vote_threshold,
                onset_threshold=bp_onset_threshold,
                vote_prob_threshold=moe_vote_prob_threshold,
                fast_mode=fast_moe,  # Respect the fast_moe setting to save memory/time
                return_metadata=True
            )
            _moe_notes.extend(mn)
            report("notes", f"MoE: {len(_moe_notes)} notes ({time.time()-t0:.1f}s)")
        except Exception as e:
            report("notes", f"MoE Error: {e}")
            import traceback; traceback.print_exc()
            mn = []
            moe_meta = {}

        # --- Style Detection & Adaptive Parameters ---
        is_classic_profile = transcription_profile.lower() in ("classic", "arpeggio")
        
        adaptive_onset = bp_onset_threshold
        adaptive_min_len = bp_minimum_note_length
        adaptive_frame_th = 0.30
        style = "NEUTRAL"
        
        pick_ratio = moe_meta.get("pick_ratio", 0.0) if moe_meta else 0.0
        finger_ratio = moe_meta.get("finger_ratio", 0.0) if moe_meta else 0.0
        
        if is_classic_profile:
            style = "CLASSIC_ARPEGGIO"
            adaptive_onset = 0.50
            adaptive_min_len = 58.0
            adaptive_frame_th = 0.30
        elif is_solo_guitar:
            style = "SOLO_GUITAR"
            adaptive_onset = 0.55
            adaptive_min_len = 70.0
            adaptive_frame_th = 0.25
        elif pick_ratio > 0.6:
            style = "STROKE"
            adaptive_onset = 0.85
            adaptive_min_len = 120.0
        elif finger_ratio > 0.6:
            style = "FINGER"
            adaptive_onset = 0.65
            adaptive_min_len = 80.0
            
        report("notes", f"[Style Profile §4.1] Profile: {transcription_profile}, Style: {style} (onset={adaptive_onset}, min_len={adaptive_min_len}ms, frame_th={adaptive_frame_th})")

        # CRNN (fallback — MoE失敗時のみ)
        if not _moe_notes:
            try:
                from guitar_transcriber import transcribe_guitar, is_model_available
                if is_model_available():
                    report("notes", "CRNN推論中 (フォールバック)...")
                    t0 = time.time()
                    cr = transcribe_guitar(str(transcription_wav_path), onset_threshold=0.5)
                    _crnn_notes.extend(cr.get("notes", []))
                    report("notes", f"CRNN: {len(_crnn_notes)} notes ({time.time()-t0:.1f}s)")
            except Exception as e:
                report("notes", f"CRNN失敗: {e}")

        # BasicPitch (MoE成功時は融合用に実行、ただしMoE単独でも十分な精度)
        try:
            from basic_pitch.inference import predict as bp_predict, Model as BPModel
            import basic_pitch
            onnx_model_path = os.path.join(
                os.path.dirname(basic_pitch.__file__),
                'saved_models', 'icassp_2022', 'nmp.onnx'
            )
            if os.path.exists(onnx_model_path):
                bp_model = BPModel(onnx_model_path)
                report("notes", "BasicPitch推論中 (ONNX)...")
            else:
                bp_model = None
                report("notes", "BasicPitch推論中...")
            t0 = time.time()
            _, midi_data, _ = bp_predict(str(transcription_wav_path),
                                          model_or_model_path=bp_model or basic_pitch.ICASSP_2022_MODEL_PATH,
                                          onset_threshold=adaptive_onset,
                                          frame_threshold=adaptive_frame_th,
                                          minimum_note_length=adaptive_min_len)
            for inst in midi_data.instruments:
                for note in inst.notes:
                    _bp_notes.append({
                        "start": float(note.start),
                        "end": float(note.end),
                        "pitch": int(note.pitch),
                        "velocity": float(note.velocity) / 127.0 if hasattr(note, "velocity") else 0.5,
                    })

            # === [TASK-940: イントロ追加パス（TASK-938設定へロールバック・凍結）] ===
            try:
                _, intro_midi, _ = bp_predict(str(transcription_wav_path),
                                              model_or_model_path=bp_model or basic_pitch.ICASSP_2022_MODEL_PATH,
                                              onset_threshold=0.20,
                                              frame_threshold=0.15,
                                              minimum_note_length=40.0)
                intro_added = 0
                for inst in intro_midi.instruments:
                    for note in inst.notes:
                        if float(note.start) < 8.0:
                            if not any(abs(n["start"] - float(note.start)) < 0.05 and n["pitch"] == int(note.pitch) for n in _bp_notes):
                                _bp_notes.append({
                                    "start": float(note.start),
                                    "end": float(note.end),
                                    "pitch": int(note.pitch),
                                    "velocity": float(note.velocity) / 127.0 if hasattr(note, "velocity") else 0.5,
                                })
                                intro_added += 1
                _bp_notes.sort(key=lambda n: n["start"])
                report("notes", f"[Intro Pass §TASK-940] イントロ追加: {intro_added}音 → 0〜8s総ノート数={len([n for n in _bp_notes if n['start'] < 8.0])}")
            except Exception as e:
                report("notes", f"イントロパススキップ: {e}")

            report("notes", f"BasicPitch: {len(_bp_notes)} notes ({time.time()-t0:.1f}s)")
        except Exception as e:
            report("notes", f"BasicPitch失敗: {e}")

    # --- Demucs (must complete before note detection) ---
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

    # Preprocessing
    if is_solo_guitar:
        transcription_wav = guitar_wav
        report("preprocess", f"[SOLO] ソロギター → 前処理スキップ")
    else:
        report("preprocess", "音声前処理中...")
        t0 = time.time()
        preprocessed_path = session_dir / "preprocessed.wav"
        try:
            from audio_preprocessor import preprocess_audio_for_transcription
            preprocess_audio_for_transcription(guitar_wav, str(preprocessed_path))
            transcription_wav = str(preprocessed_path)
            report("preprocess", f"前処理完了 ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"[pipeline] Preprocessing failed, using original: {e}")
            transcription_wav = guitar_wav

    # --- Launch parallel tasks ---
    report("beats", "並列処理開始: ビート検出 + ノート検出を同時実行...")
    t_parallel = time.time()

    with ThreadPoolExecutor(max_workers=2) as executor:
        fut_beats = executor.submit(_do_beats_and_key)
        tasks = [fut_beats]
        if not is_midi_bypass:
            fut_notes = executor.submit(_do_note_detection, transcription_wav)
            tasks.append(fut_notes)

        # Wait for both to complete
        for fut in as_completed(tasks):
            try:
                fut.result()  # raise any exceptions
            except Exception as e:
                report("parallel", f"並列タスクエラー: {e}")
                import traceback; traceback.print_exc()

    report("parallel", f"並列処理完了 ({time.time()-t_parallel:.1f}s)")

    # --- Collect results ---
    beats = _beat_result.get("beats", [])
    bpm = _beat_result.get("bpm", 120)
    time_signature = _beat_result.get("time_signature", "4/4")
    downbeats = _beat_result.get("downbeats", [])
    detected_key = _key_result_holder[0]
    initial_position = _key_result_holder[1]
    key_confidence = _key_result_holder[2]
    capo_result = _capo_result_holder[0]
    guitar_type = _guitar_type_detected[0]
    crnn_notes_list = _crnn_notes
    moe_notes_list = _moe_notes
    bp_notes_list = _bp_notes

    # --- Beat grid consistency check ---
    if is_midi_bypass and notes:
        max_note_t = max(float(n.get("end", n.get("start", 0.0) + 0.5)) for n in notes)
        if bpm <= 0:
            bpm = 88.0
        beat_interval = 60.0 / bpm
        num_beats = int(np.ceil(max_note_t / beat_interval)) + 4
        beats = [i * beat_interval for i in range(num_beats)]
        downbeats = [beats[i] for i in range(0, len(beats), 3 if time_signature == "3/4" else 4)]
        report("beats", f"MIDIバイパス用ビート生成: {len(beats)} beats (BPM={bpm})")
    elif len(beats) >= 4 and bpm > 0:
        expected_interval = 60.0 / bpm
        actual_intervals = [beats[i+1] - beats[i] for i in range(min(20, len(beats)-1))]
        actual_avg = sum(actual_intervals) / len(actual_intervals)
        ratio = expected_interval / actual_avg if actual_avg > 0 else 1.0
        if ratio > 1.3 or ratio < 0.7:
            first_beat = beats[0]
            last_beat = beats[-1]
            num_true_beats = int(round((last_beat - first_beat) / expected_interval)) + 1
            new_beats = [first_beat + i * expected_interval for i in range(num_true_beats)]
            new_beats = [b for b in new_beats if b <= last_beat + expected_interval * 0.5]
            report("beats", f"ビートグリッド補正: {len(beats)}→{len(new_beats)} beats")
            beats = new_beats
            downbeats = [beats[i] for i in range(0, len(beats), 3 if time_signature == "3/4" else 4)]

    with open(session_dir / "beats.json", "w", encoding="utf-8") as f:
        json.dump(_to_native({
            "beats": beats, "bpm": bpm,
            "time_signature": time_signature, "downbeats": downbeats,
            "rhythm_info": None,
        }), f, ensure_ascii=False)

    # --- Note fusion setup ---
    ensemble_success = False
    notes = []
    method = "none"
    model_stats = {}

    if is_midi_bypass:
        from amt_basic_pitch import parse_midi_to_notes
        notes = parse_midi_to_notes(midi_path)
        method = "midi_bypass"
        model_stats = {"midi_notes": len(notes)}
        report("notes", f"MIDIバイパス採用: {len(notes)} notes (AMTスキップ)")
    elif bp_notes_list and moe_notes_list:
        # 最優先: BPとMoEの融合 (F1=0.89)
        MATCH_ONSET_TOL = 0.10   # 100ms
        MATCH_PITCH_TOL = 1      # ±1 semitone
        fused_notes = []
        used_moe = set()
        used_bp = set()
        for i, bp_n in enumerate(bp_notes_list):
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
                    used_bp.add(i)
                    break

        # MoE独自ノート (BPに一致しなかった高確信度MoEノート)
        moe_only_min_vel = 0.50 if is_classic_profile else 0.60
        moe_only_added = 0
        for j, moe_n in enumerate(moe_notes_list):
            if j not in used_moe:
                vel = float(moe_n.get("velocity", 0))
                if vel >= moe_only_min_vel:
                    downgraded = dict(moe_n)
                    downgraded["velocity"] = vel * 0.85
                    fused_notes.append(downgraded)
                    moe_only_added += 1

        # BP独自ノート (MoEに一致しなかったBPノート)
        bp_count = len(bp_notes_list)
        moe_matched_count = len(used_bp)
        
        if bp_count > 0:
            moe_coverage = moe_matched_count / bp_count
        else:
            moe_coverage = 1.0
            
        # 論文§13.8「MoE信頼性（moe_coverage）に基づく動的閾値設定」
        if is_classic_profile or is_solo_guitar:
            # ソロギター・クラシック: 繊細なアルペジオ・弱音を拾うため 0.10 を適用
            BP_ONLY_THRESHOLD = 0.10
        elif moe_coverage < 0.40:
            # MoE信頼性極低 (超繊細な音源): BP出力をほぼ全採用
            BP_ONLY_THRESHOLD = 0.05
        elif moe_coverage < 0.70:
            # MoE信頼性中 (ソロギター/アルペジオ領域): BP弱音を積極救済
            BP_ONLY_THRESHOLD = 0.15
        else:
            # MoE信頼性高 (通常バンド/明瞭音源): 過剰検出を抑制
            BP_ONLY_THRESHOLD = 0.50
            
        report("notes", f"[Ensemble §13.8] Profile: {transcription_profile}, MoE coverage: {moe_coverage:.2%} ({moe_matched_count}/{bp_count}), Dynamic BP_ONLY threshold: {BP_ONLY_THRESHOLD}")
        bp_only_added = 0
        for i, bp_n in enumerate(bp_notes_list):
            if i not in used_bp:
                bp_vel = float(bp_n.get("velocity", 0.5))
                if bp_vel >= BP_ONLY_THRESHOLD:
                    bp_note = dict(bp_n)
                    bp_note["velocity"] = min(0.85, bp_vel * 0.9)
                    bp_note["_bp_only"] = True  # テクニック検出スキップ用フラグ
                    fused_notes.append(bp_note)
                    bp_only_added += 1


        fused_notes.sort(key=lambda n: (n["start"], n["pitch"]))

        # --- フィンガースタイル密度フィルタ (無効化: 音符の過剰な削減を防ぐため) ---
        # 以前は8分音符単位でベース1、メロディ2音に制限していましたが、アルペジオや速いピッキングの音を保護するため無効化します。
        notes = fused_notes
        method = "fusion_bp_moe"
        model_stats = {
            "bp_notes": len(bp_notes_list),
            "moe_notes": len(moe_notes_list),
            "fused_notes": len(notes),
            "moe_only_added": moe_only_added,
            "bp_only_added": bp_only_added,
        }
        report("notes", f"融合完了: BP={len(bp_notes_list)} + MoE={len(moe_notes_list)} → {len(notes)} notes "
               f"(一致={len(notes)-moe_only_added-bp_only_added}, MoE独自={moe_only_added}, BP独自={bp_only_added})")
    elif moe_notes_list:
        notes = moe_notes_list
        method = "pure_moe"
        model_stats = {"ensemble_notes": len(notes)}
        report("notes", f"MoE単独モード: {len(notes)} notes")
    elif crnn_notes_list:
        # CRNNフォールバック (MoE+BP両方失敗時)
        notes = crnn_notes_list
        method = "crnn_guitar"
        model_stats = {"crnn_notes": len(notes)}
        report("notes", f"CRNNフォールバック: {len(notes)} notes")
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
            
            # snap_idxをdownbeat（小節頭）にする
            if snap_idx > 0:
                # ビートを削るのではなく、削られるはずの snap_idx 個のビートを
                # beats[snap_idx] からテンポ(bpm)に基づいて逆方向に等間隔(sec_per_beat)で再配置し、
                # アライメントのために snap_idx を beats_per_bar の倍数に切り上げた個数にする。
                N = ((snap_idx + beats_per_bar - 1) // beats_per_bar) * beats_per_bar
                sec_per_beat = 60.0 / bpm if bpm > 0 else 0.5
                
                ref_time = beats[snap_idx]
                dummy_beats = [ref_time - (N - i) * sec_per_beat for i in range(N)]
                new_beats = dummy_beats + list(beats[snap_idx:])
                
                report("beats", f"ビート位相調整: first_bass={first_bass:.3f}s, "
                       f"nearest_beat[{snap_idx}]={ref_time:.3f}s, "
                       f"N={N} beats prepended, total beats={len(new_beats)}")
                beats = new_beats
                downbeats = [beats[i] for i in range(0, len(beats), beats_per_bar)]
                with open(session_dir / "beats.json", "w", encoding="utf-8") as f:
                    json.dump(_to_native({
                        "beats": beats, "bpm": bpm,
                        "time_signature": time_signature, "downbeats": downbeats,
                    }), f, ensure_ascii=False, indent=2)

    # --- Step 2.4: コード検出 (RMSエネルギー ＆ ノイズゲート生存ノート連動) ---
    chords = []
    try:
        t0_chords = time.time()
        from chord_detector import detect_chords, refine_chords_with_notes  # type: ignore
        from gp_renderer import _filter_noise
        gated_notes = _filter_noise(notes, 0.15) if notes else []
        chords = detect_chords(str(transcription_wav), beats=beats, key=detected_key or "C", notes=gated_notes)
        # ダイアトニックコード理論と検出単音に基づく補正
        if chords and gated_notes:
            report("chords", f"単音情報とダイアトニック理論によるコード補正を適用中... (キー: {detected_key or 'C'}, 生存ノート: {len(gated_notes)})")
            chords = refine_chords_with_notes(chords, gated_notes, key=detected_key or "C")
        report("chords", f"コード検出・補正完了: {len(chords)}区間 ({time.time()-t0_chords:.1f}s)")
        with open(session_dir / "chords.json", "w", encoding="utf-8") as f:
            json.dump(_to_native(chords), f, ensure_ascii=False, indent=2)
    except Exception as e:
        report("chords", f"コード検出スキップ: {e}")


    # --- チューニング推定 (TASK-892: ドローン解析 ＆ 動的バインド) ---
    tuning_suggestion = {"tuning": tuning_name, "confidence": 0}
    try:
        from tuning_detector import detect_tuning, detect_tuning_from_audio  # type: ignore
        if str(tuning_name).lower() in ("auto", ""):
            # ユーザーが "auto" を指定した場合のみ音声スペクトルから自動推定 (高確信度 0.70+ のみ採用)
            tuning_suggestion = detect_tuning_from_audio(str(transcription_wav), detected_key=detected_key)
            detected_cand = tuning_suggestion.get("tuning", "standard")
            conf = tuning_suggestion.get("confidence", 0.0)
            if detected_cand and detected_cand != "standard" and conf >= 0.70:
                tuning_name = detected_cand
                tuning = TUNINGS.get(tuning_name, STANDARD_TUNING)
                tuning_pitches = _get_open_string_pitches(tuning)
                report("tuning_detect", f"動的チューニング適応: {tuning_suggestion.get('label', tuning_name)} (確信度: {conf:.2f})")
            else:
                tuning_name = "standard"
                tuning = STANDARD_TUNING
                tuning_pitches = _get_open_string_pitches(tuning)
                report("tuning_detect", f"チューニング維持: standard (auto確信度不足: {conf:.2f})")
        else:
            # ユーザー明示指定（standard等）を絶対的SSOTとして採用（AI誤認を完全遮断）
            tuning = TUNINGS.get(tuning_name, STANDARD_TUNING)
            tuning_pitches = _get_open_string_pitches(tuning)
            tuning_suggestion = {"tuning": tuning_name, "confidence": 1.0}
            report("tuning_detect", f"ユーザー指定チューニング適用 (絶対的SSOT): {tuning_name}")
    except Exception as e:
        report("tuning_detect", f"チューニング推定スキップ: {e}")

    # --- Step: 音楽理論解析 ---
    report("theory", "音楽理論解析中...")
    t0 = time.time()
    rhythm_info = {'subdivision': 'straight', 'triplet_ratio': 0.0}
    detected_key_sig = detected_key or "C"
    try:
        from music_theory import detect_rhythm_pattern, detect_key_signature
        
        # MIDIベースのキー推定: オーディオベースの確信度が高い場合は上書きしない
        midi_key = detect_key_signature(notes)
        if key_confidence < 0.6 or detected_key is None:
            detected_key_sig = midi_key
        else:
            detected_key_sig = detected_key
            if midi_key != detected_key:
                print(f"[theory] キー競合: audio={detected_key}(conf={key_confidence:.2f}) vs midi={midi_key} → audio採用")

        # --- フェーズ3: 論文§6準拠クリーンパス (後処理・ヒューリスティクスの完全廃止によるアルペジオ保護) ---
        # 論文§6: "後処理（ノイズフィルタ、過度な量子化、倍音間引き等）は微細なアルペジオを破壊するため廃止し、AMTの出力を直接弦割り当てへ流す"
        report("theory", f"論文§6クリーンパス適用: 破壊的ヒューリスティック(HPC/MVS倍音フィルタ)を完全スキップ ({len(notes)} notesを100%保護)")

        # クリーンになった音符データに対してリズムパターン検出を実行
        rhythm_info = detect_rhythm_pattern(notes, beats)
        
        # 3/4拍子のアルペジオ3連符パターン補正
        # ロマンス等: onset fraction分析では検出できないが、
        # 1拍あたり3〜4音(ベース音を含む)のパターンが支配的なら3連符と判定
        if time_signature == "3/4" and rhythm_info["subdivision"] in ("straight", "mixed"):
            # numpy is imported globally at module level
            beats_arr = np.array(beats)
            notes_per_beat = []
            for bi in range(min(len(beats)-1, 60)):
                bt, nbt = beats[bi], beats[bi+1]
                count = sum(1 for n in notes if bt <= float(n["start"]) < nbt)
                if count > 0:
                    notes_per_beat.append(count)
            if notes_per_beat:
                avg_npb = np.mean(notes_per_beat)
                three_or_four_ratio = sum(1 for c in notes_per_beat if c in (3, 4)) / len(notes_per_beat)
                two_or_three = sum(1 for c in notes_per_beat if 2 <= c <= 4) / len(notes_per_beat)
                print(f"[theory] 3/4 arpeggio check: avg_npb={avg_npb:.1f}, 3-4_ratio={three_or_four_ratio:.2f}, 2-4_ratio={two_or_three:.2f}, beats_checked={len(notes_per_beat)}")
                # 3音または4音の拍が過半数（>=50%）かつ平均2.5以上4.5以下の場合
                if avg_npb >= 2.5 and avg_npb <= 4.5 and three_or_four_ratio >= 0.50:
                    rhythm_info["subdivision"] = "triplet"
                    rhythm_info["triplet_ratio"] = three_or_four_ratio
                    report("theory", f"3/4アルペジオ3連符パターン検出 (ベース音対応) "
                           f"(avg={avg_npb:.1f} notes/beat, 3-4 note ratio={three_or_four_ratio:.0%})")
        
        report("theory", f"音楽理論解析完了: rhythm={rhythm_info['subdivision']} "
               f"(triplet_ratio={rhythm_info.get('triplet_ratio', 0):.2f}), "
               f"key={detected_key_sig} ({time.time()-t0:.1f}s)")
        # beats.jsonにrhythm_infoを追記保存（_regenerate_musicxmlで参照される）
        try:
            beats_json_path = session_dir / "beats.json"
            if beats_json_path.exists():
                with open(beats_json_path, "r", encoding="utf-8") as f:
                    bd = json.load(f)
                bd["rhythm_info"] = rhythm_info
                with open(beats_json_path, "w", encoding="utf-8") as f:
                    json.dump(_to_native(bd), f, ensure_ascii=False)
        except Exception:
            pass
    except Exception as e:
        report("theory", f"音楽理論解析スキップ: {e}")

    # --- Step: 論文§6準拠クリーンパス: 全ノートを100%運指最適化へパス ---
    report("assign", f"論文§6クリーンパス: 勝手な間引きを完全禁止し全音符を運指エンジンへ投入 ({len(notes)} notes)")

    # --- Step: 弦/フレット最適化 (Viterbi DP) ---
    if method == "crnn_guitar":
        report("assign", f"CRNNハイブリッドモード: ピッチはCRNN, 弦/フレットはCNN分類器+Viterbi DP: {len(notes)} notes")
        for n in notes:
            n.pop("string", None)
            n.pop("fret", None)
            n.pop("cnn_string_probs", None)

    report("assign", "運指最適化中 (Viterbi DP)...")
    t0 = time.time()

    # ユーザー指定の noise_gate があればそれを絶対的 SSOT として最優先採用
    if noise_gate is not None:
        recommended_cut = float(noise_gate)
        report("assign", f"ユーザー指定 Noise Gate 適用 (絶対的SSOT): {recommended_cut:.2f}")
    else:
        recommended_cut = 0.0 if (is_solo_guitar or is_classic_profile) else 0.15
        report("assign", f"プロファイル自動 Noise Gate 適用: {recommended_cut:.2f}")

    try:
        from string_assigner import assign_strings_dp  # type: ignore

        # カポ適用チューニング
        capo = capo_result.get("capo", 0)
        capo_conf = capo_result.get("confidence", 0.0)
        if capo > 0 and capo_conf >= 0.95 and tuning_name == "auto":
            dp_tuning = [p + capo for p in tuning]
            report("assign", f"カポ{capo}フレット適用チューニングを使用 (確信度: {capo_conf:.2f})")
        else:
            dp_tuning = tuning

        notes = assign_strings_dp(
            notes,
            tuning=dp_tuning,
            initial_position=initial_position,
            chords=chords,
            audio_path=None if is_midi_bypass else str(wav_path),
            guitar_type=guitar_type,
            key=detected_key_sig,
        )
        report("assign", f"運指最適化完了: {len(notes)} notes ({time.time()-t0:.1f}s)")
    except Exception as e:
        import traceback
        report("assign", f"運指最適化スキップ（元出力をそのまま使用）: {e}")
        report("assign", f"[TRACEBACK] {traceback.format_exc()}")

    # フレット上限ガード（音符は絶対に削除せず、1弦のハイポジションまたはオクターブ畳み込みで全音保護）
    MAX_FRET = 22
    for n in notes:
        f = n.get("fret", 0)
        if f > MAX_FRET:
            n["string"] = 1
            n["fret"] = min(22, max(0, int(n.get("pitch", 60)) - tuning[5]))


    # --- Step: コンテキストジャンプフィルタ (論文§6準拠: 開放弦とハイポジションの往来・タッピング保護のため完全バイパス) ---
    print("[pipeline] 論文§6準拠: コンテキストジャンプフィルタを完全バイパス (ソロギターのハイポジション音を保護)", flush=True)

    # --- Step: 左手指番号割り当て (finger_assigner.py) ---
    try:
        from finger_assigner import assign_fingers
        t0_finger = time.time()
        notes = assign_fingers(notes, detected_key=detected_key_sig)
        report("assign", f"指番号割り当て完了: {len(notes)} notes ({time.time()-t0_finger:.1f}s)")
    except Exception as e:
        report("assign", f"指番号割り当てスキップ: {e}")

    # --- テクニック検出 (h/p/slide/bend) ---
    # トグルが全てOFFなら完全スキップ（37秒の短縮）
    if enable_technique_overlay or enable_technique_gp5:
        report("technique", "テクニック検出中 (h/p/slide/bend)...")
        try:
            from technique_detector import detect_techniques, add_techniques_to_musicxml_notes  # type: ignore
            t0 = time.time()
            # BP独自ノートはstring/fret推定精度が低いためテクニック検出対象外
            moe_notes  = [n for n in notes if not n.get("_bp_only")]
            bp_only_ns = [n for n in notes if n.get("_bp_only")]
            moe_notes = detect_techniques(moe_notes, bpm=bpm, audio_path=str(wav_path))
            moe_notes = add_techniques_to_musicxml_notes(moe_notes)
            notes = moe_notes + bp_only_ns
            notes.sort(key=lambda n: (float(n.get("start", 0)), int(n.get("pitch", 0))))
            tech_count = sum(1 for n in notes if n.get("technique") and n["technique"] != "normal")
            report("technique", f"テクニック検出完了: {tech_count}件 (BP独自{len(bp_only_ns)}件はスキップ) ({time.time()-t0:.1f}s)")
        except Exception as e:
            import traceback
            report("technique", f"テクニック検出スキップ: {e}")
            traceback.print_exc()
    else:
        report("technique", "テクニック検出スキップ（トグルOFF → 高速モード）")

    # --- CNN-based technique detection (PM/Harmonic/Bend/Slide/Vibrato) ---
    if enable_technique_gp5:
        report("technique_cnn", "CNN technique detection...")
        try:
            from technique_classifier_cnn import annotate_techniques_cnn
            t0 = time.time()
            moe_notes = [n for n in notes if not n.get("_bp_only")]
            bp_only_ns = [n for n in notes if n.get("_bp_only")]
            moe_notes = annotate_techniques_cnn(
                moe_notes, str(wav_path), confidence_threshold=0.80
            )
            notes = moe_notes + bp_only_ns
            notes.sort(key=lambda n: (float(n.get("start", 0)), int(n.get("pitch", 0))))
            cnn_count = sum(1 for n in notes
                           if n.get("technique_source") == "cnn")
            report("technique_cnn",
                   f"CNN detection done: {cnn_count} techniques "
                   f"({time.time()-t0:.1f}s)")
        except Exception as e:
            import traceback
            report("technique_cnn", f"CNN detection skipped: {e}")
            traceback.print_exc()
    else:
        report("technique_cnn", "CNN detection OFF (toggle disabled)")

    # --- テクニック情報に基づく指番号の微調整 ---
    if enable_technique_fingers:
        try:
            from finger_assigner import _apply_technique_constraints
            # technique_detector が note['technique'] に設定するので _technique に変換
            for n in notes:
                tech = n.get('technique', 'normal')
                if tech and tech != 'normal':
                    n['_technique'] = tech
            tech_fixes = _apply_technique_constraints(notes)
            # cleanup
            for n in notes:
                n.pop('_technique', None)
            if tech_fixes > 0:
                report("assign", f"テクニック反映指修正: {tech_fixes}件")
        except Exception as e:
            report("assign", f"テクニック指修正スキップ: {e}")
    else:
        report("assign", "テクニック指修正: OFF（トグル無効）")



    # --- 同一弦での同時発音（物理的不可能）の除去 ---
    same_str_removed = 0
    if len(notes) > 1:
        notes.sort(key=lambda n: float(n.get("start", 0)))
        cleaned_notes = []
        idx = 0
        while idx < len(notes):
            n = notes[idx]
            s = n.get("string")
            t = float(n.get("start", 0))
            if s is None:
                cleaned_notes.append(n)
                idx += 1
                continue
            # 同一弦の完全同時発音(<=15ms)を収集
            cluster = [n]
            j = idx + 1
            while j < len(notes):
                n_next = notes[j]
                if float(n_next.get("start", 0)) - t <= 0.015:
                    if n_next.get("string") == s:
                        cluster.append(n_next)
                    j += 1
                else:
                    break
            if len(cluster) == 1:
                cleaned_notes.append(n)
                idx += 1
            else:
                def get_score(item):
                    if item.get("_hard_protect_string"):
                        return 10.0
                    return float(item.get("velocity", 0.5))
                best_note = max(cluster, key=get_score)
                cleaned_notes.append(best_note)
                same_str_removed += (len(cluster) - 1)
                idx = j
        notes = cleaned_notes
        if same_str_removed > 0:
            report("assign", f"同一弦同時発音フィルタ: {same_str_removed}ノート除去 (物理的重複解消)")

    # --- 後処理1: ノート重複除去 ---
    # Pass 1: 完全重複 — 同一ピッチが短い時間窓内（<0.05秒）で重複検出される場合
    notes.sort(key=lambda n: (float(n.get("start", 0)), int(n.get("pitch", 0))))
    DEDUP_WINDOW = 0.05  # 緩和: 0.08→0.05秒 (速いパッセージを保護)
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
    # --- Step: PDF生成 ---
    # パイプライン中はスキップ: MuseScore(timeout=120s)とreportlab両方が
    # パイプラインをブロックするため。PDFは /result/{id}/pdf エンドポイントで
    # オンデマンド生成する（ユーザーがボタンを押した時のみ）。
    report("pdf", "PDF生成スキップ (オンデマンド生成: ダウンロードボタンから取得可能)")

    # Pass 2 は無効化: CRNNの0.2秒間隔重複はtriplet-eighth(0.224秒)と区別できないため、
    # タイムウィンドウベースの除去は正当な音を消すリスクが高い。
    # 代わりにtab_renderer.pyのtriplet再割り当てで同一位置のノートを統合する。

    if dedup_count > 0:
        report("assign", f"ノート重複除去: {dedup_count}ノート統合")

    # --- 後処理1.5: 共鳴音(sympathetic resonance)フィルタ ---
    # --- 後処理1.5: 共鳴音(sympathetic resonance)フィルタ (論文§6準拠: 開放弦・アルペジオ保護のため無効化) ---
    # ギターソロのアルペジオや開放弦ドローンを誤削除しないよう、AMTの出力を100%維持
    sympa_removed = 0
    print("[pipeline] 論文§6準拠: 共鳴音フィルタを完全バイパス (開放弦サステインを保護)", flush=True)

    # --- 後処理2: キー制約フィルタ ---
    # 無効化: キー制約フィルタはピッチ検出結果を破壊する可能性があるため無効化
    # Em楽曲がDキーで補正される問題を根本的に回避
    # MoEのピッチ検出精度が十分高い場合、スケール外の音は
    # 装飾音・経過音の可能性が高く、「最近隣補正」は適切でない
    key_fix_count = 0
    print(f"[DEBUG] key_filter: DISABLED (detected_key_sig={detected_key_sig}, key_confidence={key_confidence})", flush=True)

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
    final_note_entries = None
    try:
        gp5_bytes, final_note_entries = notes_to_gp5(
            notes,
            beats=beats,
            bpm=bpm,
            title=title,
            tuning=tuning,
            time_signature=time_signature,
            rhythm_info=rhythm_info,
            key_signature=detected_key_sig,
            noise_gate=recommended_cut,  # BPM適応: 遅い曲はCUT高め、速い曲は低め
            include_techniques=enable_technique_gp5,
            chords=chords,
            return_entries=True,
        )
        gp5_path = session_dir / "tab.gp5"
        with open(gp5_path, "wb") as f:
            f.write(gp5_bytes)
        report("musicxml", f"GP5生成完了: {len(gp5_bytes)} bytes")
    except Exception as e:
        report("musicxml", f"GP5生成失敗: {e}")
    # 量子化・位置情報（bar, beat_pos）付きの最終ノート情報を notes_assigned.json に保存
    notes_to_save = final_note_entries if final_note_entries is not None else notes

    # Notation Transformer (SoloTab-26K 記譜文法モデル) による声部・3連符タグの洗練
    try:
        from notation_transformer_infer import apply_notation_transformer
        notes_to_save = apply_notation_transformer(notes_to_save, beats_per_bar=beats_per_bar, divisions=12)
        notes = apply_notation_transformer(notes, beats_per_bar=beats_per_bar, divisions=12)
        report("musicxml", "Notation Transformer 適用完了 (SoloTab-26K)")
    except Exception as e:
        report("musicxml", f"Notation Transformer スキップ (フォールバック): {e}")

    with open(session_dir / "notes_assigned.json", "w", encoding="utf-8") as f:
        json.dump(_to_native(notes_to_save), f, ensure_ascii=False, indent=2)

    # Save the original complete notes (unfiltered, but with string/fret/finger assignments)
    with open(session_dir / "notes_assigned_original.json", "w", encoding="utf-8") as f:
        json.dump(_to_native(notes), f, ensure_ascii=False, indent=2)

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
        noise_gate=recommended_cut,  # BPM適応: GP5と同じ推奨CUT値
    )

    musicxml_path = session_dir / "tab.musicxml"
    with open(musicxml_path, "w", encoding="utf-8") as f:
        f.write(xml_content)

    # テクニックマップ保存 (AlphaTab API操作用)
    tech_path = session_dir / "techniques.json"
    with open(session_dir / "techniques.json", "w") as f:
        json.dump(_to_native(tech_map), f)

    # NextChord SoloTab テキスト形式出力
    try:
        from nextchord_renderer import notes_to_nextchord_text
        nextchord_txt = notes_to_nextchord_text(
            notes=notes_to_save,
            bpm=bpm,
            time_signature=time_signature,
            title=title,
            chords=refined_chords if 'refined_chords' in locals() else chords,
            beats_per_bar=beats_per_bar,
        )
        nextchord_path = session_dir / "tab.nextchord.txt"
        with open(nextchord_path, "w", encoding="utf-8") as f:
            f.write(nextchord_txt)
        report("musicxml", f"NextChord SoloTabテキスト出力完了: {nextchord_path.name}")
    except Exception as e:
        print(f"[pipeline] nextchord_renderer failed: {e}", flush=True)

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
        "noise_gate": recommended_cut,  # BPM適応CUT初期値
        "enable_technique_gp5": enable_technique_gp5,
        "enable_technique_overlay": enable_technique_overlay,
        "enable_technique_fingers": enable_technique_fingers,
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
