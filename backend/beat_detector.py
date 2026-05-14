"""
beat_detector.py — ビート検出 + 拍子推定 (madmom + librosa)
=============================================================
音声ファイルからビート位置、BPM、拍子（3/4 or 4/4）を検出する。

改善ポイント:
1. madmom DBNDownBeatTracker で拍子の1拍目位置を検出
2. 3/4 vs 4/4 の自動推定
3. BPM倍取り/半取り補正の改善
"""

import numpy as np
import atexit
import warnings
from pathlib import Path

# solotab_utils import で NumPy/collections/ffmpeg パッチが自動適用
import solotab_utils  # noqa: F401

# atexit patch: background threadでのatexitエラーを回避
_original_atexit_register = atexit.register
def _safe_atexit_register(*args, **kwargs):
    try:
        return _original_atexit_register(*args, **kwargs)
    except Exception:
        pass
atexit.register = _safe_atexit_register

# madmom (optional) — pkg_resources deprecation warning を抑制
_HAS_MADMOM = False
_HAS_MADMOM_DOWNBEAT = False
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    try:
        from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
        _HAS_MADMOM = True
        try:
            from madmom.features.downbeats import (
                RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
            )
            _HAS_MADMOM_DOWNBEAT = True
        except ImportError:
            pass
    except ImportError:
        print("[beat_detector] Warning: madmom not available")

# プロセッサのキャッシュ（NNウェイト再ロード回避）
_cached_beat_proc = None
_cached_beat_tracker = None

def _get_beat_processors():
    global _cached_beat_proc, _cached_beat_tracker
    if _cached_beat_proc is None and _HAS_MADMOM:
        print("[beat_detector] Initializing RNNBeatProcessor (first call)...")
        _cached_beat_proc = RNNBeatProcessor()
        _cached_beat_tracker = DBNBeatTrackingProcessor(fps=100)
        print("[beat_detector] Processor cached.")
    return _cached_beat_proc, _cached_beat_tracker


def detect_beats(wav_path: str, *, _beat_proc=None, _beat_tracker=None) -> dict:
    """
    Detect beat positions, estimate BPM, and detect time signature.

    Returns
    -------
    dict with keys:
        beats : list[float]  — beat times in seconds
        bpm : float          — estimated BPM
        time_signature : str — "3/4" or "4/4"
        downbeats : list[float] — downbeat (1拍目) positions
    """
    if not _HAS_MADMOM:
        return _detect_beats_librosa(wav_path)

    # ffmpeg が PATH に存在するか最終チェック (madmom のクラッシュ防止)
    import shutil
    if not shutil.which("ffmpeg"):
        print("[beat_detector] Warning: ffmpeg not found in PATH. Falling back to librosa.")
        return _detect_beats_librosa(wav_path)

    # --- 音声を最初60秒に制限（速度改善）---
    import soundfile as sf
    import tempfile
    MAX_DURATION = 30  # 秒 (30秒で十分なビートパターンが得られる)
    truncated_path = wav_path
    try:
        info = sf.info(wav_path)
        if info.duration > MAX_DURATION:
            y_full, sr_full = sf.read(wav_path, stop=int(MAX_DURATION * info.samplerate))
            tmp_f = tempfile.NamedTemporaryFile(suffix='.wav', delete=False, dir=str(Path(wav_path).parent))
            sf.write(tmp_f.name, y_full, sr_full)
            truncated_path = tmp_f.name
            print(f"[beat_detector] Audio truncated: {info.duration:.0f}s → {MAX_DURATION}s for beat detection")
    except Exception as e:
        print(f"[beat_detector] Truncation failed, using full audio: {e}")

    # --- ビートトラッキング ---
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", message="dtype.*align")
        cached_proc, cached_tracker = _get_beat_processors()
        beat_proc = _beat_proc or cached_proc
        beat_tracker = _beat_tracker or cached_tracker

        activations = beat_proc(truncated_path)
        beats = beat_tracker(activations)

    # 一時ファイル削除
    if truncated_path != wav_path:
        try:
            Path(truncated_path).unlink()
        except Exception:
            pass

    # --- BPM推定 ---
    if len(beats) > 1:
        intervals = np.diff(beats)
        avg_interval = np.median(intervals)
        raw_bpm = 60.0 / avg_interval if avg_interval > 0 else 120.0
    else:
        raw_bpm = 120.0
        intervals = np.array([])

    # --- BPM倍取り/半取り補正 ---
    bpm = _correct_bpm_basic(raw_bpm)

    # --- librosa クロスバリデーション ---
    bpm = _cross_validate_bpm(bpm, wav_path, beats)

    # --- 拍子推定 (3/4 vs 4/4) ---
    time_signature, downbeats, downbeat_phase = _detect_time_signature(
        wav_path, beats, bpm, activations
    )

    # --- ビートグリッドの位相調整 ---
    # madmomのbeats配列の先頭がdownbeat（1拍目）とは限らない。
    # downbeat_phaseを使って、最初のdownbeat以前のビートを削除する。
    if downbeat_phase > 0 and downbeat_phase < len(beats):
        print(f"[beat_detector] Phase alignment: trimming {downbeat_phase} beats "
              f"before first downbeat (beat[0]={beats[0]:.3f}s -> beat[{downbeat_phase}]={beats[downbeat_phase]:.3f}s)")
        beats = beats[downbeat_phase:]
        # downbeatsもbeatsと整合させる
        beats_per_bar = 3 if time_signature == "3/4" else 4
        downbeats = beats[::beats_per_bar].tolist()

    # --- 3/4拍子のBPM補正 ---
    # クロスバリデーション後もまだ高い場合のみ適用
    if time_signature == "3/4" and bpm > 100:
        corrected_bpm = bpm * 2.0 / 3.0
        if 50 <= corrected_bpm <= 150:
            print(f"[beat_detector] 3/4 BPM correction: {bpm:.1f} -> {corrected_bpm:.1f}")
            bpm = corrected_bpm

    # --- ビート外挿: 検出範囲外(60s以降)のビートをBPMから生成 ---
    # MAX_DURATIONで制限した場合、曲の後半にはビートがない。
    # madmomの実際のビート間隔（中央値）を使って外挿する。
    import soundfile as sf
    try:
        full_info = sf.info(wav_path)
        full_duration = full_info.duration
    except Exception:
        full_duration = MAX_DURATION
    
    beats_list = beats.tolist()
    if len(beats_list) > 1 and beats_list[-1] < full_duration - 1.0:
        # madmomの実際のビート間隔を使用（BPMではなく）
        actual_intervals = np.diff(beats)
        actual_interval = float(np.median(actual_intervals))
        last_beat = beats_list[-1]
        while last_beat + actual_interval < full_duration + actual_interval:
            last_beat += actual_interval
            beats_list.append(round(last_beat, 4))
        
        # ダウンビートも外挿
        beats_per_bar = 3 if time_signature == "3/4" else 4
        if downbeats:
            last_db = downbeats[-1]
            bar_interval = actual_interval * beats_per_bar
            while last_db + bar_interval < full_duration + bar_interval:
                last_db += bar_interval
                downbeats.append(round(last_db, 4))
        
        print(f"[beat_detector] Beats extrapolated: {len(beats)} -> {len(beats_list)} "
              f"(interval={actual_interval:.4f}s, to {beats_list[-1]:.1f}s, duration={full_duration:.1f}s)")

    return {
        "beats": beats_list,
        "bpm": round(bpm, 1),
        "time_signature": time_signature,
        "downbeats": downbeats,
    }


def _correct_bpm_basic(raw_bpm: float) -> float:
    """
    BPMの倍取り/半取りを補正。
    ギターの一般的なテンポ範囲(50-160BPM)に収める。
    
    ギター曲の実態:
    - アルペジオ/バラード: 60-100 BPM
    - ポップス/ロック: 100-140 BPM
    - 速い曲: 140-160 BPM
    - 160超は極めて稀（ビートの倍取りの可能性が高い）
    """
    bpm = raw_bpm

    # 倍取り補正: BPM > 160 → 半分にする
    # (旧: 180だったが、ギター曲で160超はほぼ倍取り)
    while bpm > 160:
        bpm /= 2

    # 半取り補正: BPM < 45 → 倍にする
    while bpm < 45:
        bpm *= 2

    return bpm


def _cross_validate_bpm(madmom_bpm: float, wav_path: str, beats: np.ndarray) -> float:
    """
    madmom BPM と librosa BPM をクロスバリデーション。
    両エンジンの結果を比較し、整数比候補から最も自然なBPMを選択する。
    
    典型的な誤検出パターン:
    - 3/4拍子 88 BPM → madmom が 130.4 BPM (×3/2) と誤認
    - アルペジオの個々の音がビートとして拾われる
    """
    import librosa

    try:
        y, sr = librosa.load(wav_path, sr=22050, mono=True)
        tempo_result = librosa.beat.beat_track(y=y, sr=sr)
        librosa_bpm = float(tempo_result[0]) if not hasattr(tempo_result[0], '__len__') else float(tempo_result[0][0])
        librosa_bpm = _correct_bpm_basic(librosa_bpm)
    except Exception as e:
        print(f"[beat_detector] librosa BPM estimation failed: {e}")
        return madmom_bpm

    print(f"[beat_detector] Cross-validation: madmom={madmom_bpm:.1f}, librosa={librosa_bpm:.1f}")

    # 両者が近い(±15%)なら madmom を信頼
    ratio = madmom_bpm / librosa_bpm if librosa_bpm > 0 else 1.0
    if 0.85 <= ratio <= 1.15:
        print(f"[beat_detector] Engines agree (ratio={ratio:.2f}), using madmom={madmom_bpm:.1f}")
        return madmom_bpm

    # 乖離がある → 整数比候補を生成してスコアリング
    candidates = [
        (madmom_bpm, "madmom"),
        (librosa_bpm, "librosa"),
        (madmom_bpm * 2.0 / 3.0, "madmom×2/3"),  # 3/4拍子の誤検出修正
        (madmom_bpm / 2.0, "madmom/2"),
        (madmom_bpm * 2.0, "madmom×2"),
        (librosa_bpm * 2.0 / 3.0, "librosa×2/3"),
        (librosa_bpm / 2.0, "librosa/2"),
        (librosa_bpm * 2.0, "librosa×2"),
    ]

    # ビート間隔の安定性スコア（正しいBPMではビート間隔が安定する）
    beat_stability = {}
    if len(beats) > 4:
        intervals = np.diff(beats)
        for bpm_val, label in candidates:
            if bpm_val < 30 or bpm_val > 300:
                continue
            expected_interval = 60.0 / bpm_val
            # ビート間隔が期待値の整数倍に近いかチェック
            ratios_to_expected = intervals / expected_interval
            rounded = np.round(ratios_to_expected)
            rounded[rounded < 1] = 1
            deviations = np.abs(ratios_to_expected - rounded) / rounded
            stability = 1.0 / (1.0 + np.median(deviations))
            beat_stability[label] = stability

    best_bpm = madmom_bpm
    best_score = -1
    best_label = "madmom"

    for bpm_val, label in candidates:
        if bpm_val < 30 or bpm_val > 300:
            continue
        nat_score = _bpm_naturalness_score(bpm_val)
        stab_score = beat_stability.get(label, 0.5)

        # 両エンジンの元値に近い候補にボーナス
        agreement_bonus = 0.0
        for ref_bpm in [madmom_bpm, librosa_bpm]:
            if ref_bpm < 30 or ref_bpm > 300:
                continue
            if 0.9 <= bpm_val / ref_bpm <= 1.1:
                agreement_bonus = 0.2
                break

        total = nat_score * 0.5 + stab_score * 0.3 + agreement_bonus
        if total > best_score:
            best_score = total
            best_bpm = bpm_val
            best_label = label

    if abs(best_bpm - madmom_bpm) > 1.0:
        print(f"[beat_detector] BPM corrected: {madmom_bpm:.1f} -> {best_bpm:.1f} ({best_label})")
    else:
        print(f"[beat_detector] BPM confirmed: {best_bpm:.1f} ({best_label})")

    return best_bpm


def _detect_time_signature(
    wav_path: str,
    beats: np.ndarray,
    bpm: float,
    activations: np.ndarray = None,
) -> tuple:
    """
    3/4 vs 4/4 の拍子を推定する。

    方法:
    1. madmom downbeat tracker が利用可能なら使用
    2. librosa の onset strength のアクセントパターンから推定
    3. ビート間隔の周期性から推定

    Returns: (time_signature, downbeats, downbeat_phase)
        downbeat_phase: beatsリストの何番目が最初のdownbeatかを示すインデックス
                        0 = 先頭がdownbeat, 1 = 2番目がdownbeat, ...
    """
    downbeats = []

    # --- 方法1: madmom downbeat tracker ---
    if _HAS_MADMOM_DOWNBEAT:
        try:
            ts, dbeats, phase = _detect_ts_madmom_downbeat(wav_path, beats)
            if ts is not None:
                return ts, dbeats, phase
        except Exception as e:
            print(f"[beat_detector] Downbeat detection failed: {e}")

    # --- 方法2: onset strength のアクセントパターン ---
    try:
        ts, dbeats, phase = _detect_ts_accent_pattern(wav_path, beats, bpm)
        if ts is not None:
            return ts, dbeats, phase
    except Exception as e:
        print(f"[beat_detector] Accent pattern detection failed: {e}")

    # --- デフォルト: 4/4 ---
    if len(beats) > 0:
        downbeats = beats[::4].tolist()
    return "4/4", downbeats, 0


def _detect_ts_madmom_downbeat(wav_path: str, rnn_beats: np.ndarray = None) -> tuple:
    """madmom の DBNDownBeatTrackingProcessor で拍子推定 + downbeat位相検出
    
    Returns: (time_signature, downbeats, downbeat_phase)
        downbeat_phase: rnn_beatsリストの何番目が最初のdownbeatに最も近いか
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", message="dtype.*align")
        proc = RNNDownBeatProcessor()
        acts = proc(wav_path)

    # 3/4 と 4/4 の両方で試す
    results = {}
    for beats_per_bar in [3, 4]:
        try:
            tracker = DBNDownBeatTrackingProcessor(
                beats_per_bar=[beats_per_bar], fps=100
            )
            db_beats = tracker(acts)
            if len(db_beats) > 0:
                # db_beats: [[time, beat_number], ...]
                times = db_beats[:, 0]
                beat_nums = db_beats[:, 1].astype(int)

                # ダウンビート(1拍目)を抽出
                downbeat_mask = beat_nums == 1
                downbeat_times = times[downbeat_mask]

                # 小節間隔からBPMを再計算
                if len(downbeat_times) > 1:
                    bar_intervals = np.diff(downbeat_times)
                    median_bar = np.median(bar_intervals)
                    beat_interval = median_bar / beats_per_bar
                    est_bpm = 60.0 / beat_interval if beat_interval > 0 else 120.0
                else:
                    est_bpm = 120.0

                results[beats_per_bar] = {
                    "beats": db_beats,
                    "downbeats": downbeat_times.tolist(),
                    "bpm": est_bpm,
                    "beat_count": len(db_beats),
                }
        except Exception:
            continue

    if not results:
        return None, [], 0

    # 両方の結果がある場合、より適切な方を選択
    chosen_bpb = None
    if 3 in results and 4 in results:
        bpm3 = results[3]["bpm"]
        bpm4 = results[4]["bpm"]

        score3 = _bpm_naturalness_score(bpm3)
        score4 = _bpm_naturalness_score(bpm4)

        print(f"[beat_detector] 3/4: BPM={bpm3:.1f} score={score3:.2f}")
        print(f"[beat_detector] 4/4: BPM={bpm4:.1f} score={score4:.2f}")

        chosen_bpb = 3 if score3 > score4 else 4
    elif 3 in results:
        chosen_bpb = 3
    elif 4 in results:
        chosen_bpb = 4

    if chosen_bpb is None:
        return None, [], 0

    ts = f"{chosen_bpb}/4"
    dbeats = results[chosen_bpb]["downbeats"]

    # --- downbeat位相の計算 ---
    # rnn_beatsの中で、最初のdownbeat位置に最も近いビートのインデックスを求める
    phase = 0
    if rnn_beats is not None and len(rnn_beats) > 0 and len(dbeats) > 0:
        first_downbeat = dbeats[0]
        # rnn_beatsから最も近いビートを探す
        diffs = np.abs(rnn_beats - first_downbeat)
        nearest_idx = int(np.argmin(diffs))
        # ビート間隔の半分以内なら位相として採用
        if len(rnn_beats) > 1:
            beat_interval = float(np.median(np.diff(rnn_beats)))
            if diffs[nearest_idx] < beat_interval * 0.5:
                phase = nearest_idx % chosen_bpb
                if phase > 0:
                    print(f"[beat_detector] Downbeat phase detected: "
                          f"first downbeat={first_downbeat:.3f}s, "
                          f"nearest beat[{nearest_idx}]={rnn_beats[nearest_idx]:.3f}s, "
                          f"phase={phase}/{chosen_bpb}")

    return ts, dbeats, phase


def _detect_ts_accent_pattern(
    wav_path: str, beats: np.ndarray, bpm: float
) -> tuple:
    """
    ビート位置でのonset strengthのアクセントパターンから拍子を推定。
    3拍子グループと4拍子グループのどちらがアクセントパターンに合うかを比較。
    """
    import librosa

    y, sr = librosa.load(wav_path, sr=22050, mono=True)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    times = librosa.times_like(onset_env, sr=sr)

    if len(beats) < 6:
        return None, []

    # 各ビート時刻でのonset strengthを取得
    beat_strengths = []
    for bt in beats:
        idx = np.argmin(np.abs(times - bt))
        beat_strengths.append(onset_env[idx] if idx < len(onset_env) else 0)
    beat_strengths = np.array(beat_strengths)

    if beat_strengths.max() > 0:
        beat_strengths = beat_strengths / beat_strengths.max()

    # 3拍子グループと4拍子グループのアクセントパターンを比較
    score_3 = _compute_accent_score(beat_strengths, 3)
    score_4 = _compute_accent_score(beat_strengths, 4)

    print(f"[beat_detector] Accent scores: 3/4={score_3:.3f}, 4/4={score_4:.3f}")

    # 3/4の場合: BPMを2/3にして妥当性チェック
    bpm_if_3 = bpm * 2 / 3 if bpm > 100 else bpm
    bpm_if_4 = bpm

    nat_3 = _bpm_naturalness_score(bpm_if_3)
    nat_4 = _bpm_naturalness_score(bpm_if_4)

    # アルペジオ特性ボーナス: ビート間隔が非常に均一な場合、
    # 3/4拍子のアルペジオ曲である可能性が高い
    arpeggio_bonus_3 = 0.0
    if len(beats) > 8:
        intervals = np.diff(beats)
        cv = np.std(intervals) / np.mean(intervals)  # 変動係数
        if cv < 0.15:  # ビート間隔が非常に均一（CV < 15%）
            # 中テンポ(60-100BPM)のアルペジオ曲 → 3/4が自然
            if 60 <= bpm_if_3 <= 100:
                arpeggio_bonus_3 = 0.5  # 強烈なボーナスを与えて133BPMの16分音符ではなく88BPMの3連符として認識させる
                print(f"[beat_detector] Arpeggio pattern detected (CV={cv:.3f}), "
                      f"3/4 bonus +{arpeggio_bonus_3}")

    # アクセントスコアとBPM自然さの両方を考慮
    combined_3 = score_3 * 0.6 + nat_3 * 0.4 + arpeggio_bonus_3
    combined_4 = score_4 * 0.6 + nat_4 * 0.4

    print(f"[beat_detector] Combined: 3/4={combined_3:.3f} (bpm~{bpm_if_3:.0f}), "
          f"4/4={combined_4:.3f} (bpm~{bpm_if_4:.0f})")

    if combined_3 >= combined_4:  # 同点以上で3/4を優先（アルペジオ曲は3/4が多い）
        chosen_bpb = 3
    else:
        chosen_bpb = 4

    # --- 位相検出 ---
    # 各位相（0, 1, ..., bpb-1）でonset strengthの平均を比較し、
    # 最も強い位相をdownbeat（1拍目）とする
    # 全帯域 + 低音域（ベース音帯域）を組み合わせて判定
    
    # 低音域のonset strength（ギターのベース音帯域 ~40-250Hz）
    try:
        S_low = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=250)
        onset_low = librosa.onset.onset_strength(S=librosa.power_to_db(S_low), sr=sr)
        times_low = librosa.times_like(onset_low, sr=sr)
        
        # 各ビートでの低音onset strength
        bass_strengths = []
        for bt in beats:
            idx = np.argmin(np.abs(times_low - bt))
            bass_strengths.append(onset_low[idx] if idx < len(onset_low) else 0)
        bass_strengths = np.array(bass_strengths)
        if bass_strengths.max() > 0:
            bass_strengths = bass_strengths / bass_strengths.max()
    except Exception:
        bass_strengths = beat_strengths  # フォールバック
    
    best_phase = 0
    best_score = -1.0
    for phase in range(chosen_bpb):
        # 全帯域のonset strength
        full_str = beat_strengths[phase::chosen_bpb]
        full_avg = float(np.mean(full_str)) if len(full_str) > 0 else 0
        # 低音域のonset strength
        bass_str = bass_strengths[phase::chosen_bpb]
        bass_avg = float(np.mean(bass_str)) if len(bass_str) > 0 else 0
        # 組み合わせスコア（低音域を重視: 60%低音 + 40%全帯域）
        combined = bass_avg * 0.6 + full_avg * 0.4
        if combined > best_score:
            best_score = combined
            best_phase = phase

    if best_phase > 0:
        print(f"[beat_detector] Downbeat phase from accent+bass: phase={best_phase}/{chosen_bpb} "
              f"(score={best_score:.3f})")

    downbeats = beats[best_phase::chosen_bpb].tolist()
    return f"{chosen_bpb}/4", downbeats, best_phase


def _compute_accent_score(beat_strengths: np.ndarray, beats_per_bar: int) -> float:
    """
    指定された拍子でグループ化した時のアクセントパターンスコアを計算。
    1拍目が強く、他が弱いほどスコアが高い。
    """
    n = len(beat_strengths)
    if n < beats_per_bar * 2:
        return 0.0

    # 各小節の各拍位置の平均値を計算
    full_bars = n // beats_per_bar
    if full_bars < 2:
        return 0.0

    truncated = beat_strengths[:full_bars * beats_per_bar]
    reshaped = truncated.reshape(full_bars, beats_per_bar)

    # 各拍位置の平均accent strength
    avg_by_position = reshaped.mean(axis=0)

    if avg_by_position.sum() == 0:
        return 0.0

    # 1拍目の相対的な強さ
    downbeat_strength = avg_by_position[0]
    other_avg = avg_by_position[1:].mean()

    # スコア: 1拍目が他より強いほど高い
    if other_avg > 0:
        ratio = downbeat_strength / other_avg
    else:
        ratio = 2.0

    # 分散の逆数で一貫性を評価（アクセントパターンが安定しているか）
    variances = reshaped.var(axis=0)
    consistency = 1.0 / (1.0 + variances.mean())

    return min(ratio * consistency * 0.5, 1.0)


def _bpm_naturalness_score(bpm: float) -> float:
    """
    BPMが音楽的に自然な範囲にあるかのスコア。
    60-120 BPMが最も自然（アコギ曲の典型的テンポ範囲）。
    120-160 は倍取りの可能性があるため低めにスコアリング。
    """
    if 60 <= bpm <= 120:
        return 1.0
    elif 50 <= bpm < 60 or 120 < bpm <= 135:
        return 0.6
    elif 45 <= bpm < 50 or 135 < bpm <= 160:
        return 0.3
    else:
        return 0.1


def _detect_beats_librosa(wav_path: str) -> dict:
    """Fallback beat detection using librosa."""
    import librosa

    y, sr = librosa.load(wav_path, sr=22050, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    bpm = float(tempo) if not hasattr(tempo, '__len__') else float(tempo[0])
    
    # 拍子推定を呼び出す
    ts, dbeats = _detect_time_signature(wav_path, beat_times, bpm)

    return {
        "beats": beat_times.tolist(),
        "bpm": round(bpm, 1),
        "time_signature": ts,
        "downbeats": dbeats,
    }
