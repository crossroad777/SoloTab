"""
spectral_verifier.py — CQTベースのノート検証・発見
==============================================================
■ 検証モード (verify): モデルが検出した各ノートのCQTエネルギーを確認し、
  音源に存在しない「幻ノート」を除去。
■ 発見モード (discover): CQTのピーク解析でモデルが見逃したノートを発見・追加。
■ velocity修正: スペクトルエネルギーを反映した自然なvelocity値を付与。
"""

import numpy as np
import librosa
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


def _compute_cqt(wav_path: str, sr: int = 22050, hop_length: int = 512):
    """CQTスペクトログラムを計算する。"""
    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    C = np.abs(librosa.cqt(
        y, sr=sr, hop_length=hop_length,
        fmin=librosa.midi_to_hz(28),
        n_bins=68,  # MIDI 28~95 = 68半音
        bins_per_octave=12
    ))
    C_db = librosa.amplitude_to_db(C, ref=np.max(C))
    return C_db, y, sr, hop_length


def _get_energy_at(C_db, pitch, start, sr, hop_length, n_frames):
    """特定のピッチ/時間位置のCQTエネルギーを返す。"""
    pitch_idx = pitch - 28
    if pitch_idx < 0 or pitch_idx >= C_db.shape[0]:
        return None, None

    frame_idx = int(start * sr / hop_length)
    frame_idx = min(frame_idx, n_frames - 1)

    frame_start = max(0, frame_idx - 1)
    frame_end = min(n_frames, frame_idx + 3)
    if frame_start >= frame_end:
        return None, None

    note_energy = float(np.mean(C_db[pitch_idx, frame_start:frame_end]))
    frame_max = float(np.max(np.max(C_db[:, frame_start:frame_end], axis=0)))
    relative_energy = note_energy - frame_max

    return note_energy, relative_energy


def _get_onset_strength(C_db: np.ndarray, pitch: int, start: float, sr: int, hop_length: int, n_frames: int) -> float:
    """ピッチ/時間位置でのオンセット強度(時間軸のエネルギー差分)を計算する。"""
    pitch_idx = pitch - 28
    if pitch_idx < 0 or pitch_idx >= C_db.shape[0]:
        return 0.0

    frame_idx = int(start * sr / hop_length)
    if frame_idx < 3 or frame_idx > n_frames - 3:
        return 0.0

    before = np.mean(C_db[pitch_idx, frame_idx-3:frame_idx-1])
    after = np.max(C_db[pitch_idx, frame_idx:frame_idx+3])
    return float(after - before)


def verify_notes_with_spectrum(
    notes: List[Dict],
    wav_path: str,
    sr: int = 22050,
) -> List[Dict]:
    """
    CQTスペクトルで各ノートの実エネルギーを検証し、
    音源に存在しないノートを除去する。

    閾値はピッチ帯域に応じて内部で自動調整される:
    - 低音域 (MIDI ≤52): 絶対 -50dB / 相対 -33dB
    - 通常域: 絶対 -40dB / 相対 -25dB

    ★追加機能: 各ノートのvelocityをスペクトルエネルギーに基づいて修正。
    """
    if not notes:
        return notes

    C_db, y, sr, hop_length = _compute_cqt(wav_path, sr)
    n_frames = C_db.shape[1]

    # 全ノートのエネルギーを収集して正規化用の統計を取る
    energies = []
    for note in notes:
        energy, _ = _get_energy_at(
            C_db, note.get("pitch", 60), note.get("start", 0),
            sr, hop_length, n_frames
        )
        if energy is not None:
            energies.append(energy)

    if energies:
        max_energy = max(energies)
        min_energy = min(energies)
        energy_range = max(max_energy - min_energy, 1.0)
    else:
        max_energy, min_energy, energy_range = 0, -60, 60

    verified = []
    removed_count = 0

    for note in notes:
        pitch = note.get("pitch", 60)
        start = note.get("start", 0)

        note_energy, relative_energy = _get_energy_at(
            C_db, pitch, start, sr, hop_length, n_frames
        )

        if note_energy is None:
            verified.append(note)
            continue

        # オンセット強度による「減衰中の倍音」の棄却
        onset_strength = _get_onset_strength(
            C_db, pitch, start, sr, hop_length, n_frames
        )
        
        # === アルペジオ保護付きオンセット判定 ===
        # ソロギターのアルペジオでは、前の音の減衰中に次の音がピッキングされるため
        # onset_strengthが0付近になるのは自然。エネルギーと相対エネルギーを
        # 複合的に評価して、実音と倍音共鳴を区別する。
        
        if start > 0.2 and onset_strength <= 0.0:
            # onset_strengthが負（完全に減衰中）の場合:
            # エネルギーが高く（-20dB以上）かつ相対エネルギーも高い（-15dB以上）なら
            # アルペジオのメロディ音として保護
            if note_energy < -20.0 or relative_energy < -15.0:
                removed_count += 1
                continue
        
        # 微弱なオンセット（≤1.0dB）で、かつエネルギーが低い場合
        if start > 0.2 and onset_strength <= 1.0 and note_energy < -25.0:
            removed_count += 1
            continue

        # ピッチ依存の閾値 (v2: 引き上げて幻ノート除去を強化)
        if pitch <= 52:
            abs_threshold = -50
            rel_threshold = -33
        else:
            abs_threshold = -40
            rel_threshold = -25

        # 検証: 絶対エネルギーと相対エネルギーの両方をチェック
        if note_energy < abs_threshold or relative_energy < rel_threshold:
            removed_count += 1
            continue

        # ★ velocity をスペクトルエネルギーに基づいて修正
        # エネルギーを0.2~1.0の範囲にマッピング
        normalized = (note_energy - min_energy) / energy_range
        spectral_velocity = 0.2 + normalized * 0.8
        spectral_velocity = max(0.2, min(1.0, spectral_velocity))

        # 既存velocityとスペクトルvelocityのブレンド
        # スペクトル情報を50%反映。これにより実際の音量がTABに反映される。
        original_vel = note.get("velocity", 0.5)
        note["velocity"] = original_vel * 0.5 + spectral_velocity * 0.5
        note["spectral_energy"] = note_energy
        note["relative_energy"] = relative_energy
        verified.append(note)

    if removed_count > 0:
        logger.info(f"[spectral_verify] Removed {removed_count} phantom notes "
                     f"({len(verified)} remaining)")

    return verified


def discover_missing_notes(
    existing_notes: List[Dict],
    wav_path: str,
    beats: Optional[List[float]] = None,
    bpm: float = 120.0,
    sr: int = 22050,
    min_note_energy_db: float = -30.0,
    min_prominence_db: float = 12.0,
) -> List[Dict]:
    """
    CQTスペクトルのピーク解析からモデルが見逃したノートを発見する。

    ★ 原曲に忠実な転写のための最重要機能。
    モデルのノート検出は完璧ではないため、以下のケースでノートが欠落する:
    - 低音域のベースノート (Demucs/モデルの弱点)
    - 急速なアルペジオの中間音
    - 高い倍音成分を持つハーモニクス

    このステップでは、CQTのオンセット検出+ピーク解析で
    「明らかにエネルギーがあるのに検出されていない」音を復元する。

    Parameters
    ----------
    existing_notes : list[dict]
        既に検出済みのノートリスト
    wav_path : str
        WAVファイルパス
    beats : list[float], optional
        ビート位置
    bpm : float
        BPM
    min_note_energy_db : float
        ノートとして認識する最小絶対エネルギー (dB)
    min_prominence_db : float
        周辺ピッチとの差でノートと判定する最小突出度 (dB)

    Returns
    -------
    list[dict]
        既存ノート + 新規発見ノートの統合リスト
    """
    C_db, y, sr_actual, hop_length = _compute_cqt(wav_path, sr)
    n_frames = C_db.shape[1]
    duration = len(y) / sr_actual

    # --- Step 1: オンセット検出 (スペクトルフラックス法) ---
    # CQTの時間方向微分からオンセット候補を特定
    spectral_flux = np.sum(np.maximum(0, np.diff(C_db, axis=1)), axis=0)

    # 適応的閾値: 局所平均の1.5倍以上をオンセットとする
    window_size = max(1, int(0.5 * sr_actual / hop_length))  # 0.5秒窓
    if len(spectral_flux) < window_size * 2:
        return existing_notes

    from scipy.ndimage import uniform_filter1d
    local_mean = uniform_filter1d(spectral_flux, size=window_size)
    onset_threshold = local_mean * 1.5 + np.percentile(spectral_flux, 30)

    onset_frames = []
    for i in range(1, len(spectral_flux) - 1):
        if (spectral_flux[i] > onset_threshold[i] and
                spectral_flux[i] > spectral_flux[i - 1] and
                spectral_flux[i] > spectral_flux[i + 1]):
            onset_frames.append(i + 1)  # +1: diff offset

    # --- Step 2: 各オンセットでピッチピーク検出 ---
    # 既存ノートのカバレッジマップを作成
    existing_set = set()
    for n in existing_notes:
        # ±50ms, ±1半音の範囲のグリッドポイントを「カバー済み」とする
        p = n.get("pitch", 60)
        t = n.get("start", 0)
        f = int(t * sr_actual / hop_length)
        for dp in range(-1, 2):
            for df in range(-2, 3):
                existing_set.add((p + dp, f + df))

    discovered = []
    beat_dur = 60.0 / max(bpm, 40)
    min_note_duration = max(0.03, beat_dur * 0.1)  # 最小音価

    for onset_frame in onset_frames:
        onset_time = onset_frame * hop_length / sr_actual

        if onset_time > duration - 0.1:
            continue

        # 各ピッチごとのオンセット差分と絶対エネルギーを計算
        pitch_stats = {}
        for pitch_idx in range(12, 49):  # MIDI 40-76 (E2-E5) ギター実用音域
            f_start = max(0, onset_frame - 1)
            f_end = min(n_frames, onset_frame + 3)
            energy = np.max(C_db[pitch_idx, f_start:f_end])
            
            f_before_start = max(0, onset_frame - 4)
            f_before_end = max(1, onset_frame - 1)
            energy_before = np.mean(C_db[pitch_idx, f_before_start:f_before_end])
            
            pitch_stats[pitch_idx] = {
                'energy': energy,
                'onset_diff': energy - energy_before
            }

        # 適応的onset_diff閾値の計算
        # 現フレームの全ピッチonset_diffの中央値 + マージンを閾値とする
        all_diffs = [pitch_stats[pi]['onset_diff'] for pi in range(12, 49)]
        median_diff = float(np.median(all_diffs))
        # 最低でも10.0dB、中央値+7.0dBの大きい方を閾値とする
        adaptive_threshold = max(10.0, median_diff + 7.0)

        # 時間軸のエネルギー急増（オンセット差分）が周辺ピッチの中で突出しているか確認
        for pitch_idx in range(12, 49):
            stats = pitch_stats[pitch_idx]
            energy = stats['energy']
            onset_diff = stats['onset_diff']

            # 絶対エネルギー閾値
            if energy < min_note_energy_db:
                continue
            
            # 適応的オンセット急増判定
            if onset_diff < adaptive_threshold:
                continue

            # スペクトル漏れ対策1: オンセット差分自体がローカルピークであること
            # 隣接するピッチのオンセット差分より大きいか確認
            prev_diff = pitch_stats.get(pitch_idx - 1, {}).get('onset_diff', -100)
            next_diff = pitch_stats.get(pitch_idx + 1, {}).get('onset_diff', -100)
            if onset_diff <= prev_diff or onset_diff <= next_diff:
                continue
            
            # スペクトル漏れ対策2: prominence条件（弱いオンセットのみ適用）
            # onset_diffが30dB以上（明確なアタック）の場合は同時アタックの可能性が
            # 高いため免除。弱いオンセット(<30dB)のみCQTリーク対策を適用。
            if onset_diff < 30.0:
                neighbor_diffs = []
                for di in [-2, -1, 1, 2]:
                    ni = pitch_idx + di
                    if ni in pitch_stats:
                        neighbor_diffs.append(pitch_stats[ni]['onset_diff'])
                if neighbor_diffs:
                    neighbor_max = max(neighbor_diffs)
                    if onset_diff < neighbor_max * 1.5:
                        continue
            
            midi_pitch = pitch_idx + 28

            # 既にカバーされていなければ新規ノートとして追加
            if (midi_pitch, onset_frame) in existing_set:
                continue

            # エネルギーからvelocityを推定
            velocity = max(0.25, min(0.85, (energy + 60) / 50))

            # 音価の推定: 次のオンセットまで、またはエネルギーが消えるまで
            end_frame = onset_frame + int(0.2 * sr_actual / hop_length)  # デフォルト0.2秒
            for f in range(onset_frame + 1, min(n_frames, onset_frame + int(2.0 * sr_actual / hop_length))):
                if C_db[pitch_idx, f] < energy - 20:  # 20dB降下で終了
                    end_frame = f
                    break
            end_time = end_frame * hop_length / sr_actual

            if end_time - onset_time < min_note_duration:
                continue

            new_note = {
                "start": round(onset_time, 4),
                "end": round(end_time, 4),
                "pitch": midi_pitch,
                "velocity": round(velocity, 3),
                "string": 0,
                "fret": 0,
                "source": "spectral_discovery",
                "spectral_energy": float(energy),
                "confidence": round(onset_diff / 20.0, 2),
            }
            discovered.append(new_note)

            # カバレッジマップに追加
            for dp in range(-1, 2):
                for df in range(-2, 3):
                    existing_set.add((midi_pitch + dp, onset_frame + df))

    # --- Step 3: エネルギーピークスキャン（一時無効化） ---
    # 誤検出が多いため一旦無効化。モデルとspectral_discoveryのみで精度確認。
    energy_discovered = []
    # TODO: 精度改善後に再有効化
    
    if energy_discovered:
        logger.info(f"[energy_peak_scan] Found {len(energy_discovered)} additional notes "
                    f"from energy peak analysis")
        discovered.extend(energy_discovered)

    if discovered:
        logger.info(f"[spectral_discover] Total {len(discovered)} additional notes discovered")

    # 統合してソート
    all_notes = existing_notes + discovered
    all_notes.sort(key=lambda n: (n["start"], n.get("pitch", 0)))

    return all_notes
