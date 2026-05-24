"""
technique_detector.py — ギター奏法テクニック検出エンジン (V3)
================================================================
Abesser et al. (ISMIR 2014/2015) に基づくF0軌跡解析と
スペクトル特徴による高精度テクニック検出。

検出テクニック:
  h   — ハンマリング・オン  (F0瞬間上行→安定)
  p   — プルオフ            (F0瞬間下行→安定)
  /   — スライドアップ      (F0線形上昇)
  \\   — スライドダウン      (F0線形下降)
  b   — ベンド              (F0上昇→ピーク→安定 or 下降)
  ~   — ビブラート          (F0振動 4–8Hz)
  gliss_up/gliss_down — グリッサンド (大幅フレット移動)
  harmonic — ナチュラルハーモニクス
  pm  — パームミュート      (スペクトル重心低下 + 短減衰)
  tr  — トリル              (h/p 4連続以上)

参考文献:
  [1] Abesser et al. (2014) "Automatic Transcription of Guitar Tones
      and Playing Techniques", ISMIR 2014.
  [2] Kehling et al. (2014) "Automatic Tablature Transcription of
      Electric Guitar Recordings", EUSIPCO 2014.
  [3] Stefani & Turchet (2022) "aGPTset", ICASSP 2022.
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Optional, Tuple


# =============================================================================
# 閾値定数
# =============================================================================
HP_MAX_IOI      = 0.25   # H/P 最大 IOI（秒）
SLIDE_MAX_IOI   = 0.40   # スライド 最大 IOI
SLIDE_MIN_FRET  = 2      # スライド 最小フレット差
GLISS_MIN_FRET  = 5      # グリッサンド 最小フレット差
BEND_MAX_IOI    = 0.25   # ベンド 最大 IOI
VIBRATO_MIN_DUR = 0.25   # ビブラート 最短持続時間
SLIDE_MAX_FRET  = 12     # スライド 最大フレット差（超えたらグリッサンド扱い）

# F0解析用
F0_SR           = 22050  # 内部リサンプリングSR
HOP_LENGTH      = 256    # PYIN hopサイズ
PYIN_FMIN       = 60     # F0最小Hz（ギター最低音E2 ≈ 82Hz）
PYIN_FMAX       = 1400   # F0最大Hz（ハイフレット）

# スペクトル特徴（PM/Harmonic）
PM_CENTROID_RATIO = 0.45  # PMはスペクトル重心が通常の45%以下
NH_FRETS          = {5, 7, 12}  # ナチュラルハーモニクス主要フレット


# =============================================================================
# メイン API
# =============================================================================

def detect_techniques(
    notes:      List[dict],
    *,
    bpm:        float = 120.0,
    key_signature: str = "C",
    audio_path: Optional[str] = None,
) -> List[dict]:
    """
    ノートリストにテクニック情報を付与する。

    Parameters
    ----------
    notes        : start/end/pitch/string/fret を持つノートリスト
    bpm          : テンポ（IOI閾値スケーリングに使用）
    key_signature: 調号
    audio_path   : 音声ファイルパス（F0解析に使用、省略可）
    """
    if len(notes) < 1:
        return notes

    # テンポ補正
    tempo_scale = min(1.6, max(0.6, 120.0 / max(bpm, 60.0)))
    hp_max    = HP_MAX_IOI    * tempo_scale
    slide_max = SLIDE_MAX_IOI * tempo_scale
    bend_max  = BEND_MAX_IOI  * tempo_scale

    # --- 音声ロード（F0解析用）---
    audio    = None
    audio_sr = None
    global_f0 = None
    global_voiced = None
    if audio_path:
        try:
            import librosa
            import time as _time
            audio, audio_sr = librosa.load(audio_path, sr=F0_SR, mono=True)
            print(f"[TechDet] Audio loaded: {len(audio)/audio_sr:.1f}s @ {audio_sr}Hz")
            # ★ F0を全体で1回だけ計算（最大の高速化ポイント）
            t0_f0 = _time.time()
            global_f0, global_voiced, _ = librosa.pyin(
                audio,
                fmin=PYIN_FMIN,
                fmax=PYIN_FMAX,
                sr=audio_sr,
                hop_length=HOP_LENGTH,
                fill_na=None,
            )
            global_f0 = np.where(global_voiced, global_f0, np.nan)
            print(f"[TechDet] Global F0 computed: {len(global_f0)} frames ({_time.time()-t0_f0:.1f}s)")
        except Exception as e:
            print(f"[TechDet] Audio load failed: {e}, falling back to rule-based")

    # --- 弦ごとに分離して処理 ---
    string_groups: Dict[int, List[int]] = {}
    for i, note in enumerate(notes):
        s = note.get("string")
        if s is not None:
            string_groups.setdefault(s, []).append(i)

    for string_num, indices in string_groups.items():
        indices_sorted = sorted(indices, key=lambda i: notes[i]["start"])

        for pos in range(len(indices_sorted)):
            curr_idx = indices_sorted[pos]
            curr     = notes[curr_idx]

            # 既に付与済みならスキップ
            if curr.get("technique") and curr["technique"] != "normal":
                continue

            if pos == 0:
                # --- ビブラート（単独ノート内F0解析）---
                if global_f0 is not None and (curr["end"] - curr["start"]) >= VIBRATO_MIN_DUR and curr.get("fret", 0) > 0:
                    tech = _detect_vibrato_from_f0(curr, audio, audio_sr, global_f0)
                    if tech:
                        curr["technique"] = tech
                continue

            prev_idx = indices_sorted[pos - 1]
            prev     = notes[prev_idx]

            # 既に付与済みならスキップ
            if prev.get("technique") and prev["technique"] != "normal":
                # currに対してビブラートだけチェック
                if global_f0 is not None and (curr["end"] - curr["start"]) >= VIBRATO_MIN_DUR and curr.get("fret", 0) > 0:
                    tech = _detect_vibrato_from_f0(curr, audio, audio_sr, global_f0)
                    if tech:
                        curr["technique"] = tech
                continue

            ioi        = curr["start"] - prev["start"]
            if ioi <= 0:
                continue

            pitch_diff = curr["pitch"] - prev["pitch"]
            abs_pitch  = abs(pitch_diff)
            fret_diff  = abs(curr.get("fret", 0) - prev.get("fret", 0))

            # ── F0軌跡解析（音声がある場合は優先） ──
            if global_f0 is not None and ioi <= max(slide_max, hp_max):
                tech = _classify_from_f0(
                    prev, curr, audio, audio_sr,
                    pitch_diff, fret_diff, hp_max, slide_max, bend_max,
                    global_f0=global_f0
                )
                if tech:
                    prev["technique"] = tech
                    continue

            # ── ルールベースフォールバック ──
            tech = _rule_based(
                ioi, pitch_diff, abs_pitch, fret_diff,
                hp_max, slide_max, bend_max,
                curr_fret=curr.get("fret", 0),
                prev_fret=prev.get("fret", 0)
            )
            if tech:
                prev["technique"] = tech
                continue

            # ── ビブラート（単独ノート）──
            if global_f0 is not None and (curr["end"] - curr["start"]) >= VIBRATO_MIN_DUR and curr.get("fret", 0) > 0:
                tech = _detect_vibrato_from_f0(curr, audio, audio_sr, global_f0)
                if tech:
                    curr["technique"] = tech

    # --- ナチュラルハーモニクス（フレット位置ベース）---
    for note in notes:
        if not note.get("technique") or note["technique"] == "normal":
            if note.get("fret") in NH_FRETS:
                _check_harmonic(note)

    # --- パームミュート（スペクトル重心ベース）---
    if audio is not None and global_f0 is not None:
        _detect_palm_mute_batch(notes, audio, audio_sr)

    # --- ブラッシング/デッドノート（YG Ex-22: ×）---
    # 音程のない打楽器的なノートを検出する
    # 方法: PYIN voiced_ratio < 0.3 + スペクトル平坦度 > 0.6
    if audio is not None and global_f0 is not None:
        _detect_dead_notes(notes, audio, audio_sr, global_f0, global_voiced)

    return notes



# =============================================================================
# F0 解析コア
# =============================================================================

def _extract_f0(
    audio: np.ndarray,
    sr:    int,
    t_start: float,
    t_end:   float,
    fmin_hz: float = PYIN_FMIN,
    fmax_hz: float = PYIN_FMAX,
    global_f0: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    指定区間のF0軌跡を取得する。
    global_f0がある場合はスライスするだけ（高速）。
    ない場合はフォールバックでpyinを呼ぶ。
    Returns: (f0_array, voiced_flag_array)
    """
    try:
        # ★ グローバルF0からスライス（高速パス）
        if global_f0 is not None:
            fps = sr / HOP_LENGTH
            start_frame = max(0, int(t_start * fps))
            end_frame = min(len(global_f0), int(t_end * fps) + 1)
            if end_frame - start_frame < 2:
                return np.array([]), np.array([])
            f0_slice = global_f0[start_frame:end_frame]
            voiced_slice = ~np.isnan(f0_slice)
            return f0_slice, voiced_slice

        # フォールバック: 個別pyin（global_f0がない場合のみ）
        import librosa
        start_sample = max(0, int(t_start * sr) - HOP_LENGTH)
        end_sample   = min(len(audio), int(t_end * sr) + HOP_LENGTH)
        segment      = audio[start_sample:end_sample]

        if len(segment) < HOP_LENGTH * 4:
            return np.array([]), np.array([])

        f0, voiced_flag, _ = librosa.pyin(
            segment,
            fmin=fmin_hz,
            fmax=fmax_hz,
            sr=sr,
            hop_length=HOP_LENGTH,
            fill_na=None,
        )
        f0_voiced = np.where(voiced_flag, f0, np.nan)
        return f0_voiced, voiced_flag
    except Exception as e:
        print(f"[TechDet] F0 extraction error: {e}")
        return np.array([]), np.array([])


def _midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def _classify_from_f0(
    prev:      dict,
    curr:      dict,
    audio:     np.ndarray,
    sr:        int,
    pitch_diff: int,
    fret_diff:  int,
    hp_max:    float,
    slide_max: float,
    bend_max:  float,
    **kwargs,
) -> Optional[str]:
    """
    F0軌跡を解析してテクニックを分類する。

    アルゴリズム (Abesser 2014 準拠):
      1. 前後ノート間のF0軌跡を抽出
      2. 軌跡の形状特徴量を計算:
         - 傾き (slope): 線形回帰の傾き
         - R²   (r2)   : 線形適合度
         - 速度変化 (jump): 最初の20%と最後の20%の平均F0差
         - 振動 (osc)  : 残差の標準偏差
      3. 特徴量パターンでテクニックを決定
    """
    ioi = curr["start"] - prev["start"]
    _global_f0 = kwargs.get('global_f0', None)

    # 分析区間: prev.start から curr.start + 0.05s
    t_a = prev["start"]
    t_b = curr["start"] + min(0.08, curr["end"] - curr["start"])
    f0, voiced = _extract_f0(audio, sr, t_a, t_b, global_f0=_global_f0)

    if len(f0) < 6:
        return None  # データ不足 → ルールベースに委ねる

    # voiced フレーム数が少なすぎる場合はスキップ
    valid  = f0[~np.isnan(f0)]
    if len(valid) < 4:
        return None

    n      = len(f0)
    t_arr  = np.arange(n) / (sr / HOP_LENGTH)  # 秒

    # IOI内のインデックス範囲
    ioi_frames = min(n, max(2, int(ioi * sr / HOP_LENGTH)))

    # ── 特徴量1: 線形回帰（傾き・R²）──
    valid_mask = ~np.isnan(f0[:ioi_frames])
    if valid_mask.sum() < 4:
        return None
    t_v = t_arr[:ioi_frames][valid_mask]
    f_v = f0[:ioi_frames][valid_mask]

    slope, intercept = np.polyfit(t_v, f_v, 1) if len(t_v) >= 2 else (0, f_v[0])
    f_pred = np.polyval([slope, intercept], t_v)
    ss_res = np.sum((f_v - f_pred) ** 2)
    ss_tot = np.sum((f_v - np.mean(f_v)) ** 2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 1e-6 else 0.0

    # ── 特徴量2: ジャンプ（最初20% vs 最後20%の平均差）──
    q = max(1, ioi_frames // 5)
    f0_start_region = f0[:q]
    f0_end_region   = f0[ioi_frames - q : ioi_frames]
    mean_start = np.nanmean(f0_start_region) if not np.all(np.isnan(f0_start_region)) else None
    mean_end   = np.nanmean(f0_end_region)   if not np.all(np.isnan(f0_end_region))   else None

    if mean_start is None or mean_end is None:
        return None

    jump_hz   = mean_end - mean_start
    jump_semi = 12 * np.log2(mean_end / mean_start) if mean_start > 0 and mean_end > 0 else 0.0

    # ── 特徴量3: ピーク検出（ベンド用）──
    f0_ioi   = f0[:ioi_frames]
    peak_idx = np.nanargmax(f0_ioi) if not np.all(np.isnan(f0_ioi)) else None
    if peak_idx is not None:
        peak_hz    = f0_ioi[peak_idx]
        peak_ratio = peak_idx / max(1, ioi_frames)  # ピーク位置比率
        peak_rise  = 12 * np.log2(peak_hz / mean_start) if mean_start > 0 and peak_hz > 0 else 0.0
    else:
        peak_ratio = 0.5
        peak_rise  = 0.0

    # ── 期待F0値（ピッチ→Hz）──
    hz_prev = _midi_to_hz(prev["pitch"])
    hz_curr = _midi_to_hz(curr["pitch"])

    # ──────────────────────────────────────────
    # テクニック判定ロジック
    # ──────────────────────────────────────────

    abs_jump = abs(jump_semi)
    abs_diff = abs(pitch_diff)

    # ── ベンド: F0が上昇してピークを形成し、終点が高い or 元に戻る ──
    # 開放弦(fret=0)はベンド不可能、ローフレット(1-2)も困難
    prev_fret = prev.get("fret", 0)
    curr_fret = curr.get("fret", 0)
    if (ioi <= bend_max + 0.05
            and peak_rise >= 0.8        # 0.8半音以上のピーク上昇（誤検出防止）
            and peak_ratio < 0.85       # ピークが最後でない
            and pitch_diff == 0         # 同フレット
            and fret_diff == 0
            and prev_fret >= 3          # 開放弦・ローフレットはベンド不可
            and curr_fret >= 3):
        return "b"

    # ── H / P: 急峻なジャンプ（線形でない、R²低い）──
    if ioi <= hp_max and abs_diff >= 1 and abs_diff <= 6:
        # H/P はF0が段階的に変化する（スライドと区別）
        # スライドはR²高い（線形）、H/Pは急峻（低R²またはジャンプ）
        if r2 < 0.65 or abs_jump >= 0.8 * abs_diff:
            # F0のジャンプ方向で判定
            if jump_semi > 0.3:
                return "h"
            elif jump_semi < -0.3:
                return "p"

    # ── スライド: 線形F0遷移（R²高い）──
    if ioi <= slide_max and SLIDE_MIN_FRET <= fret_diff:
        if r2 >= 0.55 and abs_jump >= 0.5:  # 線形かつ移動あり
            if jump_semi > 0:
                return "/"
            elif jump_semi < 0:
                return "\\"

    # ── グリッサンド: 大フレット移動 ──
    if ioi <= slide_max and fret_diff >= GLISS_MIN_FRET:
        if pitch_diff > 0:
            return "gliss_up"
        elif pitch_diff < 0:
            return "gliss_down"

    return None  # 判定不能 → ルールベースに委ねる


def _detect_vibrato_from_f0(
    note:  dict,
    audio: np.ndarray,
    sr:    int,
    global_f0: np.ndarray = None,
) -> Optional[str]:
    """
    単一ノートのF0を解析してビブラートを検出する。
    ビブラート = 4〜8Hzの周期的F0振動。
    """
    # 開放弦(fret=0)はビブラート不可能
    if note.get("fret", 0) == 0:
        return None

    try:
        f0, voiced = _extract_f0(
            audio, sr,
            note["start"] + 0.05,  # 最初50msはアタックなので除外
            note["end"]   - 0.02,
            global_f0=global_f0,
        )
        valid = f0[~np.isnan(f0)]
        if len(valid) < 20:
            return None

        # F0の標準偏差が閾値以上 → ピッチ揺れあり
        f0_std_cents = 1200 * np.std(valid) / np.mean(valid) if np.mean(valid) > 0 else 0

        if f0_std_cents < 25:  # 25セント未満は揺れなし（15→25に厳格化）
            return None

        # 周波数スペクトルで4–8Hzの振動を確認
        fps   = sr / HOP_LENGTH
        freqs = np.fft.rfftfreq(len(valid), d=1.0 / fps)
        power = np.abs(np.fft.rfft(valid - np.mean(valid))) ** 2

        vib_mask   = (freqs >= 3.5) & (freqs <= 9.0)
        total_power = power.sum()
        vib_power   = power[vib_mask].sum()

        if total_power > 0 and vib_power / total_power > 0.45:  # 0.30→0.45に厳格化
            return "~"

    except Exception:
        pass
    return None


# =============================================================================
# ルールベース フォールバック
# =============================================================================

def _rule_based(
    ioi:        float,
    pitch_diff: int,
    abs_pitch:  int,
    fret_diff:  int,
    hp_max:     float,
    slide_max:  float,
    bend_max:   float,
    curr_fret:  int = -1,
    prev_fret:  int = -1,
) -> Optional[str]:
    """F0解析なしのルールベース分類。"""
    # H / P
    if 0 < ioi <= hp_max and 0 < abs_pitch <= 6:
        return "h" if pitch_diff > 0 else "p"

    # スライド
    if 0 < ioi <= slide_max and SLIDE_MIN_FRET <= fret_diff <= SLIDE_MAX_FRET:
        if fret_diff <= 5:
            # 通常スライド
            if pitch_diff > 0:
                return "/"
            elif pitch_diff < 0:
                return "\\"

    # グリッサンド
    if 0 < ioi <= slide_max * 1.2 and fret_diff >= GLISS_MIN_FRET:
        return "gliss_up" if pitch_diff > 0 else "gliss_down"

    # ベンド（フレット押弦のみ: fret > 0 が条件）
    # オープン弦（fret=0）はベンド不可能
    # 注: pitch_diff==0の同フレット繰り返しノートは b_quarter にしない
    #     （アルペジオの繰り返し音を誤検出するため）
    #     F0解析がある場合のみベンドを検出する（_classify_from_f0が優先）
    if curr_fret >= 3 and prev_fret >= 3:
        if 0 < ioi <= bend_max and fret_diff == 0 and pitch_diff >= 1:
            if pitch_diff == 1:
                return "b_half"      # H.C: 半音
            elif pitch_diff == 2:
                return "b"           # C:   1音
            elif pitch_diff == 3:
                return "b_1half"     # 1H.C: 1音半
            elif pitch_diff >= 4:
                return "b_2"         # 2C: 2音以上
        # クォーターベンド: ルールベースでは無効（誤検出が多すぎる）
        # F0解析（_classify_from_f0）で検出された場合のみ有効
        # if 0 < ioi <= bend_max and fret_diff == 0 and pitch_diff == 0:
        #     return "b_quarter"


    return None


# =============================================================================
# ナチュラルハーモニクス
# =============================================================================

HARMONIC_FRETS = {12: 12, 7: 19, 5: 24}
DEFAULT_OPEN_STRINGS = [40, 45, 50, 55, 59, 64]

def _check_harmonic(note: dict) -> None:
    """
    ナチュラルハーモニクス判定。
    重要: フレット位置だけでなく弦番号も照合する。
    フレット5の通常音（例: 2弦5フレット=E4）が
    6弦のハーモニクスと誤判定されるバグを防ぐ。
    """
    fret = note.get("fret", -1)
    pitch = note.get("pitch", 0)
    string_num = note.get("string", -1)  # 1=1弦(最高音), 6=6弦(最低音)
    if fret not in HARMONIC_FRETS:
        return
    expected_harmonic_pitch = HARMONIC_FRETS[fret]
    open_pitch = pitch - expected_harmonic_pitch
    if open_pitch not in DEFAULT_OPEN_STRINGS:
        return
    # 弦番号と開放弦音高が一致するか確認
    if string_num >= 1 and string_num <= 6:
        string_idx = 6 - string_num  # 0=6弦(E2), 5=1弦(E4)
        actual_open = DEFAULT_OPEN_STRINGS[string_idx]
        if open_pitch != actual_open:
            return  # 別の弦のハーモニクスと誤認 → スキップ
    note["technique"] = "harmonic"


# =============================================================================
# パームミュート（バッチ処理）
# =============================================================================

def _detect_palm_mute_batch(
    notes: List[dict],
    audio: np.ndarray,
    sr:    int,
) -> None:
    """
    スペクトル重心を用いてパームミュートを検出する。

    パームミュートの特徴:
      - スペクトル重心が低い（通常音の 40〜50% 以下）
      - 持続時間が短い（減衰が速い）
    """
    try:
        import librosa
        # 全ノートの平均スペクトル重心を基準値として計算
        centroids = []
        for note in notes:
            if note.get("technique") and note["technique"] != "normal":
                continue
            dur = note["end"] - note["start"]
            if dur < 0.05:
                continue
            s = max(0, int(note["start"] * sr))
            e = min(len(audio), int(note["end"] * sr))
            seg = audio[s:e]
            if len(seg) < 512:
                continue
            sc = librosa.feature.spectral_centroid(y=seg, sr=sr).mean()
            centroids.append((note, sc))

        if not centroids:
            return

        median_centroid = np.median([c for _, c in centroids])

        for note, sc in centroids:
            if sc < median_centroid * PM_CENTROID_RATIO:
                dur = note["end"] - note["start"]
                if dur < 0.18:  # 短い音
                    if not note.get("technique") or note["technique"] == "normal":
                        note["technique"] = "pm"

    except Exception as e:
        print(f"[TechDet] Palm mute detection error: {e}")


# =============================================================================
# ブラッシング / デッドノート (YG Ex-22: ×)
# =============================================================================

def _detect_dead_notes(
    notes: List[dict],
    audio: np.ndarray,
    sr:    int,
    global_f0: np.ndarray = None,
    global_voiced: np.ndarray = None,
) -> None:
    """
    ブラッシング（デッドノート）を検出する。
    ★ 高速化: グローバルF0のvoiced_ratioを使い、per-note pyinを廃止。
    """
    try:
        import librosa
        fps = sr / HOP_LENGTH

        for note in notes:
            if note.get("technique") and note["technique"] not in ("normal", ""):
                continue

            dur = note["end"] - note["start"]
            if dur < 0.02:
                continue

            # ── 特徴量1: voiced_ratio（グローバルF0からスライス）──
            if global_voiced is not None:
                start_frame = max(0, int(note["start"] * fps))
                end_frame = min(len(global_voiced), int((note["start"] + min(dur, 0.15)) * fps))
                if end_frame > start_frame:
                    voiced_ratio = float(np.mean(global_voiced[start_frame:end_frame]))
                else:
                    voiced_ratio = 1.0
            else:
                voiced_ratio = 1.0

            # ── 特徴量2: スペクトル平坦度 ──
            s = max(0, int(note["start"] * sr))
            e = min(len(audio), int((note["start"] + min(dur, 0.15)) * sr))
            seg = audio[s:e]
            if len(seg) < 256:
                continue
            flatness = float(librosa.feature.spectral_flatness(y=seg).mean())

            # ── 判定ロジック ──
            is_unvoiced  = voiced_ratio < 0.35
            is_noisy     = flatness > 0.30  # 0.12は低すぎ（通常音もnoisy判定される）
            is_very_short = dur < 0.08

            if (is_unvoiced and is_noisy) or (is_unvoiced and is_very_short):
                note["technique"] = "x"

    except Exception as e:
        print(f"[TechDet] Dead note detection error: {e}")


# =============================================================================
# トリル後処理
# =============================================================================

def add_techniques_to_musicxml_notes(notes: List[dict]) -> List[dict]:
    """
    後処理: 連続H/Pチェーン(4音以上)をトリルに変換。
    """
    if len(notes) < 4:
        return notes

    TRILL_MIN_CHAIN = 4

    string_groups: Dict[int, List[int]] = {}
    for i, note in enumerate(notes):
        s = note.get("string")
        if s is not None:
            string_groups.setdefault(s, []).append(i)

    for string_num, indices in string_groups.items():
        indices_sorted = sorted(indices, key=lambda i: notes[i]["start"])
        chain = []

        for idx in indices_sorted:
            tech = notes[idx].get("technique")
            if tech in ("h", "p"):
                chain.append(idx)
            else:
                if len(chain) >= TRILL_MIN_CHAIN:
                    for ci in chain:
                        notes[ci]["technique"] = "tr"
                chain = []

        # 末端チェーン
        if len(chain) >= TRILL_MIN_CHAIN:
            for ci in chain:
                notes[ci]["technique"] = "tr"

    return notes
"""
"""
"""
"""
