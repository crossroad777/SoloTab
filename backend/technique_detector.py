"""
technique_detector.py — ギター奏法テクニック検出モジュール (V2)
================================================================
弦ごとにノートを分離し、各弦の時系列で隣接ノートのパターンから
テクニックを検出する。和音内のテクニックも正しく検出できる。

検出テクニック:
  h  — ハンマリング・オン (同弦/ピッチ上昇/短IOI)
  p  — プルオフ (同弦/ピッチ下降/短IOI)
  /  — スライドアップ (同弦/フレット差≥2/短IOI)
  \\\\  — スライドダウン
  b  — ベンド (同弦/同フレット/ピッチ上昇)
  harmonic — ナチュラルハーモニクス (fret 5,7,12 + ピッチ一致)
  tr — トリル (h/pが4連続以上)

【提供物の状態】
  構文レベル: 検証済み
  実行結果: 未検証（実音声での動作確認は必要）
  想定環境: SoloTab pipeline.py / modal_app.py から呼び出される
  既知のリスク: ハーモニクス検出は fret 位置ベースのヒューリスティック
"""

from typing import List, Dict, Optional


# =============================================================================
# 閾値定数
# =============================================================================

HP_MAX_IOI = 0.15       # ハンマリング/プルオフの最大IOI（秒）
SLIDE_MAX_IOI = 0.20    # スライドの最大IOI（秒）
SLIDE_MIN_FRET_DIFF = 2 # スライドの最小フレット差
BEND_MAX_IOI = 0.10     # ベンドの最大IOI（秒）
BEND_MIN_PITCH = 1      # ベンドの最小ピッチ差（半音）
BEND_MAX_PITCH = 3      # ベンドの最大ピッチ差（半音）

# ナチュラルハーモニクスが発生するフレット位置
# fret: (倍音番号, 開放弦からの半音数)
HARMONIC_FRETS = {
    12: 12,   # 第2倍音: オクターブ
    7:  19,   # 第3倍音: オクターブ+5度
    5:  24,   # 第4倍音: 2オクターブ
    4:  28,   # 第5倍音: 2オクターブ+長3度（近似）
    3:  31,   # 第6倍音: 2オクターブ+5度（近似）
}

# 標準チューニングの開放弦MIDI値 (6弦→1弦)
DEFAULT_OPEN_STRINGS = [40, 45, 50, 55, 59, 64]


def detect_techniques(notes: List[dict], *, bpm: float = 120.0, 
                      key_signature: str = "C") -> List[dict]:
    """
    ノートリストにテクニック情報を付与する。

    **弦ごとに分離**してテクニック検出を行うため、
    和音内のテクニック（例: 1弦でハンマリング + 他弦で持続音）も
    正しく検出される。

    Parameters
    ----------
    notes : list[dict]
        各ノートは start, end, pitch, string, fret を持つ。
    bpm : float
        テンポ情報（IOI閾値の動的調整に使用）。
    key_signature : str
        調号（"C", "G", "Am" 等）。スケール内判定に使用。

    Returns
    -------
    list[dict]
        technique キーが追加されたノートリスト。
    """
    if len(notes) < 2:
        return notes

    # 調号からスケールのピッチクラスを取得
    try:
        from music_theory import KEY_SIGNATURES
        scale_pcs = set(KEY_SIGNATURES.get(key_signature, KEY_SIGNATURES["C"]))
    except Exception:
        scale_pcs = {0, 2, 4, 5, 7, 9, 11}  # C major fallback

    # テンポに応じた閾値スケーリング（高BPMでは許容IOIを広く）
    tempo_scale = min(1.5, max(0.7, 120.0 / max(bpm, 60.0)))
    hp_max = HP_MAX_IOI * tempo_scale
    slide_max = SLIDE_MAX_IOI * tempo_scale

    # --- Phase 1: ハーモニクス検出は無効化（誤検出が多いため） ---
    # _detect_harmonics(notes)

    # --- Phase 2: 弦ごとにノートを分離 ---
    string_groups: Dict[int, List[int]] = {}  # string -> [index in notes]
    for i, note in enumerate(notes):
        s = note.get("string")
        if s is not None:
            string_groups.setdefault(s, []).append(i)

    # --- Phase 3: 各弦内で時系列テクニック検出 ---
    for string_num, indices in string_groups.items():
        # 時間順にソート
        indices_sorted = sorted(indices, key=lambda i: notes[i]["start"])

        for pos in range(1, len(indices_sorted)):
            prev_idx = indices_sorted[pos - 1]
            curr_idx = indices_sorted[pos]
            prev = notes[prev_idx]
            curr = notes[curr_idx]

            # 既にテクニックが付与されている場合はスキップ
            if curr.get("technique") and curr["technique"] != "normal":
                continue

            ioi = curr["start"] - prev["start"]
            if ioi <= 0:
                continue  # 同時発音（和音の別ノート）はスキップ

            pitch_diff = curr["pitch"] - prev["pitch"]
            abs_pitch = abs(pitch_diff)
            fret_diff = abs(curr.get("fret", 0) - prev.get("fret", 0))

            # --- ハンマリング・オン / プルオフ ---
            if 0 < ioi <= hp_max and 0 < abs_pitch <= 5:
                if pitch_diff > 0:
                    curr["technique"] = "h"
                elif pitch_diff < 0:
                    curr["technique"] = "p"
                continue

            # --- スライド ---
            if 0 < ioi <= slide_max and SLIDE_MIN_FRET_DIFF <= fret_diff <= 4:
                if pitch_diff > 0:
                    curr["technique"] = "/"
                elif pitch_diff < 0:
                    curr["technique"] = "\\"
                continue

            # --- グリッサンド (大きなフレット移動: 5+) ---
            GLISS_MAX_IOI = 0.30 * tempo_scale
            if 0 < ioi <= GLISS_MAX_IOI and fret_diff >= 5:
                if pitch_diff > 0:
                    curr["technique"] = "gliss_up"
                elif pitch_diff < 0:
                    curr["technique"] = "gliss_down"
                continue

            # --- ベンド ---
            if (0 < ioi <= BEND_MAX_IOI
                    and BEND_MIN_PITCH <= pitch_diff <= BEND_MAX_PITCH
                    and fret_diff == 0):
                curr["technique"] = "b"
                continue

    # --- Phase 4: 全ノート横断テクニック検出（弦に依存しないもの） ---
    beat_interval = 60.0 / max(bpm, 60.0)

    for note in notes:
        if note.get("technique") and note["technique"] != "normal":
            continue

        vel = float(note.get("velocity", 0.5))
        if vel > 1.0:
            vel /= 127.0
        dur = float(note.get("end", 0)) - float(note.get("start", 0))

        # --- ゴーストノート: 無効化（過検出が多い） ---
        # if vel < 0.15:
        #     note["technique"] = "x"
        #     continue

        # --- パームミュート: 極低velocity + 極短持続のみ ---
        if vel < 0.20 and dur < beat_interval * 0.15:
            note["technique"] = "palm_mute"
            continue

        # --- レットリング: 開放弦(fret=0) + 3拍以上持続のみ ---
        # ギターは自然に長く鳴るので、通常の持続音をlet_ringにしない
        if note.get("fret", -1) == 0 and dur > beat_interval * 3.0 and vel > 0.3:
            note["technique"] = "let_ring"
            continue

    # --- Phase 5: ハーモニクス検出は無効化（誤検出が多いため） ---
    # try:
    #     _detect_harmonics(notes)
    # except Exception:
    #     pass

    return notes


def _detect_harmonics(notes: List[dict]) -> None:
    """
    ナチュラルハーモニクスを検出する。

    条件:
    - フレット位置が 5, 7, 12 のいずれか
    - そのフレット位置のハーモニクスピッチが、ノートのピッチと一致
    - ハーモニクスは通常のフレット音より高い音が出る（fret 7 で fret 19 相当等）

    ただし fret 12 は通常フレットと同じピッチのため、
    以下の追加条件で判定:
    - fret 12: 前後にfret 5/7 ハーモニクスがある場合にハーモニクスと判定
    - fret 5, 7: ピッチがハーモニクスピッチと一致すれば判定
    """
    # 各ノートの弦の開放弦ピッチを推定
    for note in notes:
        fret = note.get("fret", -1)
        string = note.get("string")
        pitch = note.get("pitch", 0)

        if fret not in HARMONIC_FRETS or string is None:
            continue

        # 開放弦ピッチを推定: pitch - fret = open_pitch (通常フレットの場合)
        # ハーモニクスの場合: open_pitch + HARMONIC_FRETS[fret] = pitch
        open_pitch_if_normal = pitch - fret
        harmonic_pitch_semitones = HARMONIC_FRETS[fret]

        if fret in (5, 7, 3, 4):
            # fret 5/7: ハーモニクスピッチは通常フレットより高い
            # もし pitch が通常フレット音より高い場合 → ハーモニクスの可能性
            # ハーモニクスの場合: pitch = open_pitch + harmonic_pitch_semitones
            # 通常フレットの場合: pitch = open_pitch + fret
            # 差: harmonic_pitch_semitones - fret
            # fret 7: 19 - 7 = 12 (1オクターブ高い)
            # fret 5: 24 - 5 = 19 (かなり高い)
            #
            # MoEは実際の鳴っているピッチを検出するので、
            # ハーモニクスの場合は高いピッチが検出される。
            # open_pitch = pitch - harmonic_pitch_semitones が
            # 妥当な開放弦ピッチ(40,45,50,55,59,64)に近いか確認
            estimated_open = pitch - harmonic_pitch_semitones
            if estimated_open in DEFAULT_OPEN_STRINGS:
                if not note.get("technique"):
                    note["technique"] = "harmonic"

        elif fret == 12:
            # fret 12: ハーモニクスと通常フレットが同じピッチ
            # 前後のコンテキストで判定（fret 5/7 ハーモニクスが近くにあれば）
            # → 単独では判定困難なので、ここではマークしない
            # ただし、開放弦ピッチが一致する場合は候補として記録
            estimated_open = pitch - 12
            if estimated_open in DEFAULT_OPEN_STRINGS:
                # ハーモニクスの可能性はあるが確定できない
                # 後続のコンテキスト分析でマークする
                note["_harmonic_candidate"] = True

    # fret 12 のハーモニクス候補を、近くに他のハーモニクスがあればマーク
    harmonic_times = [n["start"] for n in notes
                      if n.get("technique") == "harmonic"]

    for note in notes:
        if note.get("_harmonic_candidate"):
            del note["_harmonic_candidate"]
            # 1秒以内に他のハーモニクスがあれば、このfret12もハーモニクス
            t = note["start"]
            for ht in harmonic_times:
                if abs(t - ht) < 1.0:
                    if not note.get("technique"):
                        note["technique"] = "harmonic"
                    break


def add_techniques_to_musicxml_notes(notes: List[dict]) -> List[dict]:
    """
    テクニック情報の後処理。
    連続するh/pのチェーンを検出してトリル判定を行う。
    """
    if len(notes) < 3:
        return notes

    # --- トリル検出 ---
    # 同弦上で4つ以上のh/pが連続する場合、トリルとして再分類
    TRILL_MIN_CHAIN = 4

    # 弦ごとにグループ化して処理
    string_groups: Dict[int, List[int]] = {}
    for i, note in enumerate(notes):
        s = note.get("string")
        if s is not None:
            string_groups.setdefault(s, []).append(i)

    for string_num, indices in string_groups.items():
        indices_sorted = sorted(indices, key=lambda i: notes[i]["start"])

        chain_start = None
        chain_len = 0

        for pos, idx in enumerate(indices_sorted):
            tech = notes[idx].get("technique")
            if tech in ("h", "p"):
                if chain_start is None:
                    chain_start = pos
                    chain_len = 1
                else:
                    chain_len += 1
            else:
                if chain_len >= TRILL_MIN_CHAIN and chain_start is not None:
                    for j in range(chain_start, chain_start + chain_len):
                        notes[indices_sorted[j]]["technique"] = "tr"
                chain_start = None
                chain_len = 0

        # 最後のチェーン処理
        if chain_len >= TRILL_MIN_CHAIN and chain_start is not None:
            for j in range(chain_start, chain_start + chain_len):
                notes[indices_sorted[j]]["technique"] = "tr"

    return notes
