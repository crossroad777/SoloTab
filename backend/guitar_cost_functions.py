"""
guitar_cost_functions.py — ギター運指コスト関数群
====================================================
Viterbi DP で使用する多属性コスト関数。

カテゴリ:
  1. 位置コスト   (_position_cost)   — フレット高、sweet spot
  2. 遷移コスト   (_transition_cost)  — ポジション移動、弦切替
  3. 音色コスト   (_timbre_cost)      — 開放弦ボーナス
  4. 人間工学コスト (_ergonomic_cost_chord) — フレットスパン、指交差
  5. 人間選好コスト (_human_preference_cost) — IDMT演奏データ

参考文献:
  - Bontempi "From MIDI to Rich Tablatures"
  - ISMIR "Minimax Approach to Guitar Fingering"
  - KTH "Guitar Fingering for Music Score"
  - SMC Fingerstyle論文
"""

from typing import List, Tuple, Optional
from collections import Counter
import json
import os
import numpy as np

MAX_FRET = 15  # アコースティックギターの実用フレット範囲

# --- 重みパラメータ (Optuna最適化済み: GuitarSet 10曲, 70.4% 弦一致率) ---
_DEFAULT_WEIGHTS = {
    # 位置コスト
    "w_fret_height":          0.69,    # フレット高コスト (Optuna: 0.6905)
    "w_high_fret_extra":      2.05,    # 9f超追加コスト (Optuna: 2.0494)
    "w_low_string_high_fret": 1.5,     # 低弦(4-6弦)ハイフレット倍率
    "w_sweet_spot_bonus":    -3.9,     # sweet spot (0-9f) ボーナス (Optuna: -3.8934)

    # 遷移コスト（ポジション連続性を強く重視）
    "w_movement":            24.9,     # ポジション移動コスト (Optuna: 24.9193)
    "w_position_shift":      29.0,     # ポジション跨ぎ追加コスト (Optuna: 28.953)
    "w_string_switch":        3.4,     # 弦切り替えコスト (Optuna: 3.3627)
    "w_same_string_repeat":   5.5,     # ⑧ 右手PIMA: 同弦連打ペナルティ (Optuna: 5.496)

    # 人間工学コスト
    "w_fret_span":          100.0,     # 和音フレットスパンコスト
    "w_unplayable":       10000.0,     # 物理的に弾けない配置
    "w_adjacent_stretch":    30.0,     # ⑨ 隣接弦ストレッチペナルティ (3f超)
    "w_too_many_fingers":  5000.0,     # ⑨ 4音超の同時押弦ペナルティ (バレーなし)

    # 音色コスト
    "w_open_string_bonus":   -9.7,     # 開放弦ボーナス (Optuna: -9.7349)
    "w_open_match_bonus":   -21.0,     # 開放弦でしか出せない音のボーナス (Optuna: -20.970)
    "w_barre_bonus":         -5.0,     # バレーコードボーナス (per extra string)

    # ⑦ フィンガースタイル弦域分離 (SMC Fingerstyle論文)
    "w_bass_low_string":   -20.0,     # ベース音(最低ピッチ)が低弦(4-6弦)ボーナス
    "w_melody_high_string":-15.0,     # メロディ音(最高ピッチ)が高弦(1-3弦)ボーナス
    "w_bass_wrong_string":  25.0,     # ベース音が高弦(1-3弦)にいるペナルティ
    # 人間運指選好 (IDMT human fingering)
    "w_human_pref_bonus":   -15.0,    # 人間が好むポジションへのボーナス

    # 法則3: ピッチ近接性弦保持 (3半音境界ルール)
    # <3半音 → 同弦維持ボーナス, ≥3半音 → 隣接弦遷移を許容
    "w_pitch_proximity_same_string":  -8.0,   # <3半音で同弦維持ボーナス
    "w_pitch_proximity_adj_string":   -3.0,   # ≥3半音で隣接弦遷移ボーナス

    # ⑪ 右手PIMA制約 (Skarha 2018, Optuna最適化済み)
    "w_pima_natural_bonus":   -9.3,    # R3: 自然位置ボーナス (Optuna: -9.2523)
    "w_pima_thumb_bass":     -15.0,    # R2: 親指=ベース弦ボーナス (Optuna: -15.043)
    "w_pima_thumb_wrong":     31.9,    # R2: 親指がメロディ弦ペナルティ (Optuna: 31.906)
    "w_pima_crossing":        22.4,    # R4: 右手の逆交差ペナルティ (Optuna: 22.422)
    "w_pima_ama_avoid":        8.0,    # R5: a-m-a交替回避ペナルティ
    "w_pima_same_finger":     10.5,    # R1: 同指連打禁止ペナルティ (Optuna: 10.478)

    # ⑫ Radicioni CSP: ポジション依存の指独立性 (ICMC 2004, Optuna最適化済み)
    "w_radicioni_stretch":     21.2,   # ポジション依存ストレッチ (Optuna: 21.159)
    "w_radicioni_independence": 2.6,   # 指の独立性制約 (Optuna: 2.578)
}


def _load_pdl_weights() -> dict:
    """
    PDL(Path Difference Learning)最適化重みをロード。
    optimized_weights.json が存在すれば、該当キーのみ上書きする。
    
    注意: PDLはTheory pathのみの最適化(65%)。3アプローチ統合環境では
    一部のキー（開放弦ボーナス等）はTheory pathの差別化要因として
    デフォルト値を維持する。
    """
    weights = dict(_DEFAULT_WEIGHTS)
    pdl_path = os.path.join(os.path.dirname(__file__), 'optimized_weights.json')
    # 3アプローチ統合で保護するキー（Theory pathの差別化に重要）
    _PROTECTED_KEYS = {"w_open_string_bonus", "w_open_match_bonus", 
                       "w_human_pref_bonus", "w_bass_low_string", "w_melody_high_string"}
    if os.path.exists(pdl_path):
        try:
            with open(pdl_path, 'r', encoding='utf-8') as f:
                pdl_data = json.load(f)
            pdl_weights = pdl_data.get('weights', {})
            merged, skipped = 0, 0
            for key, val in pdl_weights.items():
                if key in _PROTECTED_KEYS:
                    skipped += 1
                    continue
                if key in weights:
                    weights[key] = val
                    merged += 1
            acc = pdl_data.get('string_accuracy', 0)
            print(f"[guitar_cost] PDL重み適用: {merged}個上書き, {skipped}個保護 (PDL精度={acc:.4f})")
        except Exception as e:
            print(f"[guitar_cost] PDL重みロード失敗: {e}")
    return weights


WEIGHTS = _load_pdl_weights()

# --- ポジション定義 ---
# ギタリストは「ポジション」単位で指板を認識する
POSITION_WIDTH = 4  # 1ポジションのフレット幅（人差指〜小指）

# --- §4.1 Bio Viterbi: 生体力学的制約 ---
# 指の最大スパン (論文 Table §4.1)
_BIO_MAX_SPAN = {
    (1, 2): 4,   # 人差し指-中指: 3-4フレット
    (1, 3): 5,   # 人差し指-薬指: 4-5フレット
    (1, 4): 6,   # 人差し指-小指: 4-6フレット
    (2, 3): 3,   # 中指-薬指: 2-3フレット
    (2, 4): 4,   # 中指-小指: 3-4フレット
    (3, 4): 3,   # 薬指-小指: 2-3フレット
}


def get_finger_candidates(f: int) -> List[int]:
    """
    §4.2: フレットfに対して可能な指を返す。
    指: 0=なし(開放弦), 1=人差し, 2=中, 3=薬, 4=小指
    """
    if f == 0:
        return [0]  # 開放弦: 指なし
    fingers = []
    for finger in range(1, 5):
        # このフレットをこの指で押さえるとポジションは position = f - (finger - 1)
        position = f - (finger - 1)
        if position >= 1:
            fingers.append(finger)
    return fingers if fingers else [1]


def _bio_finger_transition_cost(finger: int, prev_finger: int,
                                 f: int, prev_f: int,
                                 s: int, prev_s: int) -> float:
    """
    §4.2: 生体力学的指遷移コスト。
    
    制約:
    1. 指の順序制約: fret(人差し指) ≤ fret(中指) ≤ fret(薬指) ≤ fret(小指)
    2. スパン制限: 指ペアごとの最大フレット幅
    3. 同一指の異フレット連打: 不自然な動き
    4. 腱結合(enslaving): 薬指の動きは中指・小指に不随意的に影響
    """
    cost = 0.0
    
    # 開放弦は指制約なし
    if finger == 0 or prev_finger == 0:
        return 0.0
    
    # 1. 指の順序制約 (絶対制約)
    # 同一弦上で、高いフレットに低い番号の指 → 指交差 → 大ペナルティ
    if s == prev_s:
        if f > prev_f and finger < prev_finger:
            cost += 30.0  # 指交差ペナルティ
        elif f < prev_f and finger > prev_finger:
            cost += 30.0  # 逆方向の指交差
    
    # 2. 同一指の異フレット連打 → 物理的に指を移動する必要がある
    if finger == prev_finger and f != prev_f:
        cost += 15.0
    
    # 3. スパン制約
    if finger != prev_finger and f > 0 and prev_f > 0:
        span = abs(f - prev_f)
        pair = (min(finger, prev_finger), max(finger, prev_finger))
        max_span = _BIO_MAX_SPAN.get(pair, 6)
        if span > max_span:
            cost += (span - max_span) * 8.0
    
    # 4. 腱結合(enslaving): 薬指(3)使用時、中指(2)・小指(4)のスパンに制約
    if finger == 3 or prev_finger == 3:
        other = prev_finger if finger == 3 else finger
        if other in (2, 4):
            span = abs(f - prev_f)
            if span > 2:
                cost += (span - 2) * 5.0  # 腱結合による追加制約
    
    return cost


# =============================================================================
# 人間運指選好マップ (IDMT-SMT-V2 由来)
# =============================================================================
_HUMAN_PREF_MAP = None

def _load_human_preference():
    """人間運指選好マップをロード (初回のみ)"""
    global _HUMAN_PREF_MAP
    if _HUMAN_PREF_MAP is not None:
        return _HUMAN_PREF_MAP

    pref_path = os.path.join(os.path.dirname(__file__), 'human_position_preference.json')
    if os.path.exists(pref_path):
        with open(pref_path, 'r', encoding='utf-8') as f:
            _HUMAN_PREF_MAP = json.load(f)
        print(f"[string_assigner] 人間運指選好マップ: {len(_HUMAN_PREF_MAP)} pitches loaded")
    else:
        _HUMAN_PREF_MAP = {}
    return _HUMAN_PREF_MAP


def _human_preference_cost(pitch: int, s: int, f: int) -> float:
    """
    人間運指選好コスト: IDMTの人間が選んだポジションにボーナスを付与。
    人間が高頻度で選ぶポジション → 低コスト（ボーナス）
    人間が選ばないポジション → コスト0（ペナルティなし）

    注意: sはget_possible_positions形式 (1=1弦E4, 6=6弦E2)
    マップはIDMT形式 (1=6弦E2, 6=1弦E4) → 変換: map_s = 7 - s
    """
    pref = _load_human_preference()
    if not pref:
        return 0.0

    pitch_data = pref.get(str(pitch))
    if not pitch_data:
        return 0.0

    prob = pitch_data.get('prob', {})
    # 弦番号変換: Viterbi形式(1=1弦) → IDMT形式(1=6弦)
    map_s = 7 - s
    key = f"{map_s}_{f}"
    p = prob.get(key, 0.0)

    if p > 0:
        # 人間がこのポジションを選んだ確率に応じたボーナス
        return WEIGHTS["w_human_pref_bonus"] * p
    return 0.0


def _get_max_span(fret: int) -> int:
    """
    ポジション依存のフレットスパン上限。
    ローポジション: フレット間隔が広い → スパン3
    ミドル: スパン4
    ハイポジション: フレット間隔が狭い → スパン5
    """
    if fret <= 3:
        return 3
    elif fret <= 9:
        return 4
    else:
        return 5


def _get_position_center(fret: int) -> int:
    """フレット番号からポジション中心を返す。"""
    if fret == 0:
        return 0  # 開放弦はポジション0
    return max(1, fret - 1)  # 人差指がfret-1にある想定


def _position_cost(s: int, f: int, pitch: int = None) -> float:
    """
    位置コスト: フレットの位置自体の弾きやすさ。
    高フレットほどコストが高い。sweet spot (0-7f) は良いスコア。
    人間運指選好マップが利用可能な場合、人間が好むポジションにボーナスを付与。
    """
    cost = 0.0

    # フレット高コスト
    cost += f * WEIGHTS["w_fret_height"]

    # 9フレット超の追加コスト（ソロギターでは7-9fも多用される）
    if f > 9:
        extra = (f - 9) * WEIGHTS["w_high_fret_extra"]
        # 低弦(4-6弦)のハイフレットはさらにコスト増
        if s >= 4:
            extra *= WEIGHTS["w_low_string_high_fret"]
        cost += extra

    # 開放弦ボーナス（手の位置に依存しない＝運指自由度が高い）
    if f == 0:
        cost -= 5.0

    # Sweet spot ボーナス (負のコスト) — 0-9fまで拡大
    if 0 <= f <= 9:
        cost += WEIGHTS["w_sweet_spot_bonus"]

    # 人間運指選好ボーナス (IDMT human fingering data)
    if pitch is not None:
        cost += _human_preference_cost(pitch, s, f)

    return cost


def _transition_cost(s: int, f: int,
                     prev_s: int, prev_f: int,
                     dt: float = 0.5,
                     pitch: int = None, prev_pitch: int = None) -> float:
    """
    遷移コスト: 前のポジションから今のポジションへの移動コスト。
    ポジション移動量 + 弦切り替え距離 + 生体力学的制約に基づく。

    Bio Viterbi統合 (LOPO +5.2%, same-player +2.6%):
    - ポジション移動は手全体の移動としてモデル化
    - 速いパッセージでの大きな移動は物理的に困難
    - 同弦上のフレット順序 = 指順序（暗黙的指割り当て）

    法則3: ピッチ近接性弦保持 (3半音境界ルール)
    - <3半音: 同弦維持が優勢 (96.6%→63.0%→18.7%)
    - ≥3半音: 隣接弦遷移が優勢 (80.1%→96.0%)
    """
    cost = 0.0

    # --- 生体力学: ポジション移動コスト (手全体の移動) ---
    if f == 0 or prev_f == 0:
        # 開放弦: ポジション制約を緩和（手の位置に依存しない）
        if f == 0:
            cost -= 3.0  # 開放弦ボーナス (強化)
        if prev_f == 0:
            cost += f * WEIGHTS["w_movement"] * 0.5
        elif f == 0:
            cost += 0  # 開放弦への移動はほぼ無コスト
    else:
        # 押弦同士: ポジション移動 = 手全体の移動
        fret_diff = abs(f - prev_f)

        # 生体力学: 速いパッセージほどポジション移動のコストが高い
        time_factor = 1.0 / max(dt, 0.05)
        bio_pos_cost = fret_diff * 7.5 * min(time_factor, 5.0)  # w_pos=0.5相当
        cost += bio_pos_cost

        # ポジション跨ぎペナルティ (4フレット超のジャンプ)
        if fret_diff > POSITION_WIDTH:
            cost += (fret_diff - POSITION_WIDTH) * WEIGHTS["w_position_shift"]

        # 生体力学: 同弦上でフレットが離れすぎ = 指ストレッチ
        if s == prev_s and fret_diff > 4:
            cost += (fret_diff - 4) * 15.0  # ストレッチペナルティ

    # 弦切り替えコスト
    string_dist = abs(s - prev_s)
    if string_dist > 0:
        cost += string_dist * WEIGHTS["w_string_switch"]
    else:
        # ⑧ 右手PIMA制約: 同じ弦の連打は右手の同指連打になり困難
        cost += WEIGHTS["w_same_string_repeat"]

    # --- 法則3: ピッチ近接性弦保持 (3半音境界ルール) ---
    # 810万ノートの統計: <3半音→同弦96.6~63%, ≥3半音→隣接弦80~96%
    if pitch is not None and prev_pitch is not None:
        interval = abs(pitch - prev_pitch)
        if interval < 3:
            # <3半音: 同弦維持を強くボーナス
            if s == prev_s:
                cost += WEIGHTS["w_pitch_proximity_same_string"]  # 負=ボーナス
        else:
            # ≥3半音: 隣接弦遷移をボーナス
            if string_dist == 1:
                cost += WEIGHTS["w_pitch_proximity_adj_string"]  # 負=ボーナス

    # --- ⑪ 右手PIMA制約 (Skarha 2018) ---
    # アルペジオ文脈で右手の自然な指使いを誘導
    # 弦番号: 1=1弦(E4), 2=2弦(B3), 3=3弦(G3), 4-6=ベース弦
    cost += _pima_transition_cost(s, prev_s)

    # --- ⑫ Radicioni CSP: ポジション依存の指独立性 (ICMC 2004) ---
    cost += _radicioni_independence_cost(f, prev_f, s, prev_s)

    return cost


# =============================================================================
# ⑪ 右手PIMA制約 (Skarha 2018 "IP for Optimal Right Hand Guitar Fingerings")
# =============================================================================
# クラシックギターの右手指割り当て:
#   p (pulgar/thumb)  → ベース弦 (4-6弦)
#   i (indice/index)  → 3弦 (自然位置)
#   m (medio/middle)  → 2弦 (自然位置)
#   a (anular/ring)   → 1弦 (自然位置)
#
# 5つの制約:
#   R1: 同指連打禁止 (同弦連打 → w_same_string_repeat で既に対応)
#   R2: 親指はベース弦専用
#   R3: ima は高弦の自然位置を好む
#   R4: 逆交差禁止 (低弦→高弦で i→m→a の順序)
#   R5: a-m-a 交替は避ける (腱結合で困難)

# PIMA自然位置マッピング (弦番号 → 推奨PIMA指)
_PIMA_NATURAL = {
    1: 'a',  # 1弦 → 薬指
    2: 'm',  # 2弦 → 中指
    3: 'i',  # 3弦 → 人差し指
    4: 'p',  # 4弦 → 親指
    5: 'p',  # 5弦 → 親指
    6: 'p',  # 6弦 → 親指
}

# PIMA指の順序 (低い番号 = 低い弦)
_PIMA_ORDER = {'p': 0, 'i': 1, 'm': 2, 'a': 3}


def _pima_transition_cost(s: int, prev_s: int) -> float:
    """
    右手PIMA遷移コスト (Skarha 2018)。
    弦の遷移パターンから右手の指使いを推定し、
    不自然な指使いにペナルティを課す。

    前提: 弦番号から暗黙的にPIMA指を推定。
    1弦=a, 2弦=m, 3弦=i, 4-6弦=p
    """
    cost = 0.0
    finger = _PIMA_NATURAL.get(s, 'p')
    prev_finger = _PIMA_NATURAL.get(prev_s, 'p')

    # R2: 親指=ベース制約
    # メロディ弦(1-3弦)にいるのに低ピッチ → 親指で弾くべきなのに弾けない
    if s >= 4:  # ベース弦
        cost += WEIGHTS["w_pima_thumb_bass"]  # 負=ボーナス
    elif prev_s >= 4 and s <= 3:
        # ベース→メロディの自然な遷移 (p→ima) → ボーナス
        cost += WEIGHTS["w_pima_natural_bonus"] * 0.5

    # R3: 自然位置ボーナス
    # 弦が自然位置に一致する場合にボーナス
    if s in (1, 2, 3):
        cost += WEIGHTS["w_pima_natural_bonus"] * 0.3  # 高弦にいること自体がボーナス

    # R4: 逆交差禁止
    # 低弦→高弦(数値が減る方向)で、PIMA順序が逆になるケース
    if s != prev_s and s <= 3 and prev_s <= 3:
        curr_order = _PIMA_ORDER.get(finger, 0)
        prev_order = _PIMA_ORDER.get(prev_finger, 0)
        # 弦が上がる(番号減る)のに指順序が下がる → 逆交差
        if s < prev_s and curr_order < prev_order:
            cost += WEIGHTS["w_pima_crossing"]
        elif s > prev_s and curr_order > prev_order:
            cost += WEIGHTS["w_pima_crossing"]

    # R1: 同指連打 (同弦連打は既にw_same_string_repeatで対応)
    # ここでは異弦だが同じPIMA指になるケースを追加
    if finger == prev_finger and s != prev_s:
        cost += WEIGHTS["w_pima_same_finger"]

    return cost


# R5 a-m-a回避 (3音文脈) — Viterbi後処理用
# Skarha: 薬指-中指-薬指(a-m-a, 弦1→2→1)の交替は腱結合で困難
_PIMA_AMA_PATTERNS = [
    (1, 2, 1),  # a-m-a: 弦1→2→1
    (2, 3, 2),  # m-i-m: 弦2→3→2 (同様に困難)
]


def pima_r5_postprocess(notes: list, tuning: list = None, max_fret: int = 15) -> list:
    """
    R5 a-m-a回避: Viterbi後処理。
    3連続ノートで a-m-a パターン(弦1→2→1等)が検出された場合、
    中央のノートの弦を変更可能なら変更する。

    Parameters
    ----------
    notes : 弦割り当て済みノートリスト
    tuning : チューニング
    max_fret : 最大フレット

    Returns
    -------
    notes : R5制約を適用したノートリスト
    """
    if len(notes) < 3:
        return notes

    if tuning is None:
        from solotab_utils import STANDARD_TUNING
        tuning = STANDARD_TUNING

    fixes = 0
    for i in range(1, len(notes) - 1):
        s_prev = notes[i - 1].get('string', 0)
        s_curr = notes[i].get('string', 0)
        s_next = notes[i + 1].get('string', 0)

        pattern = (s_prev, s_curr, s_next)
        if pattern in _PIMA_AMA_PATTERNS:
            # a-m-a detected: try to reassign middle note
            pitch = notes[i].get('pitch', 0)
            # Inline position computation to avoid circular import
            positions = []
            for ti, open_pitch in enumerate(tuning):
                fret = pitch - open_pitch
                if 0 <= fret <= max_fret:
                    string_num = 6 - ti
                    positions.append((string_num, fret))

            # Find an alternative string that breaks the pattern
            current_fret = notes[i].get('fret', 0)
            best_alt = None
            best_cost = float('inf')

            for alt_s, alt_f in positions:
                if alt_s == s_curr:
                    continue  # Same string, skip
                # Check if this breaks the a-m-a pattern
                new_pattern = (s_prev, alt_s, s_next)
                if new_pattern in _PIMA_AMA_PATTERNS:
                    continue  # Still an a-m-a pattern

                # Evaluate cost of this alternative
                cost = abs(alt_f - current_fret) * 5.0  # Fret distance penalty
                if alt_f == 0:
                    cost -= 5.0  # Open string bonus
                if cost < best_cost:
                    best_cost = cost
                    best_alt = (alt_s, alt_f)

            if best_alt and best_cost < 30.0:  # Only apply if cost is reasonable
                notes[i]['string'] = best_alt[0]
                notes[i]['fret'] = best_alt[1]
                fixes += 1

    if fixes > 0:
        print(f"[PIMA R5] a-m-a avoidance: {fixes} notes reassigned")

    return notes


# =============================================================================
# ⑫ Radicioni CSP: ポジション依存の指独立性 (ICMC 2004)
# =============================================================================

def _radicioni_independence_cost(f: int, prev_f: int, s: int, prev_s: int) -> float:
    """
    Radicioni & Lombardo (2004) のCSP生体力学モデルから着想。
    ポジション依存の指独立性: ローポジションではフレット間隔が広く
    指のストレッチがより困難。ハイポジションでは逆に容易。

    また、薬指と小指の独立性が低い（腱結合 = enslaving）ことを
    ポジションコストに反映。
    """
    cost = 0.0

    if f == 0 or prev_f == 0:
        return 0.0

    fret_diff = abs(f - prev_f)
    avg_fret = (f + prev_f) / 2

    # ポジション依存ストレッチ: ローポジションは物理的に広い
    if fret_diff > 2:
        # ローポジション(1-3f): フレット間隔=約36mm
        # ハイポジション(12f+): フレット間隔=約18mm
        # → ローポジションのストレッチはハイポジの2倍困難
        position_factor = max(0.5, 1.5 - avg_fret * 0.08)  # 1f=1.42, 12f=0.54
        cost += (fret_diff - 2) * WEIGHTS["w_radicioni_stretch"] * position_factor

    # 指の独立性制約: 薬指(3弦)と小指(4弦付近)の組み合わせは制御困難
    # 弦3と弦4の連続は薬指-小指の独立動作を要求
    if abs(s - prev_s) == 1 and min(s, prev_s) >= 3:
        # 高弦同士(3-4弦)は指の独立性が低い
        if fret_diff > 1:
            cost += WEIGHTS["w_radicioni_independence"] * fret_diff

    return cost


def _timbre_cost(s: int, f: int, tuning: List[int]) -> float:
    """
    音色コスト: 開放弦や音色的に好ましいポジションを優遇。
    開放弦のボーナスは控えめにし、過剰優遇を防ぐ。
    """
    cost = 0.0

    if f == 0:
        # 開放弦ボーナス（控えめ）
        cost += WEIGHTS["w_open_string_bonus"]

        # 開放弦でしか出せないピッチの場合に追加ボーナス
        # （他弦で同じピッチが弾けるなら開放弦にこだわる必要なし）
        string_idx = 6 - s
        if 0 <= string_idx < len(tuning):
            pitch = tuning[string_idx]
            # このピッチが他の弦でも出せるかチェック
            alt_count = 0
            for i, op in enumerate(tuning):
                if i == string_idx:
                    continue
                alt_fret = pitch - op
                if 0 < alt_fret <= MAX_FRET:  # fret>0 で弾ける代替
                    alt_count += 1
            if alt_count == 0:
                # この弦の開放弦でしか出せない音 → ボーナス
                cost += WEIGHTS["w_open_match_bonus"]

    return cost


def _ergonomic_cost_chord(combo: Tuple[Tuple[int, int], ...]) -> float:
    """
    人間工学コスト (和音用): 指の物理的制約を評価。
    - フレットスパン制約 (ポジション依存)
    - バレー判定
    - 指がかち合う判定
    """
    cost = 0.0
    all_frets = [f for _, f in combo]
    frets_nonzero = [f for f in all_frets if f > 0]

    if not frets_nonzero:
        return 0.0  # 全て開放弦 → コストゼロ

    min_fret = min(frets_nonzero)
    max_fret_val = max(frets_nonzero)
    span = max_fret_val - min_fret

    # ポジション依存のスパン制約
    max_span = _get_max_span(min_fret)
    if span > max_span:
        return WEIGHTS["w_unplayable"]  # 物理的に弾けない

    # フレットスパンコスト (スパンが広いほどコスト増)
    cost += span * WEIGHTS["w_fret_span"]

    # バレーコードボーナス: 同一フレットの複数弦を1本指で押さえる
    fret_counts = Counter(frets_nonzero)
    for fret_val, count in fret_counts.items():
        if count > 1:
            # バレーの場合、追加の弦はコストゼロに近い
            cost += (count - 1) * WEIGHTS["w_barre_bonus"]

    # 指がかち合うチェック: 低弦が高フレット・高弦が低フレットは不自然
    sorted_by_string = sorted(combo, key=lambda x: x[0], reverse=True)  # 6弦→1弦
    for i in range(len(sorted_by_string) - 1):
        s1, f1 = sorted_by_string[i]      # 低い弦
        s2, f2 = sorted_by_string[i + 1]  # 高い弦
        if f1 > 0 and f2 > 0 and f1 > f2 + 2:
            # 低弦が高弦より3フレット以上高い → 指が交差
            cost += 50.0

    # ⑨ 隣接弦ストレッチ制約 (JKU Inhibition Loss)
    # 隣接する弦で3フレット以上離れた同時押弦はストレッチが必要
    for i in range(len(sorted_by_string) - 1):
        s1, f1 = sorted_by_string[i]
        s2, f2 = sorted_by_string[i + 1]
        if f1 > 0 and f2 > 0 and abs(s1 - s2) == 1:
            fret_gap = abs(f1 - f2)
            if fret_gap > 3:
                cost += (fret_gap - 3) * WEIGHTS["w_adjacent_stretch"]

    # ⑨ 4音超の同時押弦制約 (バレーなしの場合)
    # 4本指しかないので、バレーコード以外では4弦超の同時押弦は物理的に困難
    if len(frets_nonzero) > 4:
        # バレーで同一フレットの弦は1指で押さえられるので差し引く
        unique_fret_positions = len(set(frets_nonzero))
        if unique_fret_positions > 4:
            cost += WEIGHTS["w_too_many_fingers"]
    else:
        unique_fret_positions = len(set(frets_nonzero)) if frets_nonzero else 0

    # ⑩ シェイクハンドグリップ: 6弦を親指で押さえる奏法
    # 条件: 6弦が低フレット(1-3)で、他の弦も同時に使用
    # 効果: 指交差ペナルティを軽減 + 使える指が1本増える
    string_fret_map = {s: f for s, f in combo}
    if 6 in string_fret_map:
        fret_6 = string_fret_map[6]
        other_frets = [f for s, f in combo if s != 6 and f > 0]
        if 1 <= fret_6 <= 3 and other_frets:
            # 親指で6弦を押さえられる → ボーナス
            cost -= 8.0
            # 指交差のリスクも軽減（親指は独立動作）
            if unique_fret_positions > 4:
                cost -= WEIGHTS.get("w_too_many_fingers", 100.0) * 0.5

    return cost
