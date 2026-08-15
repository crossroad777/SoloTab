"""
biomechanics_engine.py — Alice Lin (2026) "データ駆動型ギター運指" 完全統合
==========================================================================
主要理論:
1. Alice Lin (2026): "Data-Driven Guitar Fingering: Empirical Laws and Biomechanical Optimization"
2. 法則1 (ローポジション偏重): 音色差より手の到達容易性を優先
3. 法則2 (開放弦選好): 解剖学的リセットと共鳴を活用
4. 法則3 (3半音境界): Δp < 3半音は同弦維持、Δp >= 3半音は隣接弦遷移
5. 法則4 (ポジション固執): ポジション移動スパン > 2f でペナルティ
6. 法則5 (指順序性): 低ピッチ/高弦に低指番号、高ピッチ/低弦に高指番号
"""

import math
from typing import List, Tuple, Dict, Optional

INF = 1e9

# Alice Lin (2026) 26K GP5データ較正済みペナルティ重みマトリクス
CALIBRATED_WEIGHTS = {
    # 法則4: ポジション固執 (Position Stickiness)
    "w_position_shift_lin": 16.5,
    "w_shift_threshold": 2,      # 2フレットを超える移動にペナルティ
    
    # 法則3: 3半音境界 (3-Semitone Boundary)
    "w_pitch_proximity_same": -22.0,   # Δp < 3半音: 同弦維持ボーナス
    "w_pitch_proximity_adj": -14.0,    # Δp >= 3半音: 隣接弦遷移ボーナス
    
    # 解剖学的腱結合 (Junction & Independence)
    "w_ring_pinky_stretch": 45.0,      # 3-4指の2f+開きペナルティ
    "w_cross_finger_penalty": 85.0,    # 指のねじれ交差ペナルティ
    "w_ergonomic_unplayable": INF,     # 和音スパン > 5f の抹殺
    
    # 人間選好ボーナス (SoloTab-26K 較正)
    "w_human_pref_bonus": -35.0,
    "w_open_reset_bonus": -25.0,       # 開放弦による手の脱力リセット
}


def get_position_index(fret: int) -> int:
    """フレットから左手手首の基本ポジション（アンカー）を返す。"""
    if fret <= 0:
        return 0
    return max(1, fret - 1)


def get_max_stretch_for_fret(fret: int) -> int:
    """フレット位置に応じた人差し指〜小指の最大物理ストレッチ幅。"""
    if fret <= 4:
        return 4  # ローポジション（フレット幅が広いため最大4フレット）
    elif fret <= 9:
        return 5  # ミドルポジション（最大5フレット）
    else:
        return 6  # ハイポジション（フレット幅が狭いため最大6フレット）


def chord_reachability_cost(notes_sf: List[Tuple[int, int]]) -> float:
    """
    Chord Reachability (和音の到達可能性):
    同時押弦において、人差し指〜小指の最大スパン（4〜5フレット）を超える場合 INF を返す。
    """
    pressed = [(s, f) for s, f in notes_sf if f > 0]
    if len(pressed) <= 1:
        return 0.0
    if len(pressed) > 4:
        min_fret = min(f for s, f in pressed)
        barre_notes = [s for s, f in pressed if f == min_fret]
        if len(barre_notes) >= 2:
            if len(pressed) - len(barre_notes) > 3:
                return INF
        else:
            return INF

    frets = [f for s, f in pressed]
    min_f = min(frets)
    max_f = max(frets)
    span = max_f - min_f
    if span > get_max_stretch_for_fret(min_f):
        return INF
    return span * 4.0


def evaluate_alice_lin_laws(s: int, f: int, prev_s: int, prev_f: int,
                            pitch: Optional[int] = None, prev_pitch: Optional[int] = None,
                            finger: int = 1, prev_finger: int = 1,
                            ioi: float = 0.3) -> float:
    """
    Alice Lin (2026) の 5法則に基づく遷移コスト評価。
    """
    cost = 0.0

    # 開放弦 (法則2: 開放弦選好と生体力学的リセット)
    if f == 0 or prev_f == 0:
        return CALIBRATED_WEIGHTS["w_open_reset_bonus"] if f == 0 else 0.0

    # 1. 法則4: ポジション固執 (直前からのフレット移動が2フレットを超える場合ペナルティ)
    fret_diff = abs(f - prev_f)
    if fret_diff > CALIBRATED_WEIGHTS["w_shift_threshold"]:
        excess = fret_diff - CALIBRATED_WEIGHTS["w_shift_threshold"]
        # 1弦メロディ上の自然なスライド移動は軽減
        if s == 1 and prev_s == 1:
            cost += excess * (CALIBRATED_WEIGHTS["w_position_shift_lin"] * 0.15)
        else:
            time_factor = 1.0 / max(0.08, min(0.8, ioi))
            cost += (excess ** 1.4) * CALIBRATED_WEIGHTS["w_position_shift_lin"] * (time_factor ** 0.4)

    # 2. 法則3: 3半音境界 (Δp < 3 は同弦、Δp >= 3 は隣接弦)
    if pitch is not None and prev_pitch is not None:
        delta_p = abs(pitch - prev_pitch)
        if delta_p < 3:
            if s == prev_s:
                cost += CALIBRATED_WEIGHTS["w_pitch_proximity_same"]  # 同弦維持ボーナス
            else:
                cost += 15.0  # 3半音未満なのにわざわざ弦を変えるペナルティ
        else:
            if abs(s - prev_s) == 1:
                cost += CALIBRATED_WEIGHTS["w_pitch_proximity_adj"]   # 隣接弦遷移ボーナス
            elif s == prev_s and delta_p >= 7:
                cost += 25.0  # 7半音以上の同一弦無理跳躍ペナルティ

    # 3. 解剖学的腱結合制約 (薬指3と小指4の独立性・指交差)
    if finger > 0 and prev_finger > 0:
        # 薬指と小指の開き
        if (finger == 3 and prev_finger == 4) or (finger == 4 and prev_finger == 3):
            if fret_diff > 1:
                cost += CALIBRATED_WEIGHTS["w_ring_pinky_stretch"] * (fret_diff - 1)
        # クロスフィンガリング
        if (finger < prev_finger and f > prev_f) or (finger > prev_finger and f < prev_f):
            cost += CALIBRATED_WEIGHTS["w_cross_finger_penalty"]

    return cost



def position_shift_inertia_cost(s: int, f: int, prev_s: int, prev_f: int, ioi: float = 0.3) -> float:
    """
    2. Position Shift Inertia (ポジション移動の慣性)
    前後のノートで手首のポジションが変わる場合、手首の質量移動に対する慣性コスト。
    同じポジションに留まる運指を最優先し、移動する場合はIOI（時間）に応じたペナルティを課す。
    """
    if f == 0 or prev_f == 0:
        return 0.0  # 開放弦を挟む移動は手の位置をリセットできるため慣性コスト無料

    pos = get_position_index(f)
    prev_pos = get_position_index(prev_f)
    pos_diff = abs(pos - prev_pos)

    if pos_diff == 0:
        # 同一ポジション維持（生体力学的に最も安定）
        return -8.0

    # ポジション移動距離に応じた慣性ペナルティ
    # 短い時間（高速）での大きな移動ほど指数関数的に高コスト
    time_factor = 1.0 / max(0.05, min(1.0, ioi))
    inertia = (pos_diff ** 1.5) * 12.0 * (time_factor ** 0.5)

    # 1弦上の自然なスライド移動は慣性コストを割引
    if s == 1 and prev_s == 1:
        inertia *= 0.25

    return inertia


def finger_independence_cost(finger1: int, fret1: int, string1: int,
                             finger2: int, fret2: int, string2: int) -> float:
    """
    3. Finger Independence (指の独立可動域 & 腱結合制約)
    特に薬指(3)と小指(4)の腱結合、および指の交差（ねじれ）に対するペナルティ。
    """
    if finger1 == 0 or finger2 == 0:
        return 0.0

    fret_diff = abs(fret1 - fret2)
    
    # 薬指(3)と小指(4)の独立性制約 (Junod 2011 / Skarha 2018)
    if (finger1 == 3 and finger2 == 4) or (finger1 == 4 and finger2 == 3):
        if fret_diff > 1:
            # 薬指と小指を2フレット以上開くのは解剖学的に困難
            return 45.0 * (fret_diff - 1)
        elif string1 != string2 and fret_diff > 0:
            # 異なる弦で薬指・小指を逆方向に開く負荷
            return 20.0

    # クロスフィンガリング（指の逆交差）ペナルティ
    # 例: 低いフレットを高い指番号（例: 1fに4指）、高いフレットを低い指番号（例: 4fに1指）
    if finger1 < finger2 and fret1 > fret2:
        return 80.0 * (fret1 - fret2)
    if finger1 > finger2 and fret1 < fret2:
        return 80.0 * (fret2 - fret1)

    # 同一指の無理な連続跳躍 (スライド以外)
    if finger1 == finger2 and (string1 != string2 or fret_diff > 2):
        return 35.0

    return 0.0


def evaluate_biomechanics_penalty(s: int, f: int, prev_s: int, prev_f: int,
                                  finger: int = 1, prev_finger: int = 1,
                                  ioi: float = 0.3) -> float:
    """
    総合バイオメカニクスペナルティの計算（Viterbi DP の遷移コストに加算）。
    """
    cost = 0.0
    # 1. 慣性コスト
    cost += position_shift_inertia_cost(s, f, prev_s, prev_f, ioi=ioi)
    # 2. 指の独立・交差コスト
    cost += finger_independence_cost(prev_finger, prev_f, prev_s, finger, f, s)
    return cost
