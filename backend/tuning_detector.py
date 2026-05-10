"""
tuning_detector.py — チューニング自動検出
==========================================
検出されたノートのピッチ分布から、使用されているチューニングを推定する。

原理:
  各チューニングには特有の最低音(6弦開放)がある。
  検出されたノートの最低ピッチとフレット分布から
  最も可能性の高いチューニングを推定する。
"""

import numpy as np
from typing import List, Dict, Optional

from solotab_utils import TUNINGS

# チューニングの最低音 (6弦開放)
TUNING_LOWEST = {name: notes[0] for name, notes in TUNINGS.items()}

# 日本語/英語ラベル
TUNING_LABELS = {
    "standard": "スタンダード (EADGBE)",
    "half_down": "半音下げ (Eb)",
    "full_down": "全音下げ (D)",
    "drop_d": "Drop D",
    "drop_c": "Drop C",
    "double_drop_d": "Double Drop D",
    "dadgad": "DADGAD",
    "open_d": "Open D",
    "open_e": "Open E",
    "open_g": "Open G",
    "open_a": "Open A",
    "open_c": "Open C",
}


def detect_tuning(notes: List[Dict], detected_key: str = None) -> Dict:
    """
    ノートのピッチ分布からチューニングを推定する。

    Parameters
    ----------
    notes : list[dict]
        検出されたノートリスト (pitch キー必須)
    detected_key : str, optional
        検出されたキー

    Returns
    -------
    dict
        - tuning: str (チューニング名)
        - confidence: float
        - lowest_note: int (検出された最低MIDI)
        - alternatives: list
    """
    if not notes:
        return {"tuning": "standard", "confidence": 0.0, "lowest_note": 40,
                "alternatives": []}

    pitches = [n["pitch"] for n in notes if "pitch" in n]
    if not pitches:
        return {"tuning": "standard", "confidence": 0.0, "lowest_note": 40,
                "alternatives": []}

    # 最低ピッチ (下位5%の平均でノイズを緩和)
    sorted_pitches = sorted(pitches)
    low_5pct = sorted_pitches[:max(1, len(sorted_pitches) // 20)]
    lowest_avg = np.mean(low_5pct)
    
    # 各チューニングとのマッチスコア
    candidates = []
    for name, tuning_notes in TUNINGS.items():
        lowest_open = tuning_notes[0]
        
        # スコア計算
        score = 0.0
        
        # 1. 最低音との一致 (最重要)
        lowest_diff = abs(lowest_avg - lowest_open)
        if lowest_diff <= 0.5:
            score += 5.0  # 完全一致
        elif lowest_diff <= 1.5:
            score += 3.0  # 半音差
        elif lowest_diff <= 2.5:
            score += 1.0  # 全音差
        else:
            score -= lowest_diff * 0.5  # 遠いほどペナルティ
        
        # 2. 開放弦ピッチの出現率
        open_hits = 0
        for open_pitch in tuning_notes:
            # 開放弦の音(とそのオクターブ)がノート中に多いか
            count = sum(1 for p in pitches if p % 12 == open_pitch % 12)
            if count > 0:
                open_hits += min(count / len(pitches) * 10, 1.0)
        score += open_hits * 0.5
        
        # 3. スタンダードチューニングのバイアス (最も一般的)
        if name == "standard":
            score += 1.0
        
        # 4. キーとの親和性
        if detected_key:
            key_tuning_affinity = _key_tuning_affinity(detected_key, name)
            score += key_tuning_affinity
        
        candidates.append({
            "tuning": name,
            "score": float(score),
            "label": TUNING_LABELS.get(name, name),
        })
    
    # スコアでソート
    candidates.sort(key=lambda c: -c["score"])
    
    best = candidates[0]
    
    # 確信度 (top1 - top2 の差に基づく)
    score_diff = candidates[0]["score"] - candidates[1]["score"] if len(candidates) > 1 else 5.0
    confidence = min(1.0, score_diff / 5.0)
    
    return {
        "tuning": best["tuning"],
        "confidence": float(confidence),
        "lowest_note": int(round(lowest_avg)),
        "label": best["label"],
        "alternatives": [
            {"tuning": c["tuning"], "label": c["label"], "score": round(c["score"], 2)}
            for c in candidates[:3]
        ],
    }


def _key_tuning_affinity(key: str, tuning_name: str) -> float:
    """キーとチューニングの親和性スコア"""
    # Drop/Open系チューニングと特定キーの組み合わせ
    affinities = {
        ("D", "drop_d"): 1.5,
        ("Dm", "drop_d"): 1.0,
        ("G", "drop_d"): 0.5,
        ("C", "drop_c"): 1.5,
        ("Cm", "drop_c"): 1.0,
        ("D", "open_d"): 2.0,
        ("Dm", "dadgad"): 1.5,
        ("E", "open_e"): 2.0,
        ("Em", "standard"): 0.5,
        ("G", "open_g"): 2.0,
        ("A", "open_a"): 2.0,
        ("C", "open_c"): 2.0,
    }
    return affinities.get((key, tuning_name), 0.0)


if __name__ == "__main__":
    # テスト
    test_notes = [
        {"pitch": 40}, {"pitch": 45}, {"pitch": 50},  # E2, A2, D3
        {"pitch": 55}, {"pitch": 59}, {"pitch": 64},  # G3, B3, E4
    ]
    result = detect_tuning(test_notes, detected_key="E")
    print(f"Tuning: {result['tuning']} ({result['label']})")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Lowest note: MIDI {result['lowest_note']}")
    print(f"Alternatives: {result['alternatives']}")
