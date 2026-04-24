"""
capo_detector.py — カポ推定
================================
検出されたキーとコード進行から、カポ使用の可能性を推定する。

原理:
  ギタリストがカポを使う主な理由:
  1. 開放弦コード(C, G, D, A, E, Am, Em, Dm)で弾きたい
  2. シャープ/フラット系のキーを避けたい
  3. 弦楽器特有の響きを得たい

  例: Key=Bb → カポ1(実質Key=A) or カポ3(実質Key=G)
      Key=Eb → カポ1(実質Key=D) or カポ3(実質Key=C)
"""

from typing import Dict, Optional, List

# 開放弦で弾きやすいキーの「自然さスコア」
# スコアが高いほど、ギタリストがそのキーで弾く確率が高い
OPEN_KEY_SCORES = {
    "C": 0.9, "G": 1.0, "D": 0.95, "A": 0.9, "E": 1.0,
    "Am": 0.95, "Em": 1.0, "Dm": 0.85,
    "F": 0.5,  # バレーコードが必要だが一般的
}

# カポ位置→実質キーのシフト(半音数)
# カポN → 実際のキーは N半音上
CAPO_SHIFT = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}

NOTE_TO_SEMITONE = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

SEMITONE_TO_NOTE = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def _key_to_semitone(key: str) -> int:
    """キー名をセミトーンに変換 (Am → 9, G → 7)"""
    is_minor = key.endswith("m")
    root = key[:-1] if is_minor else key
    return NOTE_TO_SEMITONE.get(root, 0)


def _semitone_to_key(semitone: int, is_minor: bool) -> str:
    """セミトーン → キー名"""
    note = SEMITONE_TO_NOTE[semitone % 12]
    return note + ("m" if is_minor else "")


def detect_capo(detected_key: str, notes: list = None,
                confidence: float = 0.5) -> Dict:
    """
    カポ使用の推定。

    Parameters
    ----------
    detected_key : str
        検出されたキー (e.g. "Bb", "F#m")
    notes : list, optional
        ノートリスト (フレット分布の分析用)
    confidence : float
        キー検出の確信度

    Returns
    -------
    dict
        - capo: int (推定カポ位置, 0=カポなし)
        - effective_key: str (カポ込みの実質キー)
        - confidence: float (推定の確信度)
        - alternatives: list[dict] (他の候補)
    """
    is_minor = detected_key.endswith("m")
    root_semitone = _key_to_semitone(detected_key)

    # 各カポ位置での「自然さ」を評価
    candidates = []
    for capo in range(8):  # カポ0-7
        # カポ位置での実質キー = detected_key - capo
        effective_semitone = (root_semitone - capo) % 12
        effective_key = _semitone_to_key(effective_semitone, is_minor)

        # 開放弦キーとしての自然さ
        naturalness = OPEN_KEY_SCORES.get(effective_key, 0.1)

        # カポペナルティ (カポ位置が高いほどペナルティ)
        capo_penalty = capo * 0.08

        # フレット分布分析 (notesがある場合)
        fret_bonus = 0.0
        if notes and capo > 0:
            # カポ位置をフレット0として考えた時の最小フレットを確認
            frets = [n.get("fret", 0) for n in notes if n.get("fret", 0) > 0]
            if frets:
                min_fret = min(frets)
                # カポ位置付近にノートが集中していればボーナス
                near_capo = sum(1 for f in frets if abs(f - capo) <= 2)
                fret_bonus = near_capo / len(frets) * 0.3

        score = naturalness - capo_penalty + fret_bonus

        candidates.append({
            "capo": capo,
            "effective_key": effective_key,
            "score": score,
            "naturalness": naturalness,
        })

    # スコアでソート
    candidates.sort(key=lambda c: -c["score"])

    best = candidates[0]

    # カポ0(カポなし)のスコアと比較
    no_capo_score = next(c["score"] for c in candidates if c["capo"] == 0)

    # カポなしより明らかに良い場合のみカポを推奨
    if best["capo"] > 0 and best["score"] - no_capo_score < 0.15:
        # 差が小さい場合はカポなし
        best = next(c for c in candidates if c["capo"] == 0)

    return {
        "capo": best["capo"],
        "effective_key": best["effective_key"],
        "confidence": float(min(best["score"], 1.0)),
        "alternatives": [
            {"capo": c["capo"], "key": c["effective_key"], "score": round(c["score"], 3)}
            for c in candidates[:3]
        ],
    }


if __name__ == "__main__":
    # テスト
    test_keys = ["C", "G", "D", "Am", "Em", "Bb", "F#m", "Eb", "Ab"]
    for key in test_keys:
        result = detect_capo(key)
        print(f"Key={key:4s} → Capo {result['capo']}, Effective: {result['effective_key']}, "
              f"Conf: {result['confidence']:.2f}, Alts: {result['alternatives']}")
