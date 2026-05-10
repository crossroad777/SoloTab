"""
key_analyzer.py — キー検出 & ポジション推定
============================================
librosaのchroma特徴量を使ってキーを検出し、
ギターの自然なポジションを推定する。

Essentiaが使えない環境(Python 3.12)のため、librosaで代替実装。
"""

import numpy as np
import librosa

# キー→最適ポジション中心のマッピング
# ギターの自然なポジションを考慮
KEY_TO_POSITION = {
    # メジャーキー
    "C":  0,   # 開放弦中心
    "C#": 1,
    "D":  0,   # Drop DやOpen Dで開放弦
    "Eb": 1,
    "E":  0,   # 開放弦中心 (最も自然)
    "F":  1,
    "F#": 2,
    "G":  3,   # 3フレット中心
    "Ab": 4,
    "A":  0,   # 開放弦 (5弦開放=A)
    "Bb": 1,
    "B":  2,
    # マイナーキー (相対メジャーと同じポジション)
    "Cm":  3,
    "C#m": 4,
    "Dm":  0,
    "Ebm": 1,
    "Em":  0,  # 開放弦中心
    "Fm":  1,
    "F#m": 2,
    "Gm":  3,
    "Abm": 4,
    "Am":  0,  # 開放弦中心
    "Bbm": 1,
    "Bm":  2,
}

# Krumhansl-Kessler key profile
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

NOTE_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']


def detect_key(wav_path: str, sr: int = 22050) -> dict:
    """
    WAVファイルからキーを検出する。
    
    Returns
    -------
    dict with keys:
        - key: str (e.g. "Am", "G")
        - confidence: float (0-1)
        - position: int (推奨ポジション中心フレット)
        - all_keys: list of (key, score) top 3
    """
    # 音声読み込み
    y, sr = librosa.load(wav_path, sr=sr, mono=True)
    
    # Chroma特徴量を計算
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
    chroma_mean = np.mean(chroma, axis=1)  # 12次元のピッチクラス分布
    chroma_mean = chroma_mean / (np.sum(chroma_mean) + 1e-10)  # 正規化
    
    # 全24キー(12メジャー+12マイナー)との相関を計算
    results = []
    
    for i in range(12):
        # メジャーキー
        major_profile = np.roll(MAJOR_PROFILE, i)
        major_profile = major_profile / np.sum(major_profile)
        major_corr = np.corrcoef(chroma_mean, major_profile)[0, 1]
        results.append((NOTE_NAMES[i], major_corr))
        
        # マイナーキー
        minor_profile = np.roll(MINOR_PROFILE, i)
        minor_profile = minor_profile / np.sum(minor_profile)
        minor_corr = np.corrcoef(chroma_mean, minor_profile)[0, 1]
        results.append((NOTE_NAMES[i] + "m", minor_corr))
    
    # スコア順にソート
    results.sort(key=lambda x: -x[1])
    
    best_key = results[0][0]
    best_score = results[0][1]
    
    # 推奨ポジション
    position = KEY_TO_POSITION.get(best_key, 0)
    
    return {
        "key": best_key,
        "confidence": float(best_score),
        "position": position,
        "all_keys": [(k, float(s)) for k, s in results[:5]],
    }


def get_initial_position_for_key(key: str) -> float:
    """キーからギターの初期ポジション中心を返す"""
    return float(KEY_TO_POSITION.get(key, 0))


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python key_analyzer.py <wav_path>")
        sys.exit(1)
    
    result = detect_key(sys.argv[1])
    print(f"Key: {result['key']} (confidence: {result['confidence']:.3f})")
    print(f"Recommended position: fret {result['position']}")
    print(f"Top 5: {result['all_keys']}")
