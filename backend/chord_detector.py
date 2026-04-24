"""
chord_detector.py — コード検出 (librosa chroma テンプレートマッチング)
====================================================================
Demucs分離後のギター音声からコード進行を検出し、
TAB譜にコードシンボルを表示するためのデータを生成する。
"""

import numpy as np
import librosa
from typing import List, Dict, Tuple

# コードテンプレート (ピッチクラスセット)
# 各コードの構成音を12次元ベクトルで表現
CHORD_TEMPLATES = {}

# メジャーコード: root, major3rd, perfect5th
# マイナーコード: root, minor3rd, perfect5th
NOTES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

for i, note in enumerate(NOTES):
    # メジャー (0, 4, 7)
    template = np.zeros(12)
    template[i] = 1.0
    template[(i + 4) % 12] = 0.8
    template[(i + 7) % 12] = 0.8
    CHORD_TEMPLATES[note] = template / np.linalg.norm(template)

    # マイナー (0, 3, 7)
    template_m = np.zeros(12)
    template_m[i] = 1.0
    template_m[(i + 3) % 12] = 0.8
    template_m[(i + 7) % 12] = 0.8
    CHORD_TEMPLATES[note + 'm'] = template_m / np.linalg.norm(template_m)

    # セブンス (0, 4, 7, 10)
    template_7 = np.zeros(12)
    template_7[i] = 1.0
    template_7[(i + 4) % 12] = 0.7
    template_7[(i + 7) % 12] = 0.7
    template_7[(i + 10) % 12] = 0.5
    CHORD_TEMPLATES[note + '7'] = template_7 / np.linalg.norm(template_7)


def detect_chords(wav_path: str, beats: List[float] = None,
                  sr: int = 22050, hop_length: int = 512) -> List[Dict]:
    """
    音声からコード進行を検出する。

    Parameters
    ----------
    wav_path : str
        WAVファイルのパス
    beats : list[float], optional
        ビート位置(秒)。指定時はビート単位でコードを検出。

    Returns
    -------
    list[dict]
        各要素: {"start": float, "end": float, "chord": str, "confidence": float}
    """
    y, sr = librosa.load(wav_path, sr=sr, mono=True)
    
    # Chroma特徴量
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length)
    
    # ビート単位でchromaを平均化
    if beats and len(beats) >= 2:
        segments = []
        for i in range(len(beats) - 1):
            start_t = beats[i]
            end_t = beats[i + 1]
            # この区間のフレームを取得
            mask = (times >= start_t) & (times < end_t)
            if np.any(mask):
                avg_chroma = np.mean(chroma[:, mask], axis=1)
            else:
                avg_chroma = np.zeros(12)
            segments.append((start_t, end_t, avg_chroma))
    else:
        # ビートなし: 固定間隔(0.5秒)
        interval = 0.5
        duration = len(y) / sr
        segments = []
        for t in np.arange(0, duration, interval):
            end_t = min(t + interval, duration)
            mask = (times >= t) & (times < end_t)
            if np.any(mask):
                avg_chroma = np.mean(chroma[:, mask], axis=1)
            else:
                avg_chroma = np.zeros(12)
            segments.append((t, end_t, avg_chroma))

    # 各セグメントでテンプレートマッチング
    chords = []
    for start_t, end_t, seg_chroma in segments:
        norm = np.linalg.norm(seg_chroma)
        if norm < 0.01:
            chords.append({"start": float(start_t), "end": float(end_t),
                          "chord": "N.C.", "confidence": 0.0})
            continue

        seg_norm = seg_chroma / norm
        best_chord = "N.C."
        best_score = -1.0

        for name, template in CHORD_TEMPLATES.items():
            score = float(np.dot(seg_norm, template))
            if score > best_score:
                best_score = score
                best_chord = name

        chords.append({
            "start": float(start_t),
            "end": float(end_t),
            "chord": best_chord,
            "confidence": float(best_score),
        })

    # 連続する同一コードをマージ
    merged = []
    for c in chords:
        if merged and merged[-1]["chord"] == c["chord"]:
            merged[-1]["end"] = c["end"]
        else:
            merged.append(dict(c))

    # 低確信度のN.C.を前のコードで埋める
    for i in range(1, len(merged)):
        if merged[i]["chord"] == "N.C." and merged[i]["confidence"] < 0.3:
            merged[i]["chord"] = merged[i - 1]["chord"]

    # 再度マージ
    final = [merged[0]] if merged else []
    for c in merged[1:]:
        if final and final[-1]["chord"] == c["chord"]:
            final[-1]["end"] = c["end"]
        else:
            final.append(c)

    return final


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python chord_detector.py <wav_path>")
        sys.exit(1)

    chords = detect_chords(sys.argv[1])
    print(f"Detected {len(chords)} chord changes:")
    for c in chords[:20]:
        print(f"  {c['start']:.1f}-{c['end']:.1f}s: {c['chord']} ({c['confidence']:.2f})")
