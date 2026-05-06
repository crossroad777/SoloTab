"""
chord_detector.py — コード検出 (BTC primary + chroma fallback)
====================================================================
BTC (Bi-directional Transformer for Chords) をプライマリ検出器として使用。
BTC が利用不可の場合は librosa chroma テンプレートマッチングにフォールバック。
"""

import numpy as np
import librosa
from typing import List, Dict, Tuple

# BTC ラベル → SoloTab 内部形式の変換テーブル
# BTC large_voca: 'C', 'C:min', 'C:7', 'C:min7', 'N' etc.
# SoloTab:        'C', 'Cm',    'C7',  'Cm7',     'N.C.' etc.
def _btc_label_to_solotab(label: str) -> str:
    """BTC のコードラベルを SoloTab 形式に変換"""
    if label in ('N', 'X'):
        return 'N.C.'
    # 'A:min' -> 'Am', 'E:min7' -> 'Em7', 'G:maj' -> 'G'
    label = label.replace(':maj7', 'maj7')
    label = label.replace(':maj', '')
    label = label.replace(':min7', 'm7')
    label = label.replace(':min', 'm')
    label = label.replace(':7', '7')
    label = label.replace(':dim', 'dim')
    label = label.replace(':aug', 'aug')
    label = label.replace(':sus4', 'sus4')
    label = label.replace(':sus2', 'sus2')
    label = label.replace(':hdim7', 'm7b5')
    label = label.replace(':', '')  # 残りのコロンを除去
    return label


def detect_chords_btc(wav_path: str) -> List[Dict]:
    """
    BTC モデルによるコード検出。
    
    Returns
    -------
    list[dict] or None
        成功時: [{start, end, chord, confidence}], 失敗時: None
    """
    try:
        import sys, os
        # BTC エンジンのパスを追加
        btc_engine_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'nextchord', 'fastapi-backend'
        )
        # D:\Music\nextchord\fastapi-backend に btc_engine.py がある
        # ただし直接パスが異なる可能性があるので、複数パスを試す
        candidate_paths = [
            btc_engine_dir,
            r'D:\Music\nextchord\fastapi-backend',
        ]
        btc_dir = r'D:\Music\nextchord\BTC-ISMIR19'
        if btc_dir not in sys.path:
            sys.path.insert(0, btc_dir)
        
        for cp in candidate_paths:
            if cp not in sys.path and os.path.isdir(cp):
                sys.path.insert(0, cp)
        
        from btc_engine import get_btc_engine
        
        engine = get_btc_engine()
        seg_starts, seg_labels = engine.detect_chords(str(wav_path))
        
        if len(seg_starts) == 0:
            return None
        
        # (starts, labels) → [{start, end, chord}] 形式に変換
        chords = []
        for i in range(len(seg_starts)):
            start = float(seg_starts[i])
            end = float(seg_starts[i + 1]) if i + 1 < len(seg_starts) else start + 2.0
            label = _btc_label_to_solotab(str(seg_labels[i]))
            chords.append({
                'start': start,
                'end': end,
                'chord': label,
                'confidence': 0.85,  # BTC は全体的に高精度
                '_source': 'btc',
            })
        
        # N.C. を前のコードで埋める
        for i in range(1, len(chords)):
            if chords[i]['chord'] == 'N.C.':
                chords[i]['chord'] = chords[i - 1]['chord']
        
        # 連続同一コードをマージ
        merged = [chords[0]] if chords else []
        for c in chords[1:]:
            if merged and merged[-1]['chord'] == c['chord']:
                merged[-1]['end'] = c['end']
            else:
                merged.append(c)
        
        print(f"[chord_detector] BTC: {len(merged)} chord regions detected")
        return merged
    
    except Exception as e:
        print(f"[chord_detector] BTC unavailable: {e}")
        return None


# コードテンプレート (ピッチクラスセット) — chroma fallback 用
CHORD_TEMPLATES = {}
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


def detect_chords_chroma(wav_path: str, beats: List[float] = None,
                         sr: int = 22050, hop_length: int = 512) -> List[Dict]:
    """Chroma テンプレートマッチングによるコード検出 (フォールバック)"""
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
            mask = (times >= start_t) & (times < end_t)
            if np.any(mask):
                avg_chroma = np.mean(chroma[:, mask], axis=1)
            else:
                avg_chroma = np.zeros(12)
            segments.append((start_t, end_t, avg_chroma))
    else:
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

    print(f"[chord_detector] Chroma fallback: {len(final)} chord regions")
    return final


def detect_chords(wav_path: str, beats: List[float] = None,
                  sr: int = 22050, hop_length: int = 512) -> List[Dict]:
    """
    コード検出のメインエントリポイント。
    BTC をプライマリ、chroma をフォールバックとして使用。

    Parameters
    ----------
    wav_path : str
        WAVファイルのパス
    beats : list[float], optional
        ビート位置(秒)。chroma fallback 時に使用。

    Returns
    -------
    list[dict]
        各要素: {"start": float, "end": float, "chord": str, "confidence": float}
    """
    # 1. BTC を試行
    result = detect_chords_btc(wav_path)
    if result is not None:
        return result
    
    # 2. フォールバック: chroma テンプレートマッチング
    return detect_chords_chroma(wav_path, beats=beats, sr=sr, hop_length=hop_length)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python chord_detector.py <wav_path>")
        sys.exit(1)

    chords = detect_chords(sys.argv[1])
    print(f"Detected {len(chords)} chord changes:")
    for c in chords[:20]:
        src = c.get('_source', 'chroma')
        print(f"  {c['start']:.1f}-{c['end']:.1f}s: {c['chord']} ({c['confidence']:.2f}) [{src}]")
