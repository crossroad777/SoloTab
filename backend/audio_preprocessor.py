"""
audio_preprocessor.py — 音声前処理 (Domain Adaptation)
======================================================
YouTube音源をGuitarSet-likeな音質に近づける前処理。

1. ラウドネス正規化 (peak normalize)
2. ハイパスフィルタ (100Hz以下カット — 低域ノイズ除去)
3. EQ補正 (低音域-6dB, メロディ帯域+3dB — メロディ検出改善)
4. ジェントル圧縮 (ダイナミックレンジ縮小)

旧 ensemble_transcriber.py L29-91 から抽出。
"""

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt


def preprocess_audio_for_transcription(wav_path: str, output_path: str = None) -> str:
    """
    YouTube音源をGuitarSet-likeな音質に近づける前処理。
    """
    if output_path is None:
        output_path = wav_path  # in-place

    audio, sr = librosa.load(wav_path, sr=22050, mono=True)

    # 1. Peak normalization
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95

    # 2. High-pass filter at 100Hz (ギターの6弦E2=82Hzは通す、それ以下のノイズは除去)
    sos = butter(4, 100, btype='highpass', fs=22050, output='sos')
    audio = sosfilt(sos, audio).astype(np.float32)

    # 3. EQ補正 — メロディ帯域を強調、低音域を抑制
    # Low shelf: 250Hz以下を -6dB (0.5倍)
    sos_low = butter(2, 250, btype='lowpass', fs=22050, output='sos')
    low_band = sosfilt(sos_low, audio).astype(np.float32)
    audio = audio - low_band * 0.5  # 低音域を50%カット (-6dB)

    # High shelf: 1kHz以上を +3dB (1.41倍)
    sos_high = butter(2, 1000, btype='highpass', fs=22050, output='sos')
    high_band = sosfilt(sos_high, audio).astype(np.float32)
    audio = audio + high_band * 0.41  # 高音域を41%ブースト (+3dB)

    # Re-normalize after EQ
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95

    # 4. Gentle compression (reduce dynamic range → helps model detect quiet notes)
    threshold = 0.3
    ratio = 3.0
    abs_audio = np.abs(audio)
    mask = abs_audio > threshold
    compressed = audio.copy()
    compressed[mask] = np.sign(audio[mask]) * (
        threshold + (abs_audio[mask] - threshold) / ratio
    )
    audio = compressed

    # Re-normalize after compression
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95

    sf.write(output_path, audio, 22050)
    return output_path
