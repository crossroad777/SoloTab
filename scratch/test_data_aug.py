import numpy as np
import sys
import librosa

sys.path.append('D:/Music/nextchord-solotab/music-transcription/python')

import config
from data_processing.dataset import _augment_pink_noise, _augment_impulse_noise

def run_test():
    print("--- Data Augmentation Test ---")
    
    # 強制的に100%発動させる設定
    config.DATASET_TRAIN_AUGMENTATION_PINK_NOISE_PARAMS["probability"] = 1.0
    config.DATASET_TRAIN_AUGMENTATION_IMPULSE_PARAMS["probability"] = 1.0
    
    # ダミー音声 (1秒間の440Hzサイン波)
    sr = 22050
    audio_data = librosa.tone(440.0, sr=sr, duration=1.0)
    
    print(f"Original Audio min/max: {np.min(audio_data):.4f} / {np.max(audio_data):.4f}")
    
    # ピンクノイズ印加
    audio_pink = _augment_pink_noise(audio_data, sr)
    print(f"After Pink Noise min/max: {np.min(audio_pink):.4f} / {np.max(audio_pink):.4f}")
    
    # インパルスノイズ印加
    audio_full_aug = _augment_impulse_noise(audio_pink, sr)
    print(f"After Impulse Noise min/max: {np.min(audio_full_aug):.4f} / {np.max(audio_full_aug):.4f}")
    
    if len(audio_data) == len(audio_full_aug):
        print("\n=> Test Passed! Augmentation functions are working successfully on NumPy arrays without shape errors or crashes.")
    else:
        print("\n=> Test Failed: Audio length changed.")

if __name__ == "__main__":
    run_test()
