import sys
import os
import json
sys.path.insert(0, '..')
from string_assigner import _predict_string_probs

wav_path = 'mini_dataset/audio_mono-mic/02_Funk2-119-G_comp_mic.wav'
test_points = [
    (0.01, 48), # GT: s5 f3
    (0.01, 55), # GT: s3 f0
    (2.02, 53), # GT: s5 f8
    (2.30, 60), # GT: s4 f10
    (2.30, 65), # GT: s3 f10
]

for t, p in test_points:
    probs = _predict_string_probs(wav_path, t, p)
    print(f"Time: {t:.2f}s | Pitch: {p} | CNN probs: {probs}")
