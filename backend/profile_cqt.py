import time
import librosa
import numpy as np
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
mt_python_dir = os.path.join(project_root, "music-transcription", "python")
if mt_python_dir not in sys.path:
    sys.path.insert(0, mt_python_dir)

import config

def process_waveform(waveform):
    cqt_spec = librosa.cqt(
        y=waveform, sr=config.SAMPLE_RATE, hop_length=config.HOP_LENGTH,
        fmin=config.FMIN_CQT, n_bins=config.N_BINS_CQT, bins_per_octave=config.BINS_PER_OCTAVE_CQT
    )
    log_cqt_spec = librosa.amplitude_to_db(np.abs(cqt_spec), ref=np.max)
    return log_cqt_spec

# Create 5s dummy audio at config.SAMPLE_RATE
y = np.random.randn(5 * config.SAMPLE_RATE).astype(np.float32)

print(f"Sample Rate: {config.SAMPLE_RATE}, Hop Length: {config.HOP_LENGTH}, Bins: {config.N_BINS_CQT}")

# Test 1: HPSS alone
start = time.time()
y_harmonic, y_percussive = librosa.effects.hpss(y)
hpss_time = time.time() - start
print(f"HPSS time: {hpss_time:.3f} s")

# Test 2: CQT alone
start = time.time()
cqt_orig = process_waveform(y)
cqt_time = time.time() - start
print(f"CQT calculation time (1 waveform): {cqt_time:.3f} s")

print(f"Total calculation time for 1 audio file (3 CQTs + HPSS): {hpss_time + cqt_time * 3:.3f} s")
