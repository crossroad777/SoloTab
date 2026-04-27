"""
SoloTab Configuration — HF Space版（推論専用）
"""
import librosa

# --- Audio Parameters ---
SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
MAX_FRETS = 20

# --- CQT Parameters ---
FMIN_CQT = librosa.note_to_hz('E2')  # 82.41 Hz
N_BINS_CQT = 168
BINS_PER_OCTAVE_CQT = 24

# --- Model Constants ---
CNN_INPUT_CHANNELS = 1
CNN_OUTPUT_CHANNELS_LIST_DEFAULT = [32, 64, 128, 128, 128]
CNN_KERNEL_SIZES_DEFAULT = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
CNN_STRIDES_DEFAULT = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]
CNN_PADDINGS_DEFAULT = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]
CNN_POOLING_KERNELS_DEFAULT = [(2,1), (2,1), (2,1), (2,1), (1,1)]
CNN_POOLING_STRIDES_DEFAULT = [(2,1), (2,1), (2,1), (2,1), (1,1)]
DEFAULT_NUM_STRINGS = 6
FRET_SILENCE_CLASS_OFFSET = 1

# --- String Tuning (Standard) ---
OPEN_STRING_PITCHES = {0: 40, 1: 45, 2: 50, 3: 55, 4: 59, 5: 64}

# --- Onset Detection ---
DEFAULT_ONSET_THRESHOLD = 0.5
MIN_NOTE_DURATION_FRAMES = 2
