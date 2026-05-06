"""GAPS Multi-task スクリプト検証"""
import sys, os
sys.path.insert(0, r'd:\Music\nextchord-solotab\music-transcription\python')
os.chdir(r'd:\Music\nextchord-solotab\music-transcription\python')

# 1. Import check
print('=== 1. Import Check ===')
from data_processing.batching import collate_fn_pad
from data_processing.dataset import create_frame_level_labels
from training import loss_functions, epoch_processing
from model import architecture
import config
import pretty_midi
import torch
print('All imports OK')

# 2. pitch_to_string_fret unit tests
print('\n=== 2. pitch_to_string_fret Unit Tests ===')
sys.path.insert(0, r'd:\Music\nextchord-solotab\backend\train')
from train_gaps_multitask import parse_midi_to_raw_labels, pitch_to_string_fret

test_cases = [
    (40, (0, 0)),   # E2 = string 0, fret 0
    (45, (1, 0)),   # A2 = string 1, fret 0
    (64, (5, 0)),   # E4 = string 5, fret 0
    (60, (4, 1)),   # C4 = string 4, fret 1
    (52, (2, 2)),   # E3+2 = string 2, fret 2
]
for pitch, expected in test_cases:
    result = pitch_to_string_fret(pitch)
    status = 'PASS' if result == expected else 'FAIL'
    print(f'  pitch={pitch}: expected={expected}, got={result} [{status}]')

r_low = pitch_to_string_fret(30)
r_high = pitch_to_string_fret(90)
print(f'  pitch=30 (below range): {r_low} [{"PASS" if r_low is None else "FAIL"}]')
print(f'  pitch=90 (above range): {r_high} [{"PASS" if r_high is None else "FAIL"}]')

# 3. Real MIDI file parsing
print('\n=== 3. Real GAPS MIDI Parse ===')
midi_path = r'D:\Music\datasets\GAPS_DATA\midi\001_mvswc.mid'
labels = parse_midi_to_raw_labels(midi_path)
print(f'  File: 001_mvswc.mid')
print(f'  Shape: {labels.shape}')
if labels.shape[0] > 0:
    print(f'  Notes: {labels.shape[0]}')
    print(f'  Duration: {labels[-1, 1]:.1f}s')
    print(f'  Pitch range: {labels[:, 4].min():.0f}-{labels[:, 4].max():.0f}')
    print(f'  String distribution: {dict(zip(*torch.unique(labels[:, 2].long(), return_counts=True)))}')
    print(f'  Fret range: {labels[:, 3].min():.0f}-{labels[:, 3].max():.0f}')
    print(f'  First 5 notes:')
    for i in range(min(5, labels.shape[0])):
        print(f'    [{labels[i,0]:.3f}s -> {labels[i,1]:.3f}s, str={labels[i,2]:.0f}, fret={labels[i,3]:.0f}, pitch={labels[i,4]:.0f}]')
else:
    print('  WARNING: No notes!')

# 4. CQT + label pipeline end-to-end (1 track)
print('\n=== 4. End-to-End Pipeline (1 track) ===')
import librosa
import numpy as np
wav_path = r'D:\Music\datasets\GAPS_DATA\audio\001_mvswc.wav'
print(f'  Loading audio: {wav_path}')
audio, _ = librosa.load(wav_path, sr=config.SAMPLE_RATE, mono=True)
print(f'  Audio: {len(audio)} samples, {len(audio)/config.SAMPLE_RATE:.1f}s')

cqt = librosa.cqt(y=audio, sr=config.SAMPLE_RATE, hop_length=config.HOP_LENGTH,
                   fmin=config.FMIN_CQT, n_bins=config.N_BINS_CQT,
                   bins_per_octave=config.BINS_PER_OCTAVE_CQT)
log_cqt = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
features = torch.tensor(log_cqt, dtype=torch.float32)
print(f'  CQT features: {features.shape}')  # should be [168, T]

onset_targets, fret_targets = create_frame_level_labels(
    labels, features, config.HOP_LENGTH, config.SAMPLE_RATE, config.MAX_FRETS
)
print(f'  onset_targets: {onset_targets.shape}')  # should be [T, 6]
print(f'  fret_targets: {fret_targets.shape}')    # should be [T, 6]
print(f'  onset sum per string: {onset_targets.sum(dim=0).tolist()}')
total_onsets = onset_targets.sum().item()
print(f'  total onsets: {total_onsets:.0f} (MIDI notes: {labels.shape[0]})')

# 5. Model forward pass compatibility
print('\n=== 5. Model Forward Pass ===')
# Simulate batch: [batch=1, channels=1, freq=168, time=T]
T = features.shape[-1]
x = features.unsqueeze(0).unsqueeze(0)  # [1, 1, 168, T]
print(f'  Input shape: {x.shape}')

with torch.no_grad():
    cnn_out = architecture.TabCNN()(torch.randn(1, 1, config.N_BINS_CQT, 32))
    cnn_out_dim = cnn_out.shape[1] * cnn_out.shape[2]
    print(f'  CNN output dim: {cnn_out_dim}')

print('\n=== ALL CHECKS COMPLETE ===')
