import time
import torch
import numpy as np
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
mt_python_dir = os.path.join(project_root, "music-transcription", "python")
if mt_python_dir not in sys.path:
    sys.path.insert(0, mt_python_dir)

from data_processing.dataset import create_frame_level_labels
import config

# Create dummy input
num_frames = 430
features = torch.zeros(3, 192, num_frames)
# Let's say a track has 200 notes
raw_labels = []
for _ in range(200):
    start = np.random.rand() * 4.0
    end = start + 0.1
    string = np.random.randint(0, 6)
    fret = np.random.randint(0, 20)
    raw_labels.append([start, end, string, fret, 0])
raw_labels = torch.tensor(raw_labels, dtype=torch.float32)

start = time.time()
for _ in range(100):
    create_frame_level_labels(raw_labels, features, config.HOP_LENGTH, config.SAMPLE_RATE, config.MAX_FRETS)
print(f"Original create_frame_level_labels (100 times): {time.time()-start:.4f} s")

def fast_create_frame_level_labels(
    raw_annotation_tensor,
    feature_map_tensor,
    frame_hop_length,
    audio_sr,
    fret_max_value,
):
    num_audio_frames = feature_map_tensor.shape[-1]
    num_guitar_strings = config.DEFAULT_NUM_STRINGS
    
    onset_targets_matrix = torch.zeros((num_audio_frames, num_guitar_strings), dtype=torch.float32)
    fret_targets_matrix = torch.full(
        (num_audio_frames, num_guitar_strings),
        fret_max_value + config.FRET_SILENCE_CLASS_OFFSET,
        dtype=torch.long,
    )
    time_duration_per_frame = frame_hop_length / audio_sr
    
    if raw_annotation_tensor is not None and raw_annotation_tensor.numel() > 0:
        raw_np = raw_annotation_tensor.numpy()
        onsets = np.clip(np.round(raw_np[:, 0] / time_duration_per_frame).astype(int), 0, num_audio_frames - 1)
        offsets = np.clip(np.round(raw_np[:, 1] / time_duration_per_frame).astype(int), 0, num_audio_frames - 1)
        strings = raw_np[:, 2].astype(int)
        frets = raw_np[:, 3].astype(int)
        
        valid = (strings >= 0) & (strings < num_guitar_strings)
        onsets = onsets[valid]
        offsets = offsets[valid]
        strings = strings[valid]
        frets = frets[valid]
        
        encoded_frets = np.where(frets >= 0, np.minimum(frets, fret_max_value), fret_max_value + config.FRET_SILENCE_CLASS_OFFSET)
        
        # Vectorized assignments
        onset_targets_matrix[onsets, strings] = 1.0
        
        for onset, offset, string, fret in zip(onsets, offsets, strings, encoded_frets):
            fret_targets_matrix[onset:offset+1, string] = fret
            
    return onset_targets_matrix, fret_targets_matrix

start = time.time()
for _ in range(100):
    fast_create_frame_level_labels(raw_labels, features, config.HOP_LENGTH, config.SAMPLE_RATE, config.MAX_FRETS)
print(f"Fast create_frame_level_labels (100 times): {time.time()-start:.4f} s")
