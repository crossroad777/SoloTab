import time
import torch
import numpy as np
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
mt_python_dir = os.path.join(project_root, "music-transcription", "python")
if mt_python_dir not in sys.path:
    sys.path.insert(0, mt_python_dir)

from training import note_conversion_utils
from evaluation import metrics
import config

# Create dummy data for 100 sequences of 430 frames
num_seqs = 100
num_frames = 430
onset_preds = (torch.rand(num_seqs, num_frames, 6) > 0.9).float()
fret_preds = torch.randint(0, 21, (num_seqs, num_frames, 6))

print("Testing frames_to_notes_for_eval...")
start = time.time()
converted = []
for i in range(num_seqs):
    res = note_conversion_utils.frames_to_notes_for_eval(
        onset_preds[i],
        fret_preds[i],
        frame_hop_length=config.HOP_LENGTH,
        audio_sample_rate=config.SAMPLE_RATE,
        max_fret_value=config.MAX_FRETS,
    )
    converted.append(res)
print(f"frames_to_notes_for_eval time for {num_seqs} seqs: {time.time() - start:.4f} s")

# Dummy raw labels
def make_dummy_raw_labels():
    labels = []
    for _ in range(20): # 20 notes
        start_t = np.random.rand() * 5.0
        end_t = start_t + 0.2
        string = np.random.randint(0, 6)
        fret = np.random.randint(0, 20)
        labels.append([start_t, end_t, string, fret, 0])
    return torch.tensor(labels, dtype=torch.float32)

dummy_raws = [make_dummy_raw_labels() for _ in range(num_seqs)]

print("Testing calculate_note_level_metrics...")
start = time.time()
for p, raw in zip(converted, dummy_raws):
    metrics.calculate_note_level_metrics(p, raw)
print(f"calculate_note_level_metrics time for {num_seqs} seqs: {time.time() - start:.4f} s")

print("Testing calculate_onset_event_metrics...")
start = time.time()
for p, raw in zip(converted, dummy_raws):
    metrics.calculate_onset_event_metrics(p, raw)
print(f"calculate_onset_event_metrics time for {num_seqs} seqs: {time.time() - start:.4f} s")
