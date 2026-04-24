import time
import torch
import numpy as np
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
mt_python_dir = os.path.join(project_root, "music-transcription", "python")
if mt_python_dir not in sys.path:
    sys.path.insert(0, mt_python_dir)

import config

def frames_to_notes_numpy(
        onset_preds_binary_frames,
        fret_pred_indices_frames,
        frame_hop_length,
        audio_sample_rate,
        max_fret_value=config.MAX_FRETS,
        min_note_duration_frames=config.MIN_NOTE_DURATION_FRAMES,
        open_string_pitches=None
):
    if open_string_pitches is None:
        open_string_pitches = config.OPEN_STRING_PITCHES_MIDI

    # Convert to numpy to avoid torch .item() overhead
    if isinstance(onset_preds_binary_frames, torch.Tensor):
        onset_preds_binary_frames = onset_preds_binary_frames.cpu().numpy()
    if isinstance(fret_pred_indices_frames, torch.Tensor):
        fret_pred_indices_frames = fret_pred_indices_frames.cpu().numpy()

    num_frames, num_strings = onset_preds_binary_frames.shape
    time_per_frame = frame_hop_length / audio_sample_rate
    predicted_notes_list = []

    silence_fret_class_idx = max_fret_value + config.FRET_SILENCE_CLASS_OFFSET

    # Loop exactly as before but on numpy arrays
    for string_idx in range(num_strings):
        active_note_start_frame = None
        active_note_fret_val = None

        onset_col = onset_preds_binary_frames[:, string_idx]
        fret_col = fret_pred_indices_frames[:, string_idx]

        for frame_idx in range(num_frames):
            is_onset_active = onset_col[frame_idx] > 0.5
            current_fret_val = fret_col[frame_idx]

            note_should_terminate = False
            if active_note_start_frame is not None:
                if is_onset_active and frame_idx > active_note_start_frame:
                    note_should_terminate = True
                elif current_fret_val == silence_fret_class_idx:
                    note_should_terminate = True
                elif current_fret_val != active_note_fret_val:
                    note_should_terminate = True
                elif frame_idx == num_frames - 1:
                    note_should_terminate = True

                if note_should_terminate:
                    start_time_sec = active_note_start_frame * time_per_frame
                    end_time_sec = frame_idx * time_per_frame
                    duration_in_frames = frame_idx - active_note_start_frame

                    if duration_in_frames >= min_note_duration_frames and active_note_fret_val != silence_fret_class_idx:
                        if 0 <= active_note_fret_val <= max_fret_value:
                            pitch_midi_val = open_string_pitches[string_idx] + active_note_fret_val
                            predicted_notes_list.append({
                                'start_time': start_time_sec,
                                'end_time': end_time_sec,
                                'pitch_midi': int(round(pitch_midi_val)),
                                'string': string_idx,
                                'fret': int(active_note_fret_val)
                            })
                    active_note_start_frame = None
                    active_note_fret_val = None

            if is_onset_active and current_fret_val != silence_fret_class_idx:
                active_note_start_frame = frame_idx
                active_note_fret_val = current_fret_val

        # Handle last active note
        if active_note_start_frame is not None:
            start_time_sec = active_note_start_frame * time_per_frame
            end_time_sec = num_frames * time_per_frame
            duration_in_frames = num_frames - active_note_start_frame
            if duration_in_frames >= min_note_duration_frames and active_note_fret_val != silence_fret_class_idx:
                if 0 <= active_note_fret_val <= max_fret_value:
                    pitch_midi_val = open_string_pitches[string_idx] + active_note_fret_val
                    predicted_notes_list.append({
                        'start_time': start_time_sec,
                        'end_time': end_time_sec,
                        'pitch_midi': int(round(pitch_midi_val)),
                        'string': string_idx,
                        'fret': int(active_note_fret_val)
                    })
    return predicted_notes_list


num_seqs = 1000
num_frames = 430
onset_preds = (torch.rand(num_seqs, num_frames, 6) > 0.9).float()
fret_preds = torch.randint(0, 21, (num_seqs, num_frames, 6))

start = time.time()
converted = []
for i in range(num_seqs):
    res = frames_to_notes_numpy(
        onset_preds[i],
        fret_preds[i],
        frame_hop_length=config.HOP_LENGTH,
        audio_sample_rate=config.SAMPLE_RATE,
        max_fret_value=config.MAX_FRETS,
    )
    converted.append(res)
print(f"frames_to_notes_numpy time for {num_seqs} seqs: {time.time() - start:.4f} s")
