import sys
import os
import torch
import numpy as np
import mir_eval

project_root = r'D:\Music\nextchord-solotab\backend'
sys.path.insert(0, project_root)
from benchmark_full import load_jams_notes, calculate_alignment_offset, format_for_mireval, AUDIO_DIR, ANNOTATIONS_DIR, GUITARSET_DIR
from benchmark_baseline import transcribe_single_model
from string_assigner import assign_strings_dp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = r'D:\Music\nextchord-solotab\music-transcription\python\_processed_guitarset_data\training_output\ultimate_single_conformer\best_model.pth'
from model import architecture
model = architecture.GuitarTabCRNN(num_frames_rnn_input_dim=1280, rnn_type='GRU', rnn_hidden_size=768, rnn_layers=2, rnn_dropout=0.3, rnn_bidirectional=True)
state_dict = torch.load(model_path, map_location=device, weights_only=False)
if list(state_dict.keys())[0].startswith('module.'):
    state_dict = {k[7:]: v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.to(device)
model.eval()

target = '00_Funk1-97-C_solo'
jams_path = os.path.join(ANNOTATIONS_DIR, f'{target}.jams')
wav_path = os.path.join(AUDIO_DIR, f'{target}_mix.wav')
if not os.path.exists(wav_path):
    wav_path = os.path.join(GUITARSET_DIR, 'audio_mono-pickup_mix', f'{target}_mix.wav')

gt_notes = load_jams_notes(jams_path)
raw_notes = transcribe_single_model(wav_path, model, device, onset_threshold=0.5)
assigned_notes = assign_strings_dp(raw_notes)

import copy
aligned_notes = copy.deepcopy(assigned_notes)
offset = calculate_alignment_offset(gt_notes, aligned_notes, resolution=0.01)
for n in aligned_notes:
    n['start'] = max(0.0, n['start'] + offset)
    n['end'] = max(n['start'] + 0.01, n['end'] + offset)

ref_int, ref_p = format_for_mireval(gt_notes)
est_int, est_p = format_for_mireval(aligned_notes)

matching = mir_eval.transcription.match_notes(ref_int, ref_p, est_int, est_p, onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None)
matched_ref = set([m[0] for m in matching])
matched_est = set([m[1] for m in matching])

missed_ref = [i for i in range(len(ref_p)) if i not in matched_ref]
false_est = [i for i in range(len(est_p)) if i not in matched_est]

print(f'\n--- Analysis for {target} ---')
print(f'Total GT notes: {len(gt_notes)}, Total Pred notes: {len(aligned_notes)}')
print(f'Missed (False Negatives): {len(missed_ref)} notes ({len(missed_ref)/len(gt_notes)*100:.1f}%)')
print(f'Ghost (False Positives): {len(false_est)} notes')

import collections
print("\n[Missed Notes (Ground Truth but not predicted)]")
for idx in missed_ref[:10]:
    n = gt_notes[idx]
    print(f"  Time: {n['start']:.2f}s, Pitch: {n['pitch']:.1f}, String: {n.get('string', '?')}")

print("\n[False Positives (Predicted but not in Ground Truth)]")
for idx in false_est[:10]:
    n = aligned_notes[idx]
    print(f"  Time: {n['start']:.2f}s, Pitch: {n['pitch']:.1f}, String: {n.get('string', '?')}")

