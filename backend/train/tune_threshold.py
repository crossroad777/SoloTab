import os
import sys
import glob
import torch
import numpy as np
import mir_eval
from tqdm import tqdm

project_root = r'D:\Music\nextchord-solotab\backend'
sys.path.insert(0, project_root)
from benchmark_full import load_jams_notes, calculate_alignment_offset, format_for_mireval, AUDIO_DIR, ANNOTATIONS_DIR, GUITARSET_DIR
from benchmark_baseline import transcribe_single_model
from string_assigner import assign_strings_dp
from model import architecture

def run_threshold_sweep():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = r'D:\Music\nextchord-solotab\music-transcription\python\_processed_guitarset_data\training_output\ultimate_single_conformer\best_model.pth'
    
    model = architecture.GuitarTabCRNN(num_frames_rnn_input_dim=1280, rnn_type='GRU', rnn_hidden_size=768, rnn_layers=2, rnn_dropout=0.3, rnn_bidirectional=True)
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    jams_files = glob.glob(os.path.join(ANNOTATIONS_DIR, "*.jams"))
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    print("Starting Onset Threshold Sweep...")
    
    import copy
    
    results = {th: {"p": 0, "r": 0, "f": 0, "count": 0} for th in thresholds}
    
    for test_jams in tqdm(jams_files, desc="Processing tracks"):
        base_name = os.path.basename(test_jams).replace(".jams", "")
        wav_path = os.path.join(AUDIO_DIR, f"{base_name}_mix.wav")
        if not os.path.exists(wav_path):
             wav_path = os.path.join(GUITARSET_DIR, "audio_mono-pickup_mix", f"{base_name}_mix.wav")
             if not os.path.exists(wav_path): continue
             
        gt_notes = load_jams_notes(test_jams)
        if len(gt_notes) == 0: continue
        ref_int, ref_p = format_for_mireval(gt_notes)
        
        # We only need to run the model ONCE per track, then we can apply different thresholds on the probabilities!
        # Wait, transcribe_single_model applies threshold inside. 
        # Let's extract features and run model once to save time.
        import librosa
        import config
        y, sr = librosa.load(wav_path, sr=config.SAMPLE_RATE, mono=True)
        cqt_spec = librosa.cqt(
            y=y, sr=sr, hop_length=config.HOP_LENGTH,
            fmin=config.FMIN_CQT, n_bins=config.N_BINS_CQT, bins_per_octave=config.BINS_PER_OCTAVE_CQT
        )
        log_cqt = librosa.amplitude_to_db(np.abs(cqt_spec), ref=np.max)
        features = torch.tensor(log_cqt, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            onset_logits, fret_logits = model(features)
            onset_probs = torch.sigmoid(onset_logits[0]).cpu().numpy()
            fret_probs = torch.softmax(fret_logits[0], dim=-1).cpu().numpy()
        fret_preds_full = np.argmax(fret_probs, axis=-1)
        
        from guitar_transcriber import _frames_to_notes
        
        for th in thresholds:
            raw_notes = _frames_to_notes(onset_probs, fret_preds_full, tuning_pitches=None, onset_threshold=th)
            assigned_notes = assign_strings_dp(raw_notes)
            
            if len(assigned_notes) == 0:
                continue
                
            aligned_notes = copy.deepcopy(assigned_notes)
            offset = calculate_alignment_offset(gt_notes, aligned_notes, resolution=0.01)
            for n in aligned_notes:
                n['start'] = max(0.0, n['start'] + offset)
                n['end'] = max(n['start'] + 0.01, n['end'] + offset)
            
            est_int, est_p = format_for_mireval(aligned_notes)
            p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
                ref_int, ref_p, est_int, est_p,
                onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None
            )
            
            results[th]["p"] += p
            results[th]["r"] += r
            results[th]["f"] += f
            results[th]["count"] += 1

    print("\n=== Threshold Sweep Results (360 Tracks Average) ===")
    for th in thresholds:
        count = results[th]["count"]
        if count > 0:
            avg_p = results[th]["p"] / len(jams_files)
            avg_r = results[th]["r"] / len(jams_files)
            avg_f = results[th]["f"] / len(jams_files)
            print(f"Threshold: {th:.1f} -> P: {avg_p:.4f}, R: {avg_r:.4f}, F1: {avg_f:.4f}")
            
if __name__ == '__main__':
    run_threshold_sweep()
