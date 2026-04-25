import os
import sys
import glob
import torch
import numpy as np
from tqdm import tqdm
try:
    import mir_eval
except ImportError:
    print("mir_eval is required.")
    sys.exit(1)

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
mt_python_dir = os.path.join(project_root, "..", "music-transcription", "python")
if mt_python_dir not in sys.path:
    sys.path.insert(0, mt_python_dir)

from string_assigner import assign_strings_dp
from benchmark_full import load_jams_notes, calculate_alignment_offset, format_for_mireval, AUDIO_DIR, ANNOTATIONS_DIR, GUITARSET_DIR

from model import architecture
from solotab_utils import _to_native

def extract_features(wav_path):
    import gc
    gc.collect()
    import librosa
    import config
    import torchaudio
    y, sr = librosa.load(wav_path, sr=config.SAMPLE_RATE, mono=True)
    cqt_spec = librosa.cqt(
        y=y, sr=sr, hop_length=config.HOP_LENGTH,
        fmin=config.FMIN_CQT, n_bins=config.N_BINS_CQT, bins_per_octave=config.BINS_PER_OCTAVE_CQT
    )
    log_cqt = librosa.amplitude_to_db(np.abs(cqt_spec), ref=np.max)
    features = torch.tensor(log_cqt, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return features

def transcribe_single_model(wav_path, model, device, onset_threshold=0.7):
    features = extract_features(wav_path).to(device)
    with torch.no_grad():
        onset_logits, fret_logits = model(features)
        onset_probs = torch.sigmoid(onset_logits[0]).cpu().numpy()
        fret_probs = torch.softmax(fret_logits[0], dim=-1).cpu().numpy()
    
    fret_preds_full = np.argmax(fret_probs, axis=-1)
    
    from guitar_transcriber import _frames_to_notes
    notes = _frames_to_notes(
        onset_probs, 
        fret_preds_full,
        tuning_pitches=None,
        onset_threshold=onset_threshold
    )
    return notes

def run_official_benchmark(model_instance=None, device=None, max_tracks=None):
    if not model_instance:
        print("=== SoloTab V2.0: Scratch Baseline Validation (Double Benchmark) ===")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not model_instance:
        print(f"Loading single baseline model on {device}...")
    
    # モデルのロード (Ultimate Single Conformer: 最新のEpoch状態を直接テスト)
    if model_instance is None:
        model_path = r"D:\Music\nextchord-solotab\music-transcription\python\_processed_guitarset_data\training_output\ultimate_single_conformer\best_model.pth"
        from model import architecture
        model = architecture.GuitarTabCRNN(num_frames_rnn_input_dim=1280, rnn_type="GRU", rnn_hidden_size=768, rnn_layers=2, rnn_dropout=0.3, rnn_bidirectional=True)
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
    else:
        model = model_instance
        model.eval()
    
    jams_files = glob.glob(os.path.join(ANNOTATIONS_DIR, "*.jams"))
    if not jams_files:
        print("Error: Ground truth *.jams not found.")
        return

    if max_tracks is not None:
        jams_files = jams_files[:max_tracks]

    results = {}
    metrics = {"raw_p": 0, "raw_r": 0, "raw_f": 0, "tab_p": 0, "tab_r": 0, "tab_f": 0}
    count = 0
    
    log_file = os.path.join(project_root, "benchmark_baseline_sota.log")
    with open(log_file, "w", encoding="utf-8") as f_log:
        f_log.write("=== Baseline Benchmark Log ===\n")
        
    for test_jams in tqdm(jams_files, desc="GuitarSet Official Eval", leave=False):
        base_name = os.path.basename(test_jams).replace(".jams", "")
        wav_path = os.path.join(AUDIO_DIR, f"{base_name}_mix.wav")
        if not os.path.exists(wav_path):
             wav_path = os.path.join(GUITARSET_DIR, "audio_mono-pickup_mix", f"{base_name}_mix.wav")
             if not os.path.exists(wav_path): continue
        
        parts = base_name.split("_")
        genre = "Unknown"
        if len(parts) > 1:
            genre_tag = parts[1].split("-")[0]
            import re
            genre = re.sub(r'\d+', '', genre_tag)
            
        gt_notes = load_jams_notes(test_jams)
        
        import copy
        def calc_scores(target_notes):
            if len(target_notes) == 0: return 0, 0, 0
            aligned_notes = copy.deepcopy(target_notes)
            offset = calculate_alignment_offset(gt_notes, aligned_notes, resolution=0.01)
            for n in aligned_notes:
                n['start'] = max(0.0, n['start'] + offset)
                n['end'] = max(n['start'] + 0.01, n['end'] + offset)
            ref_intervals, ref_pitches = format_for_mireval(gt_notes)
            est_intervals, est_pitches = format_for_mireval(aligned_notes)
            p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
                ref_intervals, ref_pitches, est_intervals, est_pitches,
                onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None
            )
            return p, r, f
        
        raw_notes = transcribe_single_model(wav_path, model, device, onset_threshold=0.7)
        assigned_notes = assign_strings_dp(raw_notes)
        
        p_raw, r_raw, f_raw = calc_scores(raw_notes)
        p_tab, r_tab, f_tab = calc_scores(assigned_notes)
        
        with open(log_file, "a", encoding="utf-8") as f_log:
            f_log.write(f"[{base_name}] RAW=[F:{f_raw:.4f}] TAB=[F:{f_tab:.4f}]\n")
            
        if genre not in results:
            results[genre] = {'raw_f': 0, 'tab_f': 0, 'count': 0}
        results[genre]['raw_f'] += f_raw
        results[genre]['tab_f'] += f_tab
        results[genre]['count'] += 1
        
        metrics["raw_p"] += p_raw
        metrics["raw_r"] += r_raw
        metrics["raw_f"] += f_raw
        metrics["tab_p"] += p_tab
        metrics["tab_r"] += r_tab
        metrics["tab_f"] += f_tab
        count += 1

    report_path = os.path.join(project_root, "official_baseline_sota_report.txt")
    with open(report_path, "w", encoding="utf-8") as rep:
        rep.write("=== SoloTab V2.0 SOTA Evaluation Report (Single Baseline) ===\n")
        rep.write(f"Total Tracks Tested: {count}\n\n")
        
        for g, data in results.items():
            if data['count'] > 0:
                avg_raw = data['raw_f'] / data['count']
                avg_tab = data['tab_f'] / data['count']
                rep.write(f"Genre: {g} ({data['count']} tracks) -> Raw Pitch: {avg_raw:.4f} | TAB Viterbi: {avg_tab:.4f}\n")
                
        if count > 0:
            rep.write("-" * 50 + "\n")
            rep.write(f"🏆 OVERALL AVERAGE (Macro)\n")
            rep.write(f"[RAW Pitch (AI Ear-Copy Limit)]\n")
            rep.write(f"  F1-Score : {metrics['raw_f']/count:.4f}\n")
            rep.write(f"  Precision: {metrics['raw_p']/count:.4f}\n")
            rep.write(f"  Recall   : {metrics['raw_r']/count:.4f}\n\n")
            rep.write(f"[TAB Playable (Viterbi Applied)]\n")
            rep.write(f"  F1-Score : {metrics['tab_f']/count:.4f}\n")
            rep.write(f"  Precision: {metrics['tab_p']/count:.4f}\n")
            rep.write(f"  Recall   : {metrics['tab_r']/count:.4f}\n")
            
            
            
    final_tab_f1 = metrics['tab_f']/count if count > 0 else 0.0
    if not model_instance:
        print(f"\nEvaluation completed. Results saved to {report_path}")
    return final_tab_f1

if __name__ == "__main__":
    run_official_benchmark()
