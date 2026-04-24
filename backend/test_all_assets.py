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

def extract_features(wav_path):
    import librosa
    import config
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
        onset_probs, fret_preds_full, tuning_pitches=None, onset_threshold=onset_threshold
    )
    return notes

def main():
    print("=== SoloTab V2.0: Massive Legacy Validation (World Benchmark) ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    models_to_test = [
        "baseline_model",
        "finetuned_martin_finger_model",
        "finetuned_taylor_finger_model",
        "finetuned_luthier_finger_model",
        "finetuned_martin_pick_model",
        "finetuned_taylor_pick_model",
        "finetuned_luthier_pick_model"
    ]
    
    jams_files = glob.glob(os.path.join(ANNOTATIONS_DIR, "*.jams"))
    if not jams_files:
        print("Error: Ground truth *.jams not found.")
        return
        
    report_path = os.path.join(project_root, "official_legacy_assets_sota_report.txt")
    with open(report_path, "w", encoding="utf-8") as rep:
        rep.write("=== World Standard Benchmark Across All Assets ===\n")

    for model_dir in models_to_test:
        model_name = model_dir.replace("finetuned_", "").replace("_model", "")
        model_path = os.path.join(mt_python_dir, "_processed_guitarset_data", "training_output", model_dir, "best_model.pth")
        
        if not os.path.exists(model_path):
            print(f"Skipping {model_name} (Not found: {model_path})")
            continue
            
        print(f"\n[Testing Asset: {model_name}]")
        
        model = architecture.GuitarTabCRNN(num_frames_rnn_input_dim=1280, rnn_type="GRU", rnn_hidden_size=768, rnn_layers=2, rnn_dropout=0.3, rnn_bidirectional=True)
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        metrics = {"tab_f": 0}
        count = 0
        
        for test_jams in tqdm(jams_files, desc=model_name, leave=False):
            base_name = os.path.basename(test_jams).replace(".jams", "")
            wav_path = os.path.join(AUDIO_DIR, f"{base_name}_mix.wav")
            if not os.path.exists(wav_path):
                 wav_path = os.path.join(GUITARSET_DIR, "audio_mono-pickup_mix", f"{base_name}_mix.wav")
                 if not os.path.exists(wav_path): continue
            
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
            
            _, _, f_tab = calc_scores(assigned_notes)
            metrics["tab_f"] += f_tab
            count += 1
            
        final_f1 = metrics["tab_f"] / count if count > 0 else 0
        print(f" -> Result for {model_name}: World Standard F1: {final_f1:.4f}")
        
        with open(report_path, "a", encoding="utf-8") as rep:
            rep.write(f"- {model_name:20s} : F1: {final_f1:.4f}\n")

if __name__ == "__main__":
    main()
