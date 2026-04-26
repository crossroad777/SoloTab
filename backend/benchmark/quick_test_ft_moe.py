"""
GuitarSet FT済みモデル vs 元モデル Pure MoE 比較テスト（自己完結版）
"""
import os, sys, json, copy
import numpy as np
import scipy.signal
import torch
import librosa
from scipy import stats

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
mt_python_dir = os.path.join(project_root, "..", "music-transcription", "python")
sys.path.insert(0, project_root)
sys.path.insert(0, mt_python_dir)

import config
from model import architecture
from guitar_transcriber import _frames_to_notes

try:
    import mir_eval
except ImportError:
    print("mir_eval required"); sys.exit(1)

import glob

GUITARSET_DIR = r"D:\Music\Datasets\GuitarSet"
ANNOTATIONS_DIR = os.path.join(GUITARSET_DIR, "annotation")
AUDIO_DIR = os.path.join(GUITARSET_DIR, "audio_mono-pickup_mix")
TRAINING_OUTPUT = os.path.join(mt_python_dir, "_processed_guitarset_data", "training_output")


def load_jams_notes(jams_path):
    with open(jams_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    notes = []
    for ann in data.get('annotations', []):
        if ann.get('namespace') == 'note_midi':
            for d in ann.get('data', []):
                start = float(d.get('time', 0.0))
                dur = float(d.get('duration', 0.0))
                pitch = int(round(float(d.get('value', 0.0))))
                notes.append({"start": start, "end": start + dur, "pitch": pitch})
    return notes


def calculate_alignment_offset(gt_notes, pred_notes, resolution=0.01):
    if not gt_notes or not pred_notes: return 0.0
    max_time = max(max(n['start'] for n in gt_notes), max(n['start'] for n in pred_notes)) + 2.0
    vec_len = int(max_time / resolution)
    gt_vec = np.zeros(vec_len)
    pred_vec = np.zeros(vec_len)
    for n in gt_notes:
        idx = int(n['start'] / resolution)
        if idx < vec_len: gt_vec[idx] = 1.0
    for n in pred_notes:
        idx = int(n['start'] / resolution)
        if idx < vec_len: pred_vec[idx] = 1.0
    correlation = scipy.signal.correlate(gt_vec, pred_vec, mode='full')
    lags = scipy.signal.correlation_lags(gt_vec.size, pred_vec.size, mode='full')
    return lags[np.argmax(correlation)] * resolution


def transcribe_with_models(wav_path, model_dirs, device, vote_threshold=4, onset_threshold=0.8, vote_prob_threshold=0.5):
    y, sr = librosa.load(wav_path, sr=config.SAMPLE_RATE, mono=True)
    cqt_spec = librosa.cqt(y=y, sr=sr, hop_length=config.HOP_LENGTH,
        fmin=config.FMIN_CQT, n_bins=config.N_BINS_CQT, bins_per_octave=config.BINS_PER_OCTAVE_CQT)
    log_cqt = librosa.amplitude_to_db(np.abs(cqt_spec), ref=np.max)
    features = torch.tensor(log_cqt, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    all_onset_probs = []
    all_fret_preds = []

    for model_dir in model_dirs:
        model_path = os.path.join(TRAINING_OUTPUT, model_dir, "best_model.pth")
        if not os.path.exists(model_path): continue

        model = architecture.GuitarTabCRNN(
            num_frames_rnn_input_dim=1280, rnn_type="GRU",
            rnn_hidden_size=768, rnn_layers=2, rnn_dropout=0.3, rnn_bidirectional=True)
        sd = torch.load(model_path, map_location=device, weights_only=False)
        if list(sd.keys())[0].startswith("module."):
            sd = {k[7:]: v for k, v in sd.items()}
        model.load_state_dict(sd)
        model.to(device); model.eval()

        with torch.no_grad():
            onset_logits, fret_logits = model(features)
            onset_probs = torch.sigmoid(onset_logits[0]).cpu().numpy()
            fret_probs = torch.softmax(fret_logits[0], dim=-1).cpu().numpy()

        all_onset_probs.append(onset_probs)
        all_fret_preds.append(np.argmax(fret_probs, axis=-1))
        del model; torch.cuda.empty_cache()

    if not all_onset_probs: return []

    all_onset_probs = np.array(all_onset_probs)
    all_fret_preds = np.array(all_fret_preds)

    binary_votes = all_onset_probs > vote_prob_threshold
    vote_counts = np.sum(binary_votes, axis=0)
    consensus_onset_probs = np.max(all_onset_probs, axis=0)
    consensus_onset_probs[vote_counts < vote_threshold] = 0.0
    consensus_frets, _ = stats.mode(all_fret_preds, axis=0, keepdims=False)

    notes = _frames_to_notes(consensus_onset_probs, consensus_frets, tuning_pitches=None, onset_threshold=onset_threshold)
    for n in notes:
        n["start"] = float(n["start"])
        n["end"] = float(n["end"])
    return notes


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    jams_files = sorted(glob.glob(os.path.join(ANNOTATIONS_DIR, "*.jams")))[:5]
    print(f"Testing on {len(jams_files)} tracks, device={device}")

    models_original = [
        "finetuned_martin_finger_model", "finetuned_taylor_finger_model",
        "finetuned_luthier_finger_model", "finetuned_martin_pick_model",
        "finetuned_taylor_pick_model", "finetuned_luthier_pick_model",
    ]
    models_ft = []
    for m in models_original:
        ft_name = m.replace("_model", "_guitarset_ft")
        ft_path = os.path.join(TRAINING_OUTPUT, ft_name, "best_model.pth")
        models_ft.append(ft_name if os.path.exists(ft_path) else m)

    ft_count = sum(1 for m in models_ft if "guitarset_ft" in m)
    print(f"FT models available: {ft_count}/6")

    print("\n" + "="*60)
    print("  Pure MoE: ORIGINAL vs GuitarSet-FT")
    print("="*60)

    for label, mdirs in [("ORIGINAL", models_original), ("GS-FT  ", models_ft)]:
        scores = []
        for jams_path in jams_files:
            base = os.path.basename(jams_path).replace(".jams", "")
            wav_path = os.path.join(AUDIO_DIR, f"{base}_mix.wav")
            if not os.path.exists(wav_path): continue

            gt_notes = load_jams_notes(jams_path)
            if not gt_notes: continue

            pred_notes = transcribe_with_models(wav_path, mdirs, device)

            if pred_notes:
                aligned = copy.deepcopy(pred_notes)
                offset = calculate_alignment_offset(gt_notes, aligned)
                for n in aligned:
                    n['start'] = max(0.0, n['start'] + offset)
                    n['end'] = max(n['start'] + 0.01, n['end'] + offset)

                ref_i = np.array([[n['start'], n['end']] for n in gt_notes])
                ref_p = np.array([float(n['pitch']) for n in gt_notes])
                est_i = np.array([[n['start'], n['end']] for n in aligned])
                est_p = np.array([float(n['pitch']) for n in aligned])

                p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
                    ref_i, ref_p, est_i, est_p,
                    onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None)
            else:
                p, r, f = 0, 0, 0

            scores.append(f)
            print(f"  [{label}] {base[:25]:25s} F1={f:.4f} P={p:.4f} R={r:.4f}")

        if scores:
            print(f"  [{label}] AVG F1={np.mean(scores):.4f}")
        print()


if __name__ == "__main__":
    main()
