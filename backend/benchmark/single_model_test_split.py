"""
single_model_test_split.py — 単一モデル(Ultimate Single Conformer)のTest分割のみ評価
"""
import os, sys, glob, json, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch, librosa, mir_eval

mt_python_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "music-transcription", "python")
sys.path.insert(0, mt_python_dir)
import config
from model import architecture
from guitar_transcriber import _frames_to_notes

GUITARSET_DIR = r"D:\Music\Datasets\GuitarSet"
ANNOTATIONS_DIR = os.path.join(GUITARSET_DIR, "annotation")
AUDIO_DIR = os.path.join(GUITARSET_DIR, "audio_mono-mic")
PROCESSED_DIR = os.path.join(mt_python_dir, "_processed_guitarset_data")

MODEL_PATH = os.path.join(PROCESSED_DIR, "training_output", "ultimate_single_conformer", "best_model.pth")
ONSET_THRESHOLD = 0.7


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


def to_mireval(notes):
    if not notes:
        return np.empty((0, 2)), np.empty(0)
    intervals = np.array([[n['start'], n['end']] for n in notes], dtype=float)
    pitches_hz = np.array([440.0 * (2.0 ** ((n['pitch'] - 69.0) / 12.0)) for n in notes], dtype=float)
    return intervals, pitches_hz


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading Ultimate Single Conformer on {device}...")
    model = architecture.GuitarTabCRNN(
        num_frames_rnn_input_dim=1280, rnn_type="GRU",
        rnn_hidden_size=768, rnn_layers=2, rnn_dropout=0.3, rnn_bidirectional=True
    )
    sd = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    if list(sd.keys())[0].startswith("module."):
        sd = {k[7:]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(device).eval()

    # Load splits
    for split_name in ["test", "validation", "train"]:
        split_path = os.path.join(PROCESSED_DIR, f"{split_name}_ids.txt")
        with open(split_path) as f:
            split_ids = set(l.strip() for l in f if l.strip())

        jams_files = sorted(glob.glob(os.path.join(ANNOTATIONS_DIR, "*.jams")))
        pairs = []
        for jp in jams_files:
            base = os.path.basename(jp).replace(".jams", "")
            if base in split_ids:
                # Try mic audio first, then mix
                wav = os.path.join(AUDIO_DIR, f"{base}_mic.wav")
                if not os.path.exists(wav):
                    wav = os.path.join(GUITARSET_DIR, "audio_mono-pickup_mix", f"{base}_mix.wav")
                if os.path.exists(wav):
                    pairs.append((jp, wav, base))

        print(f"\n{'='*60}")
        print(f" {split_name.upper()} split ({len(pairs)} tracks)")
        print(f"{'='*60}")

        f1s, ps, rs = [], [], []
        for i, (jp, wp, name) in enumerate(pairs):
            gt = load_jams_notes(jp)
            if not gt:
                continue

            y, sr = librosa.load(wp, sr=config.SAMPLE_RATE, mono=True)
            cqt = librosa.cqt(y=y, sr=sr, hop_length=config.HOP_LENGTH,
                              fmin=config.FMIN_CQT, n_bins=config.N_BINS_CQT,
                              bins_per_octave=config.BINS_PER_OCTAVE_CQT)
            log_cqt = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
            features = torch.tensor(log_cqt, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                onset_logits, fret_logits = model(features)
                onset_probs = torch.sigmoid(onset_logits[0]).cpu().numpy()
                fret_probs = torch.softmax(fret_logits[0], dim=-1).cpu().numpy()

            fret_preds = np.argmax(fret_probs, axis=-1)
            pred_notes = _frames_to_notes(onset_probs, fret_preds, tuning_pitches=None, onset_threshold=ONSET_THRESHOLD)

            ref_intervals, ref_pitches = to_mireval(gt)
            if not pred_notes:
                p, r, f1 = 0.0, 0.0, 0.0
            else:
                est_intervals, est_pitches = to_mireval(pred_notes)
                p, r, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
                    ref_intervals, ref_pitches, est_intervals, est_pitches,
                    onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None
                )
            f1s.append(f1); ps.append(p); rs.append(r)
            print(f"  [{i+1}/{len(pairs)}] {name} F1={f1:.4f}")

        if f1s:
            print(f"\n  >>> {split_name.upper()}: F1={np.mean(f1s):.4f}  P={np.mean(ps):.4f}  R={np.mean(rs):.4f}  (N={len(f1s)})")

    print("\nDone.")

if __name__ == "__main__":
    main()
