"""
grid_search_3ds.py — 3DSモデル用 onset/vote threshold グリッドサーチ
推論は1回だけ行い、閾値の組み合わせをキャッシュ上で高速評価する。
"""
import os, sys, glob, json, time
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "music-transcription", "python"))

import torch, librosa, mir_eval
import config
from model import architecture
from guitar_transcriber import _frames_to_notes

GUITARSET_DIR = r"D:\Music\Datasets\GuitarSet"
ANNOTATIONS_DIR = os.path.join(GUITARSET_DIR, "annotation")
AUDIO_DIR = os.path.join(GUITARSET_DIR, "audio_mono-mic")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "music-transcription", "python", "_processed_guitarset_data", "training_output")

# 3DSモデル (7ドメイン)
MODELS_3DS = [
    "finetuned_martin_finger_multitask_3ds",
    "finetuned_taylor_finger_multitask_3ds",
    "finetuned_luthier_finger_multitask_3ds",
    "finetuned_martin_pick_multitask_3ds",
    "finetuned_taylor_pick_multitask_3ds",
    "finetuned_luthier_pick_multitask_3ds",
    "finetuned_gibson_thumb_multitask_3ds",
]


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


def run_inference_cached(wav_path):
    """7モデル推論を1回実行し、生の確率マップを返す"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y, sr = librosa.load(wav_path, sr=config.SAMPLE_RATE, mono=True)
    cqt_spec = librosa.cqt(
        y=y, sr=sr, hop_length=config.HOP_LENGTH,
        fmin=config.FMIN_CQT, n_bins=config.N_BINS_CQT, bins_per_octave=config.BINS_PER_OCTAVE_CQT
    )
    log_cqt = librosa.amplitude_to_db(np.abs(cqt_spec), ref=np.max)
    features = torch.tensor(log_cqt, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    all_onset_probs = []
    all_fret_preds = []

    for model_dir in MODELS_3DS:
        model_path = os.path.join(OUTPUT_DIR, model_dir, "best_model.pth")
        if not os.path.exists(model_path):
            print(f"  WARN: {model_dir} not found")
            continue
        model = architecture.GuitarTabCRNN(
            num_frames_rnn_input_dim=1280, rnn_type="GRU",
            rnn_hidden_size=768, rnn_layers=2, rnn_dropout=0.3, rnn_bidirectional=True
        )
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(device).eval()
        with torch.no_grad():
            onset_logits, fret_logits = model(features)
            onset_probs = torch.sigmoid(onset_logits[0]).cpu().numpy()
            fret_probs = torch.softmax(fret_logits[0], dim=-1).cpu().numpy()
        all_onset_probs.append(onset_probs)
        all_fret_preds.append(np.argmax(fret_probs, axis=-1))
        del model, state_dict
        torch.cuda.empty_cache()

    return np.array(all_onset_probs), np.array(all_fret_preds)


def decode_with_params(all_onset_probs, all_fret_preds, vote_threshold, onset_threshold, vote_prob_threshold=0.5):
    n_models = all_onset_probs.shape[0]
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-songs", type=int, default=10)
    args = parser.parse_args()

    # パラメータグリッド (7モデルなので vote は 3-7)
    vote_thresholds = [3, 4, 5, 6, 7]
    onset_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    vote_prob_thresholds = [0.4, 0.5]

    # Test分割の曲を使用（均等サンプリング）
    jams_files = sorted(glob.glob(os.path.join(ANNOTATIONS_DIR, "*.jams")))
    pairs = []
    for jams_path in jams_files:
        base = os.path.basename(jams_path).replace(".jams", "")
        wav_path = os.path.join(AUDIO_DIR, f"{base}_mic.wav")
        if os.path.exists(wav_path):
            pairs.append((jams_path, wav_path, base))

    if args.max_songs > 0:
        step = max(1, len(pairs) // args.max_songs)
        pairs = pairs[::step][:args.max_songs]

    total_combos = len(vote_thresholds) * len(onset_thresholds) * len(vote_prob_thresholds)
    print(f"Songs: {len(pairs)}, Combos: {total_combos}")

    # 推論キャッシュ
    song_data = []
    for i, (jams_path, wav_path, name) in enumerate(pairs):
        print(f"[{i+1}/{len(pairs)}] {name}")
        gt_notes = load_jams_notes(jams_path)
        if not gt_notes:
            continue
        ref_intervals, ref_pitches = to_mireval(gt_notes)
        all_onset_probs, all_fret_preds = run_inference_cached(wav_path)
        song_data.append({
            "name": name, "ref_intervals": ref_intervals, "ref_pitches": ref_pitches,
            "all_onset_probs": all_onset_probs, "all_fret_preds": all_fret_preds,
        })

    # グリッドサーチ
    best_f1 = 0
    best_params = {}
    all_results = []

    for vt in vote_thresholds:
        for ot in onset_thresholds:
            for vpt in vote_prob_thresholds:
                total_tp, total_fp, total_fn = 0, 0, 0
                for sd in song_data:
                    pred_notes = decode_with_params(sd["all_onset_probs"], sd["all_fret_preds"], vt, ot, vpt)
                    if not pred_notes:
                        total_fn += len(sd["ref_intervals"])
                        continue
                    est_intervals, est_pitches = to_mireval(pred_notes)
                    p, r, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
                        sd["ref_intervals"], sd["ref_pitches"],
                        est_intervals, est_pitches,
                        onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None
                    )
                    tp = int(round(r * len(sd["ref_intervals"])))
                    fp = len(pred_notes) - int(round(p * len(pred_notes)))
                    fn = len(sd["ref_intervals"]) - tp
                    total_tp += tp
                    total_fp += max(0, fp)
                    total_fn += max(0, fn)

                macro_p = total_tp / max(total_tp + total_fp, 1)
                macro_r = total_tp / max(total_tp + total_fn, 1)
                macro_f1 = 2 * macro_p * macro_r / max(macro_p + macro_r, 1e-8)
                result = {"vote": vt, "onset": ot, "vprob": vpt, "F1": round(macro_f1, 4), "P": round(macro_p, 4), "R": round(macro_r, 4)}
                all_results.append(result)
                marker = " <<<< BEST" if macro_f1 > best_f1 else ""
                if macro_f1 > best_f1:
                    best_f1 = macro_f1
                    best_params = result.copy()
                print(f"  v={vt} o={ot:.1f} vp={vpt:.1f} -> P={macro_p:.4f} R={macro_r:.4f} F1={macro_f1:.4f}{marker}")

    print(f"\n{'='*50}")
    print(f"BEST: vote={best_params['vote']} onset={best_params['onset']} vprob={best_params['vprob']}")
    print(f"F1={best_params['F1']:.4f} P={best_params['P']:.4f} R={best_params['R']:.4f}")
    print(f"{'='*50}")

    all_results.sort(key=lambda x: x["F1"], reverse=True)
    print("\nTop 10:")
    for i, r in enumerate(all_results[:10]):
        print(f"  {i+1}. v={r['vote']} o={r['onset']:.1f} vp={r['vprob']:.1f} -> F1={r['F1']:.4f} P={r['P']:.4f} R={r['R']:.4f}")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grid_search_3ds_results.json")
    with open(out_path, "w") as f:
        json.dump({"best": best_params, "all": all_results[:20]}, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
