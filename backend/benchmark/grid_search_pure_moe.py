"""
grid_search_pure_moe.py — Pure MoE ハイパーパラメータグリッドサーチ
==================================================================
推論は1回だけ行い、vote_threshold × onset_threshold の組み合わせを
キャッシュされた確率マップ上で高速に評価する。
"""

import os
import sys
import glob
import json
import time
import numpy as np
from scipy import stats

project_root = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(project_root) if os.path.basename(project_root) == "benchmark" else project_root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

try:
    import mir_eval
except ImportError:
    print("mir_eval required: pip install mir_eval")
    sys.exit(1)

import torch
import librosa

mt_python_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "music-transcription", "python")
sys.path.insert(0, mt_python_dir)
import config
from model import architecture
from guitar_transcriber import _frames_to_notes

GUITARSET_DIR = r"D:\Music\Datasets\GuitarSet"
ANNOTATIONS_DIR = os.path.join(GUITARSET_DIR, "annotation")
AUDIO_DIR = os.path.join(GUITARSET_DIR, "audio_mono-mic")


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
    """6モデル推論を1回実行し、生の確率マップを返す（閾値適用前）"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y, sr = librosa.load(wav_path, sr=config.SAMPLE_RATE, mono=True)
    cqt_spec = librosa.cqt(
        y=y, sr=sr, hop_length=config.HOP_LENGTH,
        fmin=config.FMIN_CQT, n_bins=config.N_BINS_CQT, bins_per_octave=config.BINS_PER_OCTAVE_CQT
    )
    log_cqt = librosa.amplitude_to_db(np.abs(cqt_spec), ref=np.max)
    features = torch.tensor(log_cqt, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    models = [
        "finetuned_martin_finger_guitarset_ft", "finetuned_taylor_finger_guitarset_ft",
        "finetuned_luthier_finger_guitarset_ft", "finetuned_martin_pick_guitarset_ft",
        "finetuned_taylor_pick_guitarset_ft", "finetuned_luthier_pick_guitarset_ft",
    ]

    all_onset_probs = []
    all_fret_preds = []

    for model_dir in models:
        model_path = os.path.join(mt_python_dir, "_processed_guitarset_data", "training_output", model_dir, "best_model.pth")
        if not os.path.exists(model_path):
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


def decode_with_params(all_onset_probs, all_fret_preds, vote_threshold, onset_threshold, vote_prob_threshold=0.4):
    """キャッシュされた確率マップから指定パラメータでノートをデコード"""
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
    parser = argparse.ArgumentParser(description="Pure MoE Hyperparameter Grid Search")
    parser.add_argument("--max-songs", type=int, default=10, help="Number of songs to evaluate")
    parser.add_argument("--style", choices=["all", "comp", "solo"], default="all")
    args = parser.parse_args()

    # パラメータグリッド
    vote_thresholds = [2, 3, 4, 5]
    onset_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    vote_prob_thresholds = [0.3, 0.4, 0.5]

    # 曲ペア収集
    jams_files = sorted(glob.glob(os.path.join(ANNOTATIONS_DIR, "*.jams")))
    if args.style != "all":
        jams_files = [j for j in jams_files if f"_{args.style}.jams" in j]

    pairs = []
    for jams_path in jams_files:
        base = os.path.basename(jams_path).replace(".jams", "")
        wav_path = os.path.join(AUDIO_DIR, f"{base}_mic.wav")
        if os.path.exists(wav_path):
            pairs.append((jams_path, wav_path, base))

    if args.max_songs > 0:
        # 均等にサンプリング
        step = max(1, len(pairs) // args.max_songs)
        pairs = pairs[::step][:args.max_songs]

    print(f"=" * 60)
    print(f" Pure MoE Grid Search")
    print(f" Songs: {len(pairs)}")
    print(f" Vote thresholds: {vote_thresholds}")
    print(f" Onset thresholds: {onset_thresholds}")
    print(f" Vote prob thresholds: {vote_prob_thresholds}")
    print(f" Total combinations: {len(vote_thresholds) * len(onset_thresholds) * len(vote_prob_thresholds)}")
    print(f"=" * 60)

    # 全曲の推論結果をキャッシュ
    song_data = []
    for i, (jams_path, wav_path, name) in enumerate(pairs):
        print(f"\n[{i+1}/{len(pairs)}] Inference: {name}")
        gt_notes = load_jams_notes(jams_path)
        if not gt_notes:
            print("  -> No GT, skipping")
            continue
        ref_intervals, ref_pitches = to_mireval(gt_notes)
        t0 = time.time()
        all_onset_probs, all_fret_preds = run_inference_cached(wav_path)
        print(f"  Inference: {time.time()-t0:.1f}s, GT notes: {len(gt_notes)}")
        song_data.append({
            "name": name,
            "gt_notes": gt_notes,
            "ref_intervals": ref_intervals,
            "ref_pitches": ref_pitches,
            "all_onset_probs": all_onset_probs,
            "all_fret_preds": all_fret_preds,
        })

    # グリッドサーチ
    print(f"\n{'=' * 60}")
    print(f" Running grid search on {len(song_data)} songs...")
    print(f"{'=' * 60}")

    best_f1 = 0
    best_params = {}
    all_results = []

    for vt in vote_thresholds:
        for ot in onset_thresholds:
            for vpt in vote_prob_thresholds:
                f1s = []
                ps = []
                rs = []
                for sd in song_data:
                    pred_notes = decode_with_params(sd["all_onset_probs"], sd["all_fret_preds"], vt, ot, vpt)
                    if not pred_notes:
                        f1s.append(0.0)
                        ps.append(0.0)
                        rs.append(0.0)
                        continue
                    est_intervals, est_pitches = to_mireval(pred_notes)
                    p, r, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
                        sd["ref_intervals"], sd["ref_pitches"],
                        est_intervals, est_pitches,
                        onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None
                    )
                    f1s.append(f1)
                    ps.append(p)
                    rs.append(r)

                mean_f1 = np.mean(f1s)
                mean_p = np.mean(ps)
                mean_r = np.mean(rs)
                result = {
                    "vote_threshold": vt, "onset_threshold": ot, "vote_prob_threshold": vpt,
                    "mean_f1": float(mean_f1), "mean_precision": float(mean_p), "mean_recall": float(mean_r),
                }
                all_results.append(result)

                marker = " <<<< BEST" if mean_f1 > best_f1 else ""
                if mean_f1 > best_f1:
                    best_f1 = mean_f1
                    best_params = result.copy()
                print(f"  vote={vt} onset={ot:.1f} vprob={vpt:.1f} -> P={mean_p:.4f} R={mean_r:.4f} F1={mean_f1:.4f}{marker}")

    # 結果サマリー
    print(f"\n{'=' * 60}")
    print(f" BEST PARAMETERS")
    print(f"{'=' * 60}")
    print(f" Vote threshold      : {best_params['vote_threshold']}")
    print(f" Onset threshold     : {best_params['onset_threshold']}")
    print(f" Vote prob threshold : {best_params['vote_prob_threshold']}")
    print(f" Mean F1             : {best_params['mean_f1']:.4f}")
    print(f" Mean Precision      : {best_params['mean_precision']:.4f}")
    print(f" Mean Recall         : {best_params['mean_recall']:.4f}")
    print(f"{'=' * 60}")

    # Top 5
    all_results.sort(key=lambda x: x["mean_f1"], reverse=True)
    print("\nTop 5 configurations:")
    for i, r in enumerate(all_results[:5]):
        print(f"  {i+1}. vote={r['vote_threshold']} onset={r['onset_threshold']:.1f} vprob={r['vote_prob_threshold']:.1f} "
              f"-> F1={r['mean_f1']:.4f} (P={r['mean_precision']:.4f} R={r['mean_recall']:.4f})")

    # JSON保存
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grid_search_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"best": best_params, "all": all_results}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
