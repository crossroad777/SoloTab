"""
duration_benchmark.py — ノート持続時間(duration)の精度評価
============================================================
GuitarSetのGround Truthと比較して、
予測ノートのduration精度を測定する。

評価指標:
- Duration MAE (秒): 平均絶対誤差
- Duration Ratio: 予測/正解の比率分布 (1.0が完璧)
- Rhythm Score: onsetがマッチしたノートのduration一致率
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os, sys, glob, json
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

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
PROCESSED_DIR = os.path.join(mt_python_dir, "_processed_guitarset_data")
TRAINING_OUTPUT_DIR = os.path.join(PROCESSED_DIR, "training_output")

VOTE_THRESHOLD = 5
ONSET_THRESHOLD = 0.8
VOTE_PROB_THRESHOLD = 0.5
ONSET_TOLERANCE = 0.05  # 50ms onset matching window

MODELS = [
    "finetuned_martin_finger_guitarset_ft", "finetuned_taylor_finger_guitarset_ft",
    "finetuned_luthier_finger_guitarset_ft", "finetuned_martin_pick_guitarset_ft",
    "finetuned_taylor_pick_guitarset_ft", "finetuned_luthier_pick_guitarset_ft",
]


def load_split_ids(split_name):
    path = os.path.join(PROCESSED_DIR, f"{split_name}_ids.txt")
    if not os.path.exists(path):
        return set()
    with open(path, "r") as f:
        return set(line.strip() for line in f if line.strip())


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
                notes.append({"start": start, "end": start + dur, "duration": dur, "pitch": pitch})
    return notes


def run_inference(wav_path, device):
    y, sr = librosa.load(wav_path, sr=config.SAMPLE_RATE, mono=True)
    cqt_spec = librosa.cqt(y=y, sr=sr, hop_length=config.HOP_LENGTH,
                           fmin=config.FMIN_CQT, n_bins=config.N_BINS_CQT,
                           bins_per_octave=config.BINS_PER_OCTAVE_CQT)
    log_cqt = librosa.amplitude_to_db(np.abs(cqt_spec), ref=np.max)
    features = torch.tensor(log_cqt, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    all_onset_probs = []
    all_fret_preds = []
    for model_dir in MODELS:
        model_path = os.path.join(TRAINING_OUTPUT_DIR, model_dir, "best_model.pth")
        if not os.path.exists(model_path):
            continue
        model = architecture.GuitarTabCRNN(
            num_frames_rnn_input_dim=1280, rnn_type="GRU",
            rnn_hidden_size=768, rnn_layers=2, rnn_dropout=0.3, rnn_bidirectional=True
        )
        sd = torch.load(model_path, map_location=device, weights_only=False)
        if list(sd.keys())[0].startswith("module."):
            sd = {k[7:]: v for k, v in sd.items()}
        model.load_state_dict(sd)
        model.to(device).eval()
        with torch.no_grad():
            onset_logits, fret_logits = model(features)
            onset_probs = torch.sigmoid(onset_logits[0]).cpu().numpy()
            fret_probs = torch.softmax(fret_logits[0], dim=-1).cpu().numpy()
        all_onset_probs.append(onset_probs)
        all_fret_preds.append(np.argmax(fret_probs, axis=-1))
        del model, sd
        torch.cuda.empty_cache()
    return np.array(all_onset_probs), np.array(all_fret_preds)


def decode(all_onset_probs, all_fret_preds):
    binary_votes = all_onset_probs > VOTE_PROB_THRESHOLD
    vote_counts = np.sum(binary_votes, axis=0)
    consensus_onset_probs = np.max(all_onset_probs, axis=0)
    consensus_onset_probs[vote_counts < VOTE_THRESHOLD] = 0.0
    consensus_frets, _ = stats.mode(all_fret_preds, axis=0, keepdims=False)
    notes = _frames_to_notes(consensus_onset_probs, consensus_frets, tuning_pitches=None, onset_threshold=ONSET_THRESHOLD)
    for n in notes:
        n["start"] = float(n["start"])
        n["end"] = float(n["end"])
        n["duration"] = n["end"] - n["start"]
    return notes


def match_notes(gt_notes, pred_notes, onset_tol=ONSET_TOLERANCE, pitch_tol=50.0):
    """Onset+pitchでマッチしたペアを返す"""
    matched = []
    used_pred = set()
    
    for gt in gt_notes:
        gt_hz = 440.0 * (2.0 ** ((gt["pitch"] - 69.0) / 12.0))
        best_pred = None
        best_onset_diff = onset_tol
        
        for j, pred in enumerate(pred_notes):
            if j in used_pred:
                continue
            pred_hz = 440.0 * (2.0 ** ((pred["pitch"] - 69.0) / 12.0))
            
            onset_diff = abs(gt["start"] - pred["start"])
            pitch_ratio = max(gt_hz, pred_hz) / max(min(gt_hz, pred_hz), 1e-8)
            cent_diff = 1200 * np.log2(pitch_ratio) if pitch_ratio > 0 else 9999
            
            if onset_diff <= onset_tol and cent_diff <= pitch_tol and onset_diff < best_onset_diff:
                best_pred = j
                best_onset_diff = onset_diff
        
        if best_pred is not None:
            used_pred.add(best_pred)
            matched.append((gt, pred_notes[best_pred]))
    
    return matched


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ids = load_split_ids("test")
    
    print(f"Device: {device}")
    print(f"Test tracks: {len(test_ids)}")
    print(f"Onset tolerance: {ONSET_TOLERANCE*1000:.0f}ms")
    print()
    
    jams_files = sorted(glob.glob(os.path.join(ANNOTATIONS_DIR, "*.jams")))
    pairs = []
    for jams_path in jams_files:
        base = os.path.basename(jams_path).replace(".jams", "")
        if base in test_ids:
            wav_path = os.path.join(AUDIO_DIR, f"{base}_mic.wav")
            if os.path.exists(wav_path):
                pairs.append((jams_path, wav_path, base))
    
    print(f"Found {len(pairs)} test tracks")
    print()
    
    all_dur_ratios = []
    all_dur_errors = []
    all_gt_durs = []
    all_pred_durs = []
    track_results = []
    
    for i, (jams_path, wav_path, name) in enumerate(pairs):
        print(f"[{i+1}/{len(pairs)}] {name}", end=" ", flush=True)
        
        gt_notes = load_jams_notes(jams_path)
        if not gt_notes:
            print("-> No GT")
            continue
        
        all_onset_probs, all_fret_preds = run_inference(wav_path, device)
        pred_notes = decode(all_onset_probs, all_fret_preds)
        
        matched = match_notes(gt_notes, pred_notes)
        
        if not matched:
            print(f"-> No matches")
            continue
        
        dur_errors = []
        dur_ratios = []
        for gt, pred in matched:
            gt_dur = gt["duration"]
            pred_dur = pred["duration"]
            
            error = abs(pred_dur - gt_dur)
            ratio = pred_dur / gt_dur if gt_dur > 0.01 else 1.0
            
            dur_errors.append(error)
            dur_ratios.append(ratio)
            all_gt_durs.append(gt_dur)
            all_pred_durs.append(pred_dur)
        
        all_dur_errors.extend(dur_errors)
        all_dur_ratios.extend(dur_ratios)
        
        mae = np.mean(dur_errors)
        median_ratio = np.median(dur_ratios)
        within_20pct = sum(1 for r in dur_ratios if 0.8 <= r <= 1.2) / len(dur_ratios)
        within_50pct = sum(1 for r in dur_ratios if 0.5 <= r <= 1.5) / len(dur_ratios)
        
        track_results.append({
            "name": name, "matched": len(matched), "gt_total": len(gt_notes),
            "mae": mae, "median_ratio": median_ratio,
            "within_20pct": within_20pct, "within_50pct": within_50pct,
        })
        
        print(f"matched={len(matched)}/{len(gt_notes)} MAE={mae:.3f}s ratio={median_ratio:.2f} ±20%={within_20pct:.1%} ±50%={within_50pct:.1%}")
    
    # === Summary ===
    print(f"\n{'='*70}")
    print(f" DURATION ACCURACY BENCHMARK (Test {len(track_results)} tracks)")
    print(f"{'='*70}")
    
    total_matched = sum(r["matched"] for r in track_results)
    print(f"\n Total matched notes: {total_matched}")
    print(f" Duration MAE: {np.mean(all_dur_errors):.4f}s ({np.mean(all_dur_errors)*1000:.1f}ms)")
    print(f" Duration Median: GT={np.median(all_gt_durs):.3f}s  Pred={np.median(all_pred_durs):.3f}s")
    print(f" Duration Mean:   GT={np.mean(all_gt_durs):.3f}s  Pred={np.mean(all_pred_durs):.3f}s")
    print(f" Median Ratio (pred/gt): {np.median(all_dur_ratios):.3f}")
    print(f" Mean Ratio (pred/gt):   {np.mean(all_dur_ratios):.3f}")
    
    within_10 = sum(1 for r in all_dur_ratios if 0.9 <= r <= 1.1) / len(all_dur_ratios)
    within_20 = sum(1 for r in all_dur_ratios if 0.8 <= r <= 1.2) / len(all_dur_ratios)
    within_50 = sum(1 for r in all_dur_ratios if 0.5 <= r <= 1.5) / len(all_dur_ratios)
    
    print(f"\n Duration Accuracy (ratio within tolerance):")
    print(f"   ±10%: {within_10:.1%}")
    print(f"   ±20%: {within_20:.1%}")
    print(f"   ±50%: {within_50:.1%}")
    
    # Ratio distribution
    print(f"\n Duration Ratio Distribution:")
    buckets = [(0, 0.25, "<<短い (0-25%)"), (0.25, 0.5, "<短い (25-50%)"),
               (0.5, 0.8, "やや短い (50-80%)"), (0.8, 1.2, "★正確 (80-120%)"),
               (1.2, 1.5, "やや長い (120-150%)"), (1.5, 2.0, ">長い (150-200%)"),
               (2.0, 100, ">>長い (200%+)")]
    for lo, hi, label in buckets:
        count = sum(1 for r in all_dur_ratios if lo <= r < hi)
        pct = count / len(all_dur_ratios)
        bar = "█" * int(pct * 40)
        print(f"   {label:25s} {count:5d} ({pct:5.1%}) {bar}")
    
    # Worst tracks
    print(f"\n Worst 5 tracks (by MAE):")
    sorted_tracks = sorted(track_results, key=lambda x: x["mae"], reverse=True)
    for r in sorted_tracks[:5]:
        print(f"   {r['name']:35s} MAE={r['mae']:.3f}s ratio={r['median_ratio']:.2f} ±20%={r['within_20pct']:.1%}")


if __name__ == "__main__":
    main()
