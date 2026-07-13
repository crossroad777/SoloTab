"""
duration_benchmark_quantized.py — 量子化後のduration精度評価
=============================================================
_frames_to_notes → _assign_to_bars (心理音響学+グリッドスナップ) → 
最終duration_divs を正解と比較。

これが実際にGP5に出力されてユーザーが聴く音の間隔。
"""
import os, sys, glob, json, re
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

# tab_renderer from SoloTab backend
from tab_renderer import _assign_to_bars

GUITARSET_DIR = r"D:\Music\Datasets\GuitarSet"
ANNOTATIONS_DIR = os.path.join(GUITARSET_DIR, "annotation")
AUDIO_DIR = os.path.join(GUITARSET_DIR, "audio_mono-mic")
PROCESSED_DIR = os.path.join(mt_python_dir, "_processed_guitarset_data")
TRAINING_OUTPUT_DIR = os.path.join(PROCESSED_DIR, "training_output")

VOTE_THRESHOLD = 5
ONSET_THRESHOLD = 0.8
VOTE_PROB_THRESHOLD = 0.5
ONSET_TOLERANCE = 0.05

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


def extract_bpm(name):
    """GuitarSet filename -> BPM. e.g. '00_BN3-119-G_solo' -> 119"""
    m = re.search(r'-(\d+)-', name)
    return int(m.group(1)) if m else 120


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


def match_notes_with_entries(gt_notes, entries, bpm, onset_tol=ONSET_TOLERANCE, pitch_tol=50.0):
    """GT notes vs quantized entries をマッチ"""
    sec_per_beat = 60.0 / bpm
    divisions = 12
    matched = []
    used = set()
    
    for gt in gt_notes:
        gt_hz = 440.0 * (2.0 ** ((gt["pitch"] - 69.0) / 12.0))
        best_entry = None
        best_diff = onset_tol
        
        for j, e in enumerate(entries):
            if j in used:
                continue
            e_hz = 440.0 * (2.0 ** ((e["pitch"] - 69.0) / 12.0))
            onset_diff = abs(gt["start"] - e["start_time"])
            pitch_ratio = max(gt_hz, e_hz) / max(min(gt_hz, e_hz), 1e-8)
            cent_diff = 1200 * np.log2(pitch_ratio) if pitch_ratio > 0 else 9999
            
            if onset_diff <= onset_tol and cent_diff <= pitch_tol and onset_diff < best_diff:
                best_entry = j
                best_diff = onset_diff
        
        if best_entry is not None:
            used.add(best_entry)
            e = entries[best_entry]
            # duration_divs -> seconds
            quantized_dur_sec = e["duration_divs"] / divisions * sec_per_beat
            matched.append((gt, e, quantized_dur_sec))
    
    return matched


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ids = load_split_ids("test")
    
    print(f"Device: {device}")
    print(f"Test tracks: {len(test_ids)}")
    print()
    
    jams_files = sorted(glob.glob(os.path.join(ANNOTATIONS_DIR, "*.jams")))
    pairs = []
    for jams_path in jams_files:
        base = os.path.basename(jams_path).replace(".jams", "")
        if base in test_ids:
            wav_path = os.path.join(AUDIO_DIR, f"{base}_mic.wav")
            if os.path.exists(wav_path):
                pairs.append((jams_path, wav_path, base))
    
    all_ratios = []
    all_errors = []
    track_results = []
    
    for i, (jams_path, wav_path, name) in enumerate(pairs):
        print(f"[{i+1}/{len(pairs)}] {name}", end=" ", flush=True)
        
        gt_notes = load_jams_notes(jams_path)
        if not gt_notes:
            print("-> No GT"); continue
        
        bpm = extract_bpm(name)
        sec_per_beat = 60.0 / bpm
        
        # Generate beat grid
        audio_dur = max(n["end"] for n in gt_notes) + 2.0
        beats = [i * sec_per_beat for i in range(int(audio_dur / sec_per_beat) + 4)]
        
        # Run MoE inference
        all_onset_probs, all_fret_preds = run_inference(wav_path, device)
        pred_notes = decode(all_onset_probs, all_fret_preds)
        
        if not pred_notes:
            print("-> No predictions"); continue
        
        # Run _assign_to_bars (psychoacoustic model + grid quantization)
        entries = _assign_to_bars(pred_notes, beats, beats_per_bar=4)
        
        if not entries:
            print("-> No entries"); continue
        
        # Match GT vs quantized entries
        matched = match_notes_with_entries(gt_notes, entries, bpm)
        
        if not matched:
            print("-> No matches"); continue
        
        ratios = []
        errors = []
        for gt, entry, q_dur in matched:
            gt_dur = gt["duration"]
            if gt_dur < 0.01: continue
            ratio = q_dur / gt_dur
            error = abs(q_dur - gt_dur)
            ratios.append(ratio)
            errors.append(error)
        
        all_ratios.extend(ratios)
        all_errors.extend(errors)
        
        mae = np.mean(errors)
        med_ratio = np.median(ratios)
        w20 = sum(1 for r in ratios if 0.8 <= r <= 1.2) / max(len(ratios), 1)
        w50 = sum(1 for r in ratios if 0.5 <= r <= 1.5) / max(len(ratios), 1)
        
        track_results.append({"name": name, "bpm": bpm, "matched": len(matched),
                              "mae": mae, "ratio": med_ratio, "w20": w20, "w50": w50})
        
        print(f"BPM={bpm} matched={len(matched)}/{len(gt_notes)} MAE={mae:.3f}s ratio={med_ratio:.2f} w20={w20:.1%} w50={w50:.1%}")
    
    # === Summary ===
    print(f"\n{'='*70}")
    print(f" QUANTIZED DURATION BENCHMARK (Test {len(track_results)} tracks)")
    print(f"{'='*70}")
    
    print(f"\n [RAW (前回)] Duration MAE: 237ms, Median ratio: 1.19, Mean ratio: 2.00")
    print(f" [QUANTIZED]  Duration MAE: {np.mean(all_errors)*1000:.1f}ms, Median ratio: {np.median(all_ratios):.3f}, Mean ratio: {np.mean(all_ratios):.3f}")
    
    w10 = sum(1 for r in all_ratios if 0.9 <= r <= 1.1) / len(all_ratios)
    w20 = sum(1 for r in all_ratios if 0.8 <= r <= 1.2) / len(all_ratios)
    w50 = sum(1 for r in all_ratios if 0.5 <= r <= 1.5) / len(all_ratios)
    print(f"\n Duration Accuracy:")
    print(f"   +/-10%: {w10:.1%}")
    print(f"   +/-20%: {w20:.1%}")
    print(f"   +/-50%: {w50:.1%}")
    
    # Ratio buckets
    print(f"\n Duration Ratio Distribution:")
    buckets = [(0, 0.5, "<<short"), (0.5, 0.8, "<short"), (0.8, 1.2, "* GOOD *"),
               (1.2, 1.5, ">long"), (1.5, 2.0, ">>long"), (2.0, 100, ">>>way too long")]
    for lo, hi, label in buckets:
        count = sum(1 for r in all_ratios if lo <= r < hi)
        pct = count / len(all_ratios)
        bar = "#" * int(pct * 50)
        print(f"   {label:20s} {count:5d} ({pct:5.1%}) {bar}")


if __name__ == "__main__":
    main()
