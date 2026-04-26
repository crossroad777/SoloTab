"""
detailed_benchmark.py — Pure MoE FT 全曲詳細ベンチマーク
========================================================
最適パラメータ固定で、ジャンル別・スタイル別・プレイヤー別の詳細分析を行う。
"""
import os, sys, glob, json, time
import numpy as np
from scipy import stats

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

# 最適パラメータ（全360曲グリッドサーチ結果）
VOTE_THRESHOLD = 5
ONSET_THRESHOLD = 0.8
VOTE_PROB_THRESHOLD = 0.5

MODELS = [
    "finetuned_martin_finger_guitarset_ft", "finetuned_taylor_finger_guitarset_ft",
    "finetuned_luthier_finger_guitarset_ft", "finetuned_martin_pick_guitarset_ft",
    "finetuned_taylor_pick_guitarset_ft", "finetuned_luthier_pick_guitarset_ft",
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
        model_path = os.path.join(mt_python_dir, "_processed_guitarset_data", "training_output", model_dir, "best_model.pth")
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
    return notes


def parse_track_name(name):
    """Parse GuitarSet filename into components.
    Example: '00_BN1-129-Eb_comp' -> player=00, genre=BN, style=comp
    """
    parts = name.split("_")
    player = parts[0]
    genre_part = parts[1].split("-")[0]  # e.g., 'BN1' -> 'BN'
    genre = ''.join([c for c in genre_part if c.isalpha()])
    style = parts[-1]  # 'comp' or 'solo'
    return player, genre, style


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Parameters: vote={VOTE_THRESHOLD}, onset={ONSET_THRESHOLD}, vprob={VOTE_PROB_THRESHOLD}")

    jams_files = sorted(glob.glob(os.path.join(ANNOTATIONS_DIR, "*.jams")))
    pairs = []
    for jams_path in jams_files:
        base = os.path.basename(jams_path).replace(".jams", "")
        wav_path = os.path.join(AUDIO_DIR, f"{base}_mic.wav")
        if os.path.exists(wav_path):
            pairs.append((jams_path, wav_path, base))

    print(f"Total tracks: {len(pairs)}")

    # Per-track results
    results = []
    for i, (jams_path, wav_path, name) in enumerate(pairs):
        print(f"[{i+1}/{len(pairs)}] {name}", end=" ", flush=True)
        gt_notes = load_jams_notes(jams_path)
        if not gt_notes:
            print("-> No GT, skip")
            continue

        all_onset_probs, all_fret_preds = run_inference(wav_path, device)
        pred_notes = decode(all_onset_probs, all_fret_preds)
        
        ref_intervals, ref_pitches = to_mireval(gt_notes)
        if not pred_notes:
            p, r, f1 = 0.0, 0.0, 0.0
        else:
            est_intervals, est_pitches = to_mireval(pred_notes)
            p, r, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
                ref_intervals, ref_pitches, est_intervals, est_pitches,
                onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None
            )

        player, genre, style = parse_track_name(name)
        results.append({
            "name": name, "player": player, "genre": genre, "style": style,
            "f1": f1, "precision": p, "recall": r,
            "gt_notes": len(gt_notes), "pred_notes": len(pred_notes),
        })
        print(f"F1={f1:.4f} P={p:.4f} R={r:.4f} (GT={len(gt_notes)}, Pred={len(pred_notes)})")

    # ===== Aggregate Results =====
    print(f"\n{'='*80}")
    print(f" DETAILED BENCHMARK RESULTS (360 tracks)")
    print(f" Parameters: vote={VOTE_THRESHOLD}, onset={ONSET_THRESHOLD}, vprob={VOTE_PROB_THRESHOLD}")
    print(f"{'='*80}")

    all_f1 = [r["f1"] for r in results]
    all_p = [r["precision"] for r in results]
    all_r = [r["recall"] for r in results]
    print(f"\n OVERALL: F1={np.mean(all_f1):.4f}  P={np.mean(all_p):.4f}  R={np.mean(all_r):.4f}  (N={len(results)})")

    # Genre breakdown
    genre_map = {"BN": "Bossa Nova", "Funk": "Funk", "Jazz": "Jazz", "Rock": "Rock", "SS": "Singer-Songwriter"}
    print(f"\n{'─'*60}")
    print(f" BY GENRE")
    print(f"{'─'*60}")
    print(f" {'Genre':<22} {'Tracks':>6} {'F1':>8} {'P':>8} {'R':>8}")
    print(f" {'─'*54}")
    for g_key in ["BN", "Funk", "Jazz", "Rock", "SS"]:
        g_results = [r for r in results if r["genre"] == g_key]
        if not g_results:
            continue
        gf1 = np.mean([r["f1"] for r in g_results])
        gp = np.mean([r["precision"] for r in g_results])
        gr = np.mean([r["recall"] for r in g_results])
        print(f" {genre_map.get(g_key, g_key):<22} {len(g_results):>6} {gf1:>8.4f} {gp:>8.4f} {gr:>8.4f}")

    # Style breakdown
    print(f"\n{'─'*60}")
    print(f" BY STYLE")
    print(f"{'─'*60}")
    print(f" {'Style':<22} {'Tracks':>6} {'F1':>8} {'P':>8} {'R':>8}")
    print(f" {'─'*54}")
    for style_key in ["comp", "solo"]:
        s_results = [r for r in results if r["style"] == style_key]
        if not s_results:
            continue
        sf1 = np.mean([r["f1"] for r in s_results])
        sp = np.mean([r["precision"] for r in s_results])
        sr = np.mean([r["recall"] for r in s_results])
        label = "Comping (伴奏)" if style_key == "comp" else "Solo (独奏)"
        print(f" {label:<22} {len(s_results):>6} {sf1:>8.4f} {sp:>8.4f} {sr:>8.4f}")

    # Genre × Style
    print(f"\n{'─'*60}")
    print(f" BY GENRE × STYLE")
    print(f"{'─'*60}")
    print(f" {'Genre × Style':<30} {'N':>4} {'F1':>8} {'P':>8} {'R':>8}")
    print(f" {'─'*54}")
    for g_key in ["BN", "Funk", "Jazz", "Rock", "SS"]:
        for style_key in ["comp", "solo"]:
            gs_results = [r for r in results if r["genre"] == g_key and r["style"] == style_key]
            if not gs_results:
                continue
            gsf1 = np.mean([r["f1"] for r in gs_results])
            gsp = np.mean([r["precision"] for r in gs_results])
            gsr = np.mean([r["recall"] for r in gs_results])
            label = f"{genre_map.get(g_key, g_key)} ({style_key})"
            print(f" {label:<30} {len(gs_results):>4} {gsf1:>8.4f} {gsp:>8.4f} {gsr:>8.4f}")

    # Player breakdown
    print(f"\n{'─'*60}")
    print(f" BY PLAYER")
    print(f"{'─'*60}")
    print(f" {'Player':<22} {'Tracks':>6} {'F1':>8} {'P':>8} {'R':>8}")
    print(f" {'─'*54}")
    players = sorted(set(r["player"] for r in results))
    for p_key in players:
        p_results = [r for r in results if r["player"] == p_key]
        pf1 = np.mean([r["f1"] for r in p_results])
        pp = np.mean([r["precision"] for r in p_results])
        pr_val = np.mean([r["recall"] for r in p_results])
        print(f" Player {p_key:<16} {len(p_results):>6} {pf1:>8.4f} {pp:>8.4f} {pr_val:>8.4f}")

    # Worst 10 tracks
    print(f"\n{'─'*60}")
    print(f" WORST 10 TRACKS")
    print(f"{'─'*60}")
    sorted_results = sorted(results, key=lambda x: x["f1"])
    for i, r in enumerate(sorted_results[:10]):
        print(f" {i+1:>2}. {r['name']:<35} F1={r['f1']:.4f} P={r['precision']:.4f} R={r['recall']:.4f} (GT={r['gt_notes']}, Pred={r['pred_notes']})")

    # Best 10 tracks
    print(f"\n{'─'*60}")
    print(f" BEST 10 TRACKS")
    print(f"{'─'*60}")
    for i, r in enumerate(sorted_results[-10:]):
        print(f" {i+1:>2}. {r['name']:<35} F1={r['f1']:.4f} P={r['precision']:.4f} R={r['recall']:.4f}")

    # Save JSON
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detailed_benchmark_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "parameters": {"vote_threshold": VOTE_THRESHOLD, "onset_threshold": ONSET_THRESHOLD, "vote_prob_threshold": VOTE_PROB_THRESHOLD},
            "overall": {"f1": float(np.mean(all_f1)), "precision": float(np.mean(all_p)), "recall": float(np.mean(all_r)), "n_tracks": len(results)},
            "per_track": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
