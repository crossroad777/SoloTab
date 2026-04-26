"""
benchmark_pure_moe.py — Pure MoE (6-model ensemble) F1 Benchmark on GuitarSet
===============================================================================
GuitarSet全曲でPure MoEのF1スコアを計測する。
評価基準: mir_eval transcription (Onset 50ms, Pitch 50 cents, offset無視)
"""

import os
import sys
import glob
import json
import time
import numpy as np

# パス設定
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
backend_dir = os.path.dirname(os.path.abspath(__file__))
backend_parent = os.path.dirname(backend_dir)
sys.path.insert(0, backend_parent)

try:
    import mir_eval
except ImportError:
    print("mir_eval is required. Install: pip install mir_eval")
    sys.exit(1)

from pure_moe_transcriber import transcribe_pure_moe

GUITARSET_DIR = r"D:\Music\Datasets\GuitarSet"
ANNOTATIONS_DIR = os.path.join(GUITARSET_DIR, "annotation")
AUDIO_DIR = os.path.join(GUITARSET_DIR, "audio_mono-mic")


def load_jams_notes(jams_path):
    """JAMSファイルからGTノートを抽出（全弦統合）"""
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
    """mir_eval用のnumpy配列に変換"""
    if not notes:
        return np.empty((0, 2)), np.empty(0)
    intervals = np.array([[n['start'], n['end']] for n in notes], dtype=float)
    pitches_hz = np.array([440.0 * (2.0 ** ((n['pitch'] - 69.0) / 12.0)) for n in notes], dtype=float)
    return intervals, pitches_hz


def evaluate_song(jams_path, wav_path, vote_threshold=3, onset_threshold=0.5):
    """1曲を評価してP/R/F1を返す"""
    gt_notes = load_jams_notes(jams_path)
    if not gt_notes:
        return None

    ref_intervals, ref_pitches = to_mireval(gt_notes)

    # Pure MoE推論
    pred_notes = transcribe_pure_moe(wav_path, vote_threshold=vote_threshold, onset_threshold=onset_threshold)
    if not pred_notes:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "gt_count": len(gt_notes), "pred_count": 0}

    est_intervals, est_pitches = to_mireval(pred_notes)

    # mir_eval評価 (Onset 50ms, Pitch 50 cents, offset無視)
    p, r, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches, est_intervals, est_pitches,
        onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None
    )
    return {"precision": p, "recall": r, "f1": f1, "gt_count": len(gt_notes), "pred_count": len(pred_notes)}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pure MoE GuitarSet Benchmark")
    parser.add_argument("--max-songs", type=int, default=0, help="Max songs to evaluate (0=all)")
    parser.add_argument("--vote", type=int, default=3, help="Vote threshold (default: 3/6)")
    parser.add_argument("--thresh", type=float, default=0.5, help="Onset threshold (default: 0.5)")
    parser.add_argument("--style", choices=["all", "comp", "solo"], default="all", help="Filter by style")
    args = parser.parse_args()

    # JAMS/WAVペアを収集
    jams_files = sorted(glob.glob(os.path.join(ANNOTATIONS_DIR, "*.jams")))
    if not jams_files:
        print(f"[Error] No JAMS files in {ANNOTATIONS_DIR}")
        sys.exit(1)

    # スタイルフィルタ
    if args.style != "all":
        jams_files = [j for j in jams_files if f"_{args.style}.jams" in j]

    pairs = []
    for jams_path in jams_files:
        base = os.path.basename(jams_path).replace(".jams", "")
        wav_path = os.path.join(AUDIO_DIR, f"{base}_mic.wav")
        if os.path.exists(wav_path):
            pairs.append((jams_path, wav_path, base))

    if args.max_songs > 0:
        pairs = pairs[:args.max_songs]

    print(f"=" * 60)
    print(f" Pure MoE GuitarSet Benchmark")
    print(f" Songs: {len(pairs)}, Vote: {args.vote}/6, Onset threshold: {args.thresh}")
    print(f"=" * 60)

    results = []
    t_total = time.time()

    for i, (jams_path, wav_path, name) in enumerate(pairs):
        print(f"\n[{i+1}/{len(pairs)}] {name}")
        t0 = time.time()
        try:
            r = evaluate_song(jams_path, wav_path, vote_threshold=args.vote, onset_threshold=args.thresh)
            if r is None:
                print(f"  -> Skipped (no GT notes)")
                continue
            elapsed = time.time() - t0
            r["name"] = name
            r["time"] = elapsed
            results.append(r)
            print(f"  P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f} "
                  f"(GT={r['gt_count']}, Pred={r['pred_count']}, {elapsed:.1f}s)")
        except Exception as e:
            print(f"  -> Error: {e}")
            import traceback
            traceback.print_exc()

    # 集計
    if not results:
        print("\nNo results to summarize.")
        return

    precisions = [r["precision"] for r in results]
    recalls = [r["recall"] for r in results]
    f1s = [r["f1"] for r in results]
    total_time = time.time() - t_total

    print(f"\n{'=' * 60}")
    print(f" RESULTS: Pure MoE ({len(results)} songs)")
    print(f"{'=' * 60}")
    print(f" Mean Precision : {np.mean(precisions):.4f} (±{np.std(precisions):.4f})")
    print(f" Mean Recall    : {np.mean(recalls):.4f} (±{np.std(recalls):.4f})")
    print(f" Mean F1        : {np.mean(f1s):.4f} (±{np.std(f1s):.4f})")
    print(f" Median F1      : {np.median(f1s):.4f}")
    print(f" Min F1         : {np.min(f1s):.4f} ({results[np.argmin(f1s)]['name']})")
    print(f" Max F1         : {np.max(f1s):.4f} ({results[np.argmax(f1s)]['name']})")
    print(f" Total time     : {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"{'=' * 60}")

    # 結果をJSON保存
    out_path = os.path.join(backend_dir, "pure_moe_benchmark_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {"vote_threshold": args.vote, "onset_threshold": args.thresh, "style": args.style},
            "summary": {
                "mean_precision": float(np.mean(precisions)),
                "mean_recall": float(np.mean(recalls)),
                "mean_f1": float(np.mean(f1s)),
                "median_f1": float(np.median(f1s)),
                "std_f1": float(np.std(f1s)),
                "num_songs": len(results),
            },
            "per_song": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
