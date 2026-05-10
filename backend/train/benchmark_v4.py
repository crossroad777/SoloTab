"""
benchmark_v4.py — 高精度ベンチマーク（ソロ/コンプ分離 + エラー分析）
====================================================================
GuitarSet solo のみでベンチマーク → Viterbiは単音向けなので精度向上が見込まれる。
エラーパターンを分析して次の改善点を特定する。

Usage:
    python backend/train/benchmark_v4.py
"""
import json, glob, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from collections import Counter, defaultdict
import string_assigner

GUITARSET_DIR = r"D:\Music\datasets\GuitarSet\annotation"
from solotab_utils import STANDARD_TUNING
DS_TO_STRING = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}


def load_phrases(filter_type="solo", max_files=None):
    """GuitarSetからフレーズをロード。filter_type: 'solo', 'comp', 'all'"""
    jams_files = sorted(glob.glob(os.path.join(GUITARSET_DIR, "*.jams")))
    if filter_type == "solo":
        jams_files = [f for f in jams_files if "_solo" in os.path.basename(f)]
    elif filter_type == "comp":
        jams_files = [f for f in jams_files if "_comp" in os.path.basename(f)]
    if max_files:
        jams_files = jams_files[:max_files]

    phrases = []
    for jf in jams_files:
        with open(jf, "r") as f:
            data = json.load(f)
        file_notes = []
        for ann in data.get("annotations", []):
            if ann.get("namespace") != "note_midi":
                continue
            ds = ann.get("annotation_metadata", {}).get("data_source", "")
            try:
                ds_idx = int(ds)
            except (ValueError, TypeError):
                continue
            if ds_idx not in DS_TO_STRING:
                continue
            string_idmt = DS_TO_STRING[ds_idx]
            string_std = 7 - string_idmt  # 標準形式に変換
            open_pitch = STANDARD_TUNING[ds_idx]
            for obs in ann.get("data", []):
                pitch = round(obs.get("value", 0))
                if pitch <= 0:
                    continue
                fret = pitch - open_pitch
                if 0 <= fret <= 22:
                    file_notes.append({
                        "pitch": pitch,
                        "string": string_std,  # 標準形式で保存
                        "fret": fret,
                        "start": obs.get("time", 0),
                        "duration": obs.get("duration", 0),
                    })
        if file_notes:
            file_notes.sort(key=lambda n: n["start"])
            phrases.append((os.path.basename(jf), file_notes))
    return phrases


def evaluate(phrases, w_human=-15.0, verbose=False):
    """Viterbi DPでの弦割り当て精度を評価"""
    string_assigner.WEIGHTS["w_human_pref_bonus"] = w_human

    total_correct = 0
    total_fret_correct = 0
    total_both_correct = 0
    total_notes = 0
    errors = []
    per_string_correct = Counter()
    per_string_total = Counter()

    for filename, notes in phrases:
        midi_notes = [{"pitch": n["pitch"], "start": n["start"],
                       "duration": n["duration"]} for n in notes]
        try:
            assigned = string_assigner.assign_strings_dp(midi_notes)
        except Exception as e:
            if verbose:
                print(f"  ERROR {filename}: {e}")
            continue

        for a in assigned:
            pred_s = a.get("string", 0)
            pred_f = a.get("fret", -1)
            pitch = a.get("pitch", 0)
            start = a.get("start", 0)

            best_match = None
            best_dist = float("inf")
            for gt in notes:
                if gt["pitch"] == pitch:
                    dist = abs(gt["start"] - start)
                    if dist < best_dist:
                        best_dist = dist
                        best_match = gt

            if best_match and best_dist < 0.1:
                total_notes += 1
                gt_s = best_match["string"]
                gt_f = best_match["fret"]
                per_string_total[gt_s] += 1

                s_ok = (pred_s == gt_s)
                f_ok = (pred_f == gt_f)

                if s_ok:
                    total_correct += 1
                    per_string_correct[gt_s] += 1
                if f_ok:
                    total_fret_correct += 1
                if s_ok and f_ok:
                    total_both_correct += 1
                if not s_ok:
                    errors.append({
                        "pitch": pitch,
                        "gt": f"S{gt_s}F{gt_f}",
                        "pred": f"S{pred_s}F{pred_f}",
                        "file": filename if verbose else "",
                    })

    return {
        "correct": total_correct,
        "fret_correct": total_fret_correct,
        "both_correct": total_both_correct,
        "total": total_notes,
        "errors": errors,
        "per_string_correct": per_string_correct,
        "per_string_total": per_string_total,
    }


def print_results(label, result):
    t = result["total"]
    if t == 0:
        print(f"{label}: No data")
        return

    s_acc = result["correct"] / t * 100
    f_acc = result["fret_correct"] / t * 100
    b_acc = result["both_correct"] / t * 100

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  String accuracy:  {result['correct']:5d}/{t:5d} = {s_acc:.1f}%")
    print(f"  Fret accuracy:    {result['fret_correct']:5d}/{t:5d} = {f_acc:.1f}%")
    print(f"  Both (S+F):       {result['both_correct']:5d}/{t:5d} = {b_acc:.1f}%")

    print(f"\n  Per-string accuracy:")
    for s in range(1, 7):
        c = result["per_string_correct"].get(s, 0)
        tt = result["per_string_total"].get(s, 0)
        a = c / tt * 100 if tt > 0 else 0
        bar = "#" * int(a / 2)
        print(f"    S{s} ({['E4','B3','G3','D3','A2','E2'][s-1]}): {c:4d}/{tt:4d} = {a:5.1f}% {bar}")

    # Top error patterns
    if result["errors"]:
        patterns = Counter(f"{e['gt']}->{e['pred']}" for e in result["errors"])
        print(f"\n  Top 10 error patterns:")
        for pattern, count in patterns.most_common(10):
            pct = count / t * 100
            print(f"    {pattern:20s} x{count:4d} ({pct:.1f}%)")

        # Error by pitch
        pitch_errors = Counter(e["pitch"] for e in result["errors"])
        print(f"\n  Most error-prone pitches:")
        for pitch, count in pitch_errors.most_common(5):
            pct = count / t * 100
            print(f"    MIDI {pitch:3d}: {count:4d} errors ({pct:.1f}%)")


def main():
    print("=== High-Precision Fingering Benchmark v4 ===\n")

    # 1. Solo vs Comp vs All
    for ftype in ["solo", "comp", "all"]:
        phrases = load_phrases(filter_type=ftype, max_files=60)
        total_n = sum(len(n) for _, n in phrases)
        print(f"\n--- {ftype.upper()} ---")
        print(f"  Files: {len(phrases)}, Notes: {total_n}")

        result = evaluate(phrases, w_human=-15.0)
        print_results(f"{ftype.upper()} (human pref -15)", result)

    # 2. Solo のみで重みスイープ
    solo_phrases = load_phrases(filter_type="solo", max_files=60)
    print(f"\n\n{'='*60}")
    print(f"  SOLO WEIGHT SWEEP")
    print(f"{'='*60}")
    for w in [0, -5, -10, -15, -20, -30, -50]:
        r = evaluate(solo_phrases, w_human=w)
        acc = r["correct"] / r["total"] * 100 if r["total"] > 0 else 0
        print(f"  w_human={w:4d}: {r['correct']}/{r['total']} = {acc:.1f}%")


if __name__ == "__main__":
    main()
