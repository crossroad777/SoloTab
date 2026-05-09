"""
CNNエラーの人間演奏パターン分析
================================
S2(B3)とS6(E2)のエラーが何を意味するか:
- どのピッチで間違えるか
- 人間はどの弦を選び、CNNはどの弦を選ぶか
- 演奏上の文脈（ポジション、前後の音）
"""
import sys, os, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jams
from collections import Counter, defaultdict
from string_assigner import (
    _load_string_classifier, _predict_string_probs,
    STANDARD_TUNING, MAX_FRET
)

ANNOTATION_DIR = r"D:\Music\Datasets\GuitarSet\annotation"
AUDIO_DIR = r"D:\Music\Datasets\GuitarSet\audio_mono-mic"
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_name(midi):
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def main():
    _load_string_classifier()

    jams_files = sorted([f for f in glob.glob(os.path.join(ANNOTATION_DIR, "*.jams"))
                         if "_solo" in os.path.basename(f)])

    # Collect all errors
    errors = []  # (pitch, gt_string, gt_fret, pred_string, pred_fret, prev_gt_string, prev_gt_fret)

    for jf in jams_files:
        basename = os.path.splitext(os.path.basename(jf))[0]
        audio = os.path.join(AUDIO_DIR, basename + "_mic.wav")
        if not os.path.exists(audio):
            continue

        jam = jams.load(jf)
        notes = []
        idx = 0
        for ann in jam.annotations:
            if ann.namespace != "note_midi":
                continue
            sn = 6 - idx; idx += 1
            if sn < 1 or sn > 6: continue
            si = 6 - sn
            for obs in ann.data:
                p = int(round(obs.value))
                st = float(obs.time); dur = float(obs.duration)
                fret = p - STANDARD_TUNING[si]
                if 0 <= fret <= MAX_FRET:
                    notes.append({"pitch": p, "start": st, "duration": dur,
                                  "gt_string": sn, "gt_fret": fret})
        notes.sort(key=lambda n: n["start"])

        for i, n in enumerate(notes):
            probs = _predict_string_probs(audio, n["start"], n["pitch"])
            if not probs:
                continue
            # Pick best physically possible
            possible = []
            for si2, op in enumerate(STANDARD_TUNING):
                s2 = 6 - si2; f2 = n["pitch"] - op
                if 0 <= f2 <= 19:
                    possible.append((s2, f2, probs.get(s2, 0.0)))
            if not possible:
                continue
            best = max(possible, key=lambda x: x[2])
            pred_s, pred_f = best[0], best[1]

            if pred_s != n["gt_string"]:
                prev_s = notes[i - 1]["gt_string"] if i > 0 else -1
                prev_f = notes[i - 1]["gt_fret"] if i > 0 else -1
                errors.append({
                    "pitch": n["pitch"], "note": midi_to_name(n["pitch"]),
                    "gt_s": n["gt_string"], "gt_f": n["gt_fret"],
                    "pred_s": pred_s, "pred_f": pred_f,
                    "prev_gt_s": prev_s, "prev_gt_f": prev_f,
                    "cnn_probs": probs,
                })

    print(f"Total errors: {len(errors)}")

    # === Analysis 1: Error patterns (GT string -> Pred string) ===
    print("\n=== Error patterns: GT -> Pred ===")
    pattern_count = Counter()
    for e in errors:
        key = f"S{e['gt_s']}->S{e['pred_s']}"
        pattern_count[key] += 1
    for pat, cnt in pattern_count.most_common(15):
        pct = cnt / len(errors) * 100
        print(f"  {pat}: {cnt} ({pct:.1f}%)")

    # === Analysis 2: Most error-prone pitches ===
    print("\n=== Most error-prone pitches (human plays S2 but CNN picks wrong) ===")
    s2_errors = [e for e in errors if e["gt_s"] == 2]
    pitch_count = Counter(e["pitch"] for e in s2_errors)
    for p, cnt in pitch_count.most_common(10):
        # What does CNN pick instead?
        wrong_pred = Counter(e["pred_s"] for e in s2_errors if e["pitch"] == p)
        pred_str = ", ".join(f"S{s}:{c}" for s, c in wrong_pred.most_common(3))
        print(f"  {midi_to_name(p)}({p}): {cnt} errors -> {pred_str}")

    print("\n=== Most error-prone pitches (human plays S6 but CNN picks wrong) ===")
    s6_errors = [e for e in errors if e["gt_s"] == 6]
    pitch_count6 = Counter(e["pitch"] for e in s6_errors)
    for p, cnt in pitch_count6.most_common(10):
        wrong_pred = Counter(e["pred_s"] for e in s6_errors if e["pitch"] == p)
        pred_str = ", ".join(f"S{s}:{c}" for s, c in wrong_pred.most_common(3))
        print(f"  {midi_to_name(p)}({p}): {cnt} errors -> {pred_str}")

    # === Analysis 3: What fret positions do humans choose vs CNN? ===
    print("\n=== S2 errors: human fret vs CNN fret ===")
    s2_fret_gt = Counter(e["gt_f"] for e in s2_errors)
    s2_fret_pr = Counter(e["pred_f"] for e in s2_errors)
    print(f"  GT frets:   {dict(s2_fret_gt.most_common(8))}")
    print(f"  Pred frets: {dict(s2_fret_pr.most_common(8))}")

    avg_gt = sum(e["gt_f"] for e in s2_errors) / max(len(s2_errors), 1)
    avg_pr = sum(e["pred_f"] for e in s2_errors) / max(len(s2_errors), 1)
    print(f"  Avg GT fret: {avg_gt:.1f}, Avg Pred fret: {avg_pr:.1f}")

    # === Analysis 4: Direction of errors ===
    print("\n=== Error direction summary ===")
    higher = sum(1 for e in errors if e["pred_s"] < e["gt_s"])  # thinner
    lower = sum(1 for e in errors if e["pred_s"] > e["gt_s"])   # thicker
    print(f"  CNN picks THINNER string: {higher} ({higher/len(errors)*100:.1f}%)")
    print(f"  CNN picks THICKER string: {lower} ({lower/len(errors)*100:.1f}%)")

    # === Analysis 5: Context — does error correlate with position change? ===
    print("\n=== Error context: preceding note on same string? ===")
    same_string_prev = sum(1 for e in errors if e["prev_gt_s"] == e["gt_s"])
    diff_string_prev = sum(1 for e in errors if e["prev_gt_s"] != e["gt_s"] and e["prev_gt_s"] != -1)
    print(f"  Same string as prev: {same_string_prev} ({same_string_prev/len(errors)*100:.1f}%)")
    print(f"  Diff string from prev: {diff_string_prev} ({diff_string_prev/len(errors)*100:.1f}%)")


if __name__ == "__main__":
    main()
