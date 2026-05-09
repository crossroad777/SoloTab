"""
Confidence-Gated Position Correction
======================================
CNNの確信度が低い(top2の差が小さい)場合にのみ、
ポジション制約で太い弦を選ばせる。

方針: CNN確信度高い→CNNを信頼、低い→指の形(ポジション)で決める
"""
import sys, os, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jams
import numpy as np
from collections import Counter
from string_assigner import (
    _load_string_classifier, _predict_string_probs,
    STANDARD_TUNING, MAX_FRET
)

ANNOTATION_DIR = r"D:\Music\Datasets\GuitarSet\annotation"
AUDIO_DIR = r"D:\Music\Datasets\GuitarSet\audio_mono-mic"


def load_solo_data(max_files=None):
    jams_files = sorted([f for f in glob.glob(os.path.join(ANNOTATION_DIR, "*.jams"))
                         if "_solo" in os.path.basename(f)])
    if max_files:
        jams_files = jams_files[:max_files]
    all_data = []
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
        if notes:
            notes.sort(key=lambda n: n["start"])
            all_data.append({"audio": audio, "basename": basename, "notes": notes})
    return all_data


def evaluate_confidence_gated(data, conf_threshold=0.3, pos_width=4):
    """
    CNN確信度が高い(top prob - 2nd prob > threshold) → CNNを信頼
    CNN確信度が低い → 前の音と同じポジション内の候補を優先(太い弦ボーナス)
    """
    _load_string_classifier()
    correct = 0; total = 0
    ps_correct = Counter(); ps_total = Counter()
    gated_correct = 0; gated_total = 0; gated_applied = 0

    for file_data in data:
        audio = file_data["audio"]
        notes = file_data["notes"]
        prev_fret = -1  # 前の音のフレット

        for i, n in enumerate(notes):
            probs = _predict_string_probs(audio, n["start"], n["pitch"])
            if not probs:
                continue

            # 物理的候補
            possible = []
            for si, op in enumerate(STANDARD_TUNING):
                s = 6 - si; f = n["pitch"] - op
                if 0 <= f <= 19:
                    possible.append((s, f, probs.get(s, 0.0)))
            if not possible:
                continue

            # CNN top choice
            possible.sort(key=lambda x: -x[2])
            top_prob = possible[0][2]
            second_prob = possible[1][2] if len(possible) > 1 else 0.0
            confidence = top_prob - second_prob

            if confidence >= conf_threshold or prev_fret < 0:
                # High confidence: trust CNN
                best_s, best_f = possible[0][0], possible[0][1]
            else:
                # Low confidence: prefer candidate in same position as previous note
                gated_applied += 1
                best_s, best_f = possible[0][0], possible[0][1]
                best_score = -1

                for (s, f, p) in possible:
                    # Score: CNN prob + position bonus
                    fret_dist = abs(f - prev_fret) if prev_fret >= 0 else 0
                    in_position = 1.0 if fret_dist <= pos_width else 0.0
                    # Combined score: CNN + position + thick string preference
                    score = p + 0.3 * in_position + 0.1 * (s / 6.0)
                    if score > best_score:
                        best_score = score
                        best_s, best_f = s, f

            gt_s = n["gt_string"]
            total += 1; ps_total[gt_s] += 1
            if best_s == gt_s:
                correct += 1; ps_correct[gt_s] += 1
                if confidence < conf_threshold and prev_fret >= 0:
                    gated_correct += 1
            if confidence < conf_threshold and prev_fret >= 0:
                gated_total += 1

            prev_fret = best_f

    return correct, total, ps_correct, ps_total, gated_applied, gated_correct, gated_total


def main():
    print("=== Confidence-Gated Position Correction ===\n")
    data = load_solo_data(max_files=60)
    print(f"Files: {len(data)}\n")

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

    best_acc = 0

    print(f"{'Threshold':>10s} {'Acc':>8s} {'Gated':>8s} {'GateAcc':>8s} {'S1':>6s} {'S2':>6s} {'S3':>6s} {'S4':>6s} {'S5':>6s} {'S6':>6s}")
    print("-" * 90)

    for th in thresholds:
        c, t, ps, pst, ga, gc, gt2 = evaluate_confidence_gated(data, conf_threshold=th)
        acc = c / t * 100 if t > 0 else 0
        gate_acc = gc / gt2 * 100 if gt2 > 0 else 0
        marker = " *" if acc > best_acc else ""
        if acc > best_acc:
            best_acc = acc

        per_s = []
        for s in range(1, 7):
            sc = ps.get(s, 0); st = pst.get(s, 0)
            per_s.append(f"{sc/st*100:5.1f}" if st > 0 else "  N/A")
        print(f"{th:10.2f} {acc:7.1f}%{marker} {ga:7d}  {gate_acc:6.1f}% {' '.join(per_s)}")

    print(f"\nBest: {best_acc:.1f}%")


if __name__ == "__main__":
    main()
