"""
Position-Aware CNN-Viterbi
==========================
CNNエラーの根本原因: 人間はポジションを維持するが、CNNは低フレット/細い弦を好む。
→ 「前の音と同じポジション(4フレット幅)にいるなら太い弦を優先する」ロジック追加。

Emission: -log(CNN_prob) + position_bonus
Transition: w_ts * |Δstring| + w_tf * |Δfret| + w_pos * position_shift_penalty
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


def get_candidates(pitch):
    cands = []
    for si, op in enumerate(STANDARD_TUNING):
        s = 6 - si; f = pitch - op
        if 0 <= f <= 19:
            cands.append((s, f))
    return cands


def viterbi_position_aware(notes, cnn_probs_list, w_ts=0.3, w_tf=0.03,
                            w_pos_shift=0.0, w_thick_bonus=0.0,
                            pos_width=4):
    """
    Position-aware Viterbi.
    
    Additional costs:
    - w_pos_shift: penalty when fret jumps > pos_width from previous
    - w_thick_bonus: bonus for choosing thicker string (higher number)
      when CNN confidence is close between candidates
    """
    n = len(notes)
    if n == 0:
        return []

    candidates = []
    for i in range(n):
        cands = get_candidates(notes[i]["pitch"])
        if not cands:
            cands = [(1, 0)]
        candidates.append(cands)

    INF = 1e9
    dp = []
    backptr = []

    # Init
    cands0 = candidates[0]
    probs0 = cnn_probs_list[0] or {}
    costs0 = []
    for (s, f) in cands0:
        p = probs0.get(s, 1e-10)
        emission = -np.log(max(p, 1e-10))
        # Thick string bonus: higher string number = thicker = bonus
        thick = -w_thick_bonus * (s / 6.0)
        costs0.append(emission + thick)
    dp.append(costs0)
    backptr.append([-1] * len(cands0))

    for i in range(1, n):
        cands_i = candidates[i]
        probs_i = cnn_probs_list[i] or {}
        costs_i = []
        bp_i = []

        # Time gap (for position shift feasibility)
        dt = notes[i]["start"] - notes[i - 1]["start"]

        for j, (s_j, f_j) in enumerate(cands_i):
            p = probs_i.get(s_j, 1e-10)
            emission = -np.log(max(p, 1e-10))
            thick = -w_thick_bonus * (s_j / 6.0)

            best_cost = INF
            best_prev = 0
            for k, (s_k, f_k) in enumerate(candidates[i - 1]):
                # String switch cost
                trans_string = w_ts * abs(s_j - s_k)
                # Fret distance cost
                trans_fret = w_tf * abs(f_j - f_k)
                # Position shift: penalty for jumping > pos_width frets
                fret_jump = abs(f_j - f_k)
                pos_penalty = 0.0
                if fret_jump > pos_width and dt < 0.5:  # only penalize fast passages
                    pos_penalty = w_pos_shift * (fret_jump - pos_width)

                cost = dp[i - 1][k] + trans_string + trans_fret + pos_penalty + emission + thick
                if cost < best_cost:
                    best_cost = cost
                    best_prev = k
            costs_i.append(best_cost)
            bp_i.append(best_prev)

        dp.append(costs_i)
        backptr.append(bp_i)

    # Backtrace
    last_costs = dp[-1]
    best_j = min(range(len(last_costs)), key=lambda j: last_costs[j])
    result = [None] * n
    j = best_j
    for i in range(n - 1, -1, -1):
        s, f = candidates[i][j]
        result[i] = (s, f)
        j = backptr[i][j]
    return result


def evaluate(data, w_ts, w_tf, w_pos, w_thick):
    _load_string_classifier()
    correct = 0; total = 0
    ps_correct = Counter(); ps_total = Counter()

    for file_data in data:
        audio = file_data["audio"]
        notes = file_data["notes"]
        cnn_probs_list = []
        for no in notes:
            probs = _predict_string_probs(audio, no["start"], no["pitch"])
            cnn_probs_list.append(probs)

        result = viterbi_position_aware(notes, cnn_probs_list, w_ts, w_tf, w_pos, w_thick)
        for i, (pred_s, pred_f) in enumerate(result):
            gt_s = notes[i]["gt_string"]
            total += 1; ps_total[gt_s] += 1
            if pred_s == gt_s:
                correct += 1; ps_correct[gt_s] += 1

    return correct, total, ps_correct, ps_total


def main():
    print("=== Position-Aware CNN-Viterbi ===\n")
    data = load_solo_data(max_files=60)
    print(f"Files: {len(data)}\n")

    configs = [
        # (name, w_ts, w_tf, w_pos_shift, w_thick_bonus)
        ("Baseline CNN-Vit",      0.3,  0.03, 0.0,  0.0),
        ("+ pos_shift=0.1",       0.3,  0.03, 0.1,  0.0),
        ("+ pos_shift=0.3",       0.3,  0.03, 0.3,  0.0),
        ("+ pos_shift=0.5",       0.3,  0.03, 0.5,  0.0),
        ("+ thick=0.1",           0.3,  0.03, 0.0,  0.1),
        ("+ thick=0.3",           0.3,  0.03, 0.0,  0.3),
        ("+ thick=0.5",           0.3,  0.03, 0.0,  0.5),
        ("+ thick=1.0",           0.3,  0.03, 0.0,  1.0),
        ("pos=0.3 thick=0.3",     0.3,  0.03, 0.3,  0.3),
        ("pos=0.3 thick=0.5",     0.3,  0.03, 0.3,  0.5),
        ("pos=0.5 thick=0.5",     0.3,  0.03, 0.5,  0.5),
        ("pos=0.1 thick=0.5",     0.3,  0.03, 0.1,  0.5),
        ("pos=0.1 thick=1.0",     0.3,  0.03, 0.1,  1.0),
        ("pos=0.3 thick=1.0",     0.3,  0.03, 0.3,  1.0),
    ]

    best_acc = 0; best_name = ""; best_ps = None; best_pst = None

    print(f"{'Config':<25s} {'Acc':>8s} {'S1':>6s} {'S2':>6s} {'S3':>6s} {'S4':>6s} {'S5':>6s} {'S6':>6s}")
    print("-" * 80)
    for name, wts, wtf, wpos, wthick in configs:
        c, t, ps, pst = evaluate(data, wts, wtf, wpos, wthick)
        acc = c / t * 100 if t > 0 else 0
        marker = " *" if acc > best_acc else ""
        if acc > best_acc:
            best_acc = acc; best_name = name; best_ps = ps; best_pst = pst

        per_s = []
        for s in range(1, 7):
            sc = ps.get(s, 0); st = pst.get(s, 0)
            per_s.append(f"{sc/st*100:5.1f}" if st > 0 else "  N/A")
        print(f"{name:<25s} {acc:7.1f}%{marker} {' '.join(per_s)}")

    print(f"\nBest: {best_name} ({best_acc:.1f}%)")


if __name__ == "__main__":
    main()
