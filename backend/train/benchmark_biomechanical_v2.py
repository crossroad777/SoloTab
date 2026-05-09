"""
Biomechanical Viterbi v2: S6回帰修正版
======================================
v1では95.8%達成したがS6が84%に下がった。
原因: S6(E2)は開放弦が多く、ポジション制約が逆効果。
修正: 開放弦はポジション制約を緩和 + 低弦(S5/S6)の遷移コスト調整
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

FINGER_EASE = {0: 0.40, 1: 0.35, 2: 0.30, 3: 0.25, 4: 0.10}

MAX_SPAN = {
    (1, 2): 4, (1, 3): 5, (1, 4): 6,
    (2, 3): 3, (2, 4): 4, (3, 4): 3,
}


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
            if f == 0:
                cands.append((s, f, 0))
            else:
                for finger in range(1, 5):
                    cands.append((s, f, finger))
    return cands


def transition_cost_v2(s1, f1, fig1, s2, f2, fig2, dt,
                        w_string=0.3, w_pos=0.5, w_finger_order=10.0,
                        w_same_finger=2.0, w_stretch=1.0, w_open_bonus=0.0):
    cost = 0.0
    cost += w_string * abs(s1 - s2)

    # 開放弦の場合はポジション制約を大幅緩和
    if fig1 == 0 or fig2 == 0:
        # 開放弦ボーナス: 開放弦は手の位置に関係なく弾ける
        if fig2 == 0:
            cost -= w_open_bonus
        return cost

    pos1 = f1 - (fig1 - 1)
    pos2 = f2 - (fig2 - 1)
    pos_shift = abs(pos2 - pos1)
    if pos_shift > 0:
        time_factor = 1.0 / max(dt, 0.05)
        cost += w_pos * pos_shift * min(time_factor, 5.0)

    if fig1 == fig2 and f1 != f2:
        cost += w_same_finger

    if s1 == s2:
        if (fig2 > fig1 and f2 < f1) or (fig2 < fig1 and f2 > f1):
            cost += w_finger_order

    if fig1 != fig2 and fig1 > 0 and fig2 > 0:
        fret_span = abs(f2 - f1)
        pair = (min(fig1, fig2), max(fig1, fig2))
        max_s = MAX_SPAN.get(pair, 6)
        if fret_span > max_s:
            cost += w_stretch * (fret_span - max_s)

    return cost


def viterbi_bio_v2(notes, cnn_probs_list, w_string, w_pos, w_ease, w_open):
    n = len(notes)
    if n == 0:
        return []
    candidates = [get_candidates(notes[i]["pitch"]) or [(1, 0, 0)] for i in range(n)]
    INF = 1e9
    dp = []; backptr = []

    probs0 = cnn_probs_list[0] or {}
    costs0 = []
    for (s, f, fig) in candidates[0]:
        p = probs0.get(s, 1e-10)
        emission = -np.log(max(p, 1e-10))
        ease = -w_ease * FINGER_EASE.get(fig, 0.1)
        costs0.append(emission + ease)
    dp.append(costs0)
    backptr.append([-1] * len(candidates[0]))

    for i in range(1, n):
        probs_i = cnn_probs_list[i] or {}
        dt = notes[i]["start"] - notes[i - 1]["start"]
        costs_i = []; bp_i = []
        for j, (s_j, f_j, fig_j) in enumerate(candidates[i]):
            p = probs_i.get(s_j, 1e-10)
            emission = -np.log(max(p, 1e-10))
            ease = -w_ease * FINGER_EASE.get(fig_j, 0.1)
            best_cost = INF; best_prev = 0
            for k, (s_k, f_k, fig_k) in enumerate(candidates[i - 1]):
                trans = transition_cost_v2(s_k, f_k, fig_k, s_j, f_j, fig_j, dt,
                                           w_string=w_string, w_pos=w_pos,
                                           w_open_bonus=w_open)
                cost = dp[i - 1][k] + trans + emission + ease
                if cost < best_cost:
                    best_cost = cost; best_prev = k
            costs_i.append(best_cost); bp_i.append(best_prev)
        dp.append(costs_i); backptr.append(bp_i)

    last_costs = dp[-1]
    best_j = min(range(len(last_costs)), key=lambda j: last_costs[j])
    result = [None] * n
    j = best_j
    for i in range(n - 1, -1, -1):
        result[i] = candidates[i][j]
        j = backptr[i][j]
    return result


def evaluate(data, ws, wp, we, wo):
    _load_string_classifier()
    correct = 0; total = 0
    ps_correct = Counter(); ps_total = Counter()
    for fd in data:
        notes = fd["notes"]
        cnn_probs = [_predict_string_probs(fd["audio"], n["start"], n["pitch"]) for n in notes]
        result = viterbi_bio_v2(notes, cnn_probs, ws, wp, we, wo)
        for i, (ps2, pf, pfig) in enumerate(result):
            gt = notes[i]["gt_string"]
            total += 1; ps_total[gt] += 1
            if ps2 == gt:
                correct += 1; ps_correct[gt] += 1
    return correct, total, ps_correct, ps_total


def main():
    print("=== Biomechanical Viterbi v2 (S6 fix) ===\n")
    data = load_solo_data(60)
    print(f"Files: {len(data)}\n")

    configs = [
        ("v1 best (baseline)",     0.3, 0.5, 0.5, 0.0),
        ("+ open=0.3",             0.3, 0.5, 0.5, 0.3),
        ("+ open=0.5",             0.3, 0.5, 0.5, 0.5),
        ("+ open=1.0",             0.3, 0.5, 0.5, 1.0),
        ("+ open=2.0",             0.3, 0.5, 0.5, 2.0),
        ("pos=0.3 open=0.5",       0.3, 0.3, 0.5, 0.5),
        ("pos=0.3 open=1.0",       0.3, 0.3, 0.5, 1.0),
        ("pos=0.1 open=0.5",       0.3, 0.1, 0.5, 0.5),
        ("pos=0.1 open=1.0",       0.3, 0.1, 0.5, 1.0),
        ("pos=0.5 open=1.0 e=0.3", 0.3, 0.5, 0.3, 1.0),
    ]
    best_acc = 0; best_name = ""

    print(f"{'Config':<28s} {'Acc':>8s} {'S1':>6s} {'S2':>6s} {'S3':>6s} {'S4':>6s} {'S5':>6s} {'S6':>6s}")
    print("-" * 80)
    for name, ws, wp, we, wo in configs:
        c, t, ps, pst = evaluate(data, ws, wp, we, wo)
        acc = c / t * 100
        marker = " *" if acc > best_acc else ""
        if acc > best_acc:
            best_acc = acc; best_name = name; bps = ps; bpst = pst
        per = [f"{ps.get(s,0)/pst.get(s,1)*100:5.1f}" for s in range(1, 7)]
        print(f"{name:<28s} {acc:7.1f}%{marker} {' '.join(per)}")

    print(f"\nBest: {best_name} ({best_acc:.1f}%)")
    for s in range(1, 7):
        sn = ["E4","B3","G3","D3","A2","E2"][s-1]
        print(f"  S{s}({sn}): {bps.get(s,0)}/{bpst.get(s,0)} = {bps.get(s,0)/bpst.get(s,1)*100:.1f}%")


if __name__ == "__main__":
    main()
