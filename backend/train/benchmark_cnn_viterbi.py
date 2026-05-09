"""
CNN-Viterbi Hybrid: CNN確率をViterbi DPのemissionに統合
========================================================
CNN単音判定(93.3%) + Viterbi遷移コストで前後文脈を加味。
人間は弦を切り替えるときにコストがかかる → 遷移コストが効く。
"""
import sys, os, glob, json
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
    """物理的に可能な(string, fret)を列挙"""
    cands = []
    for si, op in enumerate(STANDARD_TUNING):
        s = 6 - si
        f = pitch - op
        if 0 <= f <= 19:
            cands.append((s, f))
    return cands


def viterbi_cnn(notes, cnn_probs_list, w_trans_string=0.0, w_trans_fret=0.0):
    """
    CNN確率をemissionコストに、弦/フレット遷移をtransitionコストに使うViterbi。
    
    emission_cost = -log(cnn_prob)  (CNN確率が高い → コスト低)
    transition_cost = w_ts * |Δstring| + w_tf * |Δfret|
    """
    n = len(notes)
    if n == 0:
        return []

    # 各ノートの候補
    candidates = []
    for i in range(n):
        cands = get_candidates(notes[i]["pitch"])
        if not cands:
            cands = [(1, 0)]  # fallback
        candidates.append(cands)

    # Viterbi
    INF = 1e9
    # dp[i][j] = 候補jまでの最小コスト
    dp = []
    backptr = []

    # Init
    cands0 = candidates[0]
    probs0 = cnn_probs_list[0] or {}
    costs0 = []
    for (s, f) in cands0:
        p = probs0.get(s, 1e-10)
        emission = -np.log(max(p, 1e-10))
        costs0.append(emission)
    dp.append(costs0)
    backptr.append([-1] * len(cands0))

    # Forward
    for i in range(1, n):
        cands_i = candidates[i]
        probs_i = cnn_probs_list[i] or {}
        costs_i = []
        bp_i = []

        for j, (s_j, f_j) in enumerate(cands_i):
            p = probs_i.get(s_j, 1e-10)
            emission = -np.log(max(p, 1e-10))

            best_cost = INF
            best_prev = 0
            for k, (s_k, f_k) in enumerate(candidates[i - 1]):
                trans = (w_trans_string * abs(s_j - s_k)
                         + w_trans_fret * abs(f_j - f_k))
                cost = dp[i - 1][k] + trans + emission
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


def evaluate(data, w_ts, w_tf):
    _load_string_classifier()
    correct = 0; total = 0
    ps_correct = Counter(); ps_total = Counter()

    for file_data in data:
        audio = file_data["audio"]
        notes = file_data["notes"]

        # Batch CNN probs
        cnn_probs_list = []
        for n in notes:
            probs = _predict_string_probs(audio, n["start"], n["pitch"])
            cnn_probs_list.append(probs)

        # Viterbi
        result = viterbi_cnn(notes, cnn_probs_list, w_ts, w_tf)

        for i, (pred_s, pred_f) in enumerate(result):
            gt_s = notes[i]["gt_string"]
            total += 1
            ps_total[gt_s] += 1
            if pred_s == gt_s:
                correct += 1
                ps_correct[gt_s] += 1

    return correct, total, ps_correct, ps_total


def main():
    print("=== CNN-Viterbi Hybrid Benchmark ===\n")
    data = load_solo_data(max_files=60)
    total_notes = sum(len(d["notes"]) for d in data)
    print(f"Files: {len(data)}, Notes: {total_notes}\n")

    configs = [
        # (name, w_trans_string, w_trans_fret)
        ("CNN only (w_ts=0, w_tf=0)",     0.0,  0.0),
        ("w_ts=0.5, w_tf=0",              0.5,  0.0),
        ("w_ts=1.0, w_tf=0",              1.0,  0.0),
        ("w_ts=2.0, w_tf=0",              2.0,  0.0),
        ("w_ts=3.0, w_tf=0",              3.0,  0.0),
        ("w_ts=1.0, w_tf=0.05",           1.0,  0.05),
        ("w_ts=2.0, w_tf=0.05",           2.0,  0.05),
        ("w_ts=2.0, w_tf=0.1",            2.0,  0.1),
        ("w_ts=3.0, w_tf=0.05",           3.0,  0.05),
        ("w_ts=5.0, w_tf=0.1",            5.0,  0.1),
    ]

    best_acc = 0; best_name = ""; best_ps = None; best_pst = None

    print(f"{'Config':<30s} {'Acc':>8s}")
    print("-" * 40)
    for name, wts, wtf in configs:
        c, t, ps, pst = evaluate(data, wts, wtf)
        acc = c / t * 100 if t > 0 else 0
        marker = " ★" if acc > best_acc else ""
        if acc > best_acc:
            best_acc = acc; best_name = name; best_ps = ps; best_pst = pst
        print(f"{name:<30s} {acc:7.1f}%{marker}")

    print(f"\n=== Best: {best_name} ({best_acc:.1f}%) ===")
    for s in range(1, 7):
        sn = ["E4", "B3", "G3", "D3", "A2", "E2"][s - 1]
        c = best_ps.get(s, 0); t = best_pst.get(s, 0)
        a = c / t * 100 if t > 0 else 0
        print(f"  S{s}({sn}): {c}/{t} = {a:.1f}%")


if __name__ == "__main__":
    main()
