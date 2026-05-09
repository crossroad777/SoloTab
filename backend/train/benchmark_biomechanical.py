"""
Biomechanical Viterbi: 生体力学的制約付きViterbi
================================================
指の関節は一方向にしか曲がらない。指の順序は不変。
手のポジション(4フレット幅)から外れるには手全体を動かす必要がある。

このViterbiは弦+フレットだけでなく「指番号」も状態に含める。
- state = (string, fret, finger) ただしfinger=1-4
- 遷移制約: 同時刻の音は指が交差しない(fret順 = finger順)
- emissionコスト: CNN確率 (あれば)
- 遷移コスト: ポジション移動 + 指ストレッチ + 指の使いやすさ
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

# 指の難易度 (Hori 2016)
FINGER_EASE = {1: 0.35, 2: 0.30, 3: 0.25, 4: 0.10}

# 指ペア間の最大スパン (フレット数)
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
    """物理的に可能な(string, fret)を列挙。開放弦はfinger=0。"""
    cands = []
    for si, op in enumerate(STANDARD_TUNING):
        s = 6 - si; f = pitch - op
        if 0 <= f <= 19:
            if f == 0:
                cands.append((s, f, 0))  # 開放弦
            else:
                # フレット1-4は指1-4で弾ける。5以上も指1-4のどれか。
                for finger in range(1, 5):
                    cands.append((s, f, finger))
    return cands


def transition_cost(s1, f1, fig1, s2, f2, fig2, dt,
                    w_string=0.3, w_pos=0.5, w_finger_order=10.0,
                    w_same_finger=2.0, w_stretch=1.0):
    """
    遷移コスト: 前の音(s1,f1,fig1) → 次の音(s2,f2,fig2)
    
    制約:
    - 同じ指で異なるフレットはペナルティ大(速いパッセージでは不可能)
    - 指の順序違反は巨大ペナルティ(物理的に不可能)
    - ポジション移動(手全体の移動)はコスト
    - ストレッチ(指間スパン)はコスト
    """
    cost = 0.0

    # 弦の移動コスト
    cost += w_string * abs(s1 - s2)

    # 開放弦の場合はポジション制約なし
    if fig1 == 0 or fig2 == 0:
        return cost

    # ポジション移動: 人差し指の推定位置の差
    pos1 = f1 - (fig1 - 1)  # 人差し指のフレット推定
    pos2 = f2 - (fig2 - 1)
    pos_shift = abs(pos2 - pos1)
    if pos_shift > 0:
        # 速いパッセージでの大きなポジション移動はペナルティ大
        time_factor = 1.0 / max(dt, 0.05)  # 速いほどペナルティ
        cost += w_pos * pos_shift * min(time_factor, 5.0)

    # 同じ指で異なるフレット → ペナルティ
    if fig1 == fig2 and f1 != f2:
        cost += w_same_finger

    # 指の順序制約: 同じ弦上で指の順序とフレットの順序が矛盾 → 巨大ペナルティ
    # (異なる弦では指交差は許容される場合がある)
    if s1 == s2:
        if (fig2 > fig1 and f2 < f1) or (fig2 < fig1 and f2 > f1):
            cost += w_finger_order

    # ストレッチペナルティ
    if fig1 != fig2:
        fret_span = abs(f2 - f1)
        pair = (min(fig1, fig2), max(fig1, fig2))
        max_s = MAX_SPAN.get(pair, 6)
        if fret_span > max_s:
            cost += w_stretch * (fret_span - max_s)

    return cost


def viterbi_biomechanical(notes, cnn_probs_list,
                           w_string=0.3, w_pos=0.5, w_finger_ease=0.1):
    """
    生体力学制約付きViterbi。状態=(string, fret, finger)。
    """
    n = len(notes)
    if n == 0:
        return []

    # 各ノートの候補
    candidates = []
    for i in range(n):
        cands = get_candidates(notes[i]["pitch"])
        if not cands:
            cands = [(1, 0, 0)]
        candidates.append(cands)

    INF = 1e9
    dp = []
    backptr = []

    # Init
    cands0 = candidates[0]
    probs0 = cnn_probs_list[0] or {}
    costs0 = []
    for (s, f, fig) in cands0:
        p = probs0.get(s, 1e-10)
        emission = -np.log(max(p, 1e-10))
        # 指の使いやすさボーナス
        ease = -w_finger_ease * FINGER_EASE.get(fig, 0.1)
        costs0.append(emission + ease)
    dp.append(costs0)
    backptr.append([-1] * len(cands0))

    # Forward
    for i in range(1, n):
        cands_i = candidates[i]
        probs_i = cnn_probs_list[i] or {}
        dt = notes[i]["start"] - notes[i - 1]["start"]
        costs_i = []
        bp_i = []

        for j, (s_j, f_j, fig_j) in enumerate(cands_i):
            p = probs_i.get(s_j, 1e-10)
            emission = -np.log(max(p, 1e-10))
            ease = -w_finger_ease * FINGER_EASE.get(fig_j, 0.1)

            best_cost = INF
            best_prev = 0
            for k, (s_k, f_k, fig_k) in enumerate(candidates[i - 1]):
                trans = transition_cost(s_k, f_k, fig_k, s_j, f_j, fig_j, dt,
                                        w_string=w_string, w_pos=w_pos)
                cost = dp[i - 1][k] + trans + emission + ease
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
        s, f, fig = candidates[i][j]
        result[i] = (s, f, fig)
        j = backptr[i][j]
    return result


def evaluate(data, w_string=0.3, w_pos=0.5, w_finger_ease=0.1):
    _load_string_classifier()
    correct = 0; total = 0
    ps_correct = Counter(); ps_total = Counter()

    for file_data in data:
        audio = file_data["audio"]
        notes = file_data["notes"]
        cnn_probs_list = [_predict_string_probs(audio, no["start"], no["pitch"])
                          for no in notes]

        result = viterbi_biomechanical(notes, cnn_probs_list,
                                        w_string, w_pos, w_finger_ease)
        for i, (pred_s, pred_f, pred_fig) in enumerate(result):
            gt_s = notes[i]["gt_string"]
            total += 1; ps_total[gt_s] += 1
            if pred_s == gt_s:
                correct += 1; ps_correct[gt_s] += 1

    return correct, total, ps_correct, ps_total


def main():
    print("=== Biomechanical Viterbi Benchmark ===\n")
    data = load_solo_data(max_files=60)
    print(f"Files: {len(data)}\n")

    configs = [
        # (name, w_string, w_pos, w_finger_ease)
        ("CNN only (no bio)",       0.0, 0.0, 0.0),
        ("bio: w_pos=0.1",          0.3, 0.1, 0.1),
        ("bio: w_pos=0.3",          0.3, 0.3, 0.1),
        ("bio: w_pos=0.5",          0.3, 0.5, 0.1),
        ("bio: w_pos=1.0",          0.3, 1.0, 0.1),
        ("bio: w_pos=0.3 ease=0.3", 0.3, 0.3, 0.3),
        ("bio: w_pos=0.5 ease=0.3", 0.3, 0.5, 0.3),
        ("bio: w_pos=0.5 ease=0.5", 0.3, 0.5, 0.5),
        ("bio: w_pos=1.0 ease=0.5", 0.3, 1.0, 0.5),
    ]

    best_acc = 0; best_name = ""

    print(f"{'Config':<28s} {'Acc':>8s} {'S1':>6s} {'S2':>6s} {'S3':>6s} {'S4':>6s} {'S5':>6s} {'S6':>6s}")
    print("-" * 80)
    for name, ws, wp, we in configs:
        c, t, ps, pst = evaluate(data, ws, wp, we)
        acc = c / t * 100 if t > 0 else 0
        marker = " *" if acc > best_acc else ""
        if acc > best_acc:
            best_acc = acc; best_name = name

        per_s = []
        for s in range(1, 7):
            sc = ps.get(s, 0); st = pst.get(s, 0)
            per_s.append(f"{sc/st*100:5.1f}" if st > 0 else "  N/A")
        print(f"{name:<28s} {acc:7.1f}%{marker} {' '.join(per_s)}")

    print(f"\nBest: {best_name} ({best_acc:.1f}%)")


if __name__ == "__main__":
    main()
