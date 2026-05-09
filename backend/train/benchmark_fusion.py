"""
CNN + Human Preference融合ベンチマーク
=====================================
CNN弦確率(91.9%) + 人間選好マップ + 物理制約のスコア融合で95%+を目指す。

Scoring: score(s,f) = w_cnn * CNN_prob(s) + w_human * HumanPref(pitch,s,f) + w_phys * PhysConstraint(s,f)
"""
import sys, os, glob, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jams
from collections import Counter
from string_assigner import (
    _load_string_classifier, _predict_string_probs,
    _load_human_preference, STANDARD_TUNING, MAX_FRET
)

ANNOTATION_DIR = r"D:\Music\Datasets\GuitarSet\annotation"
AUDIO_DIR = r"D:\Music\Datasets\GuitarSet\audio_mono-mic"


def load_solo_gt(max_files=None):
    """GuitarSet soloファイルのGT読み込み"""
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
            sn = 6 - idx
            idx += 1
            if sn < 1 or sn > 6:
                continue
            si = 6 - sn
            for obs in ann.data:
                p = int(round(obs.value))
                st = float(obs.time)
                dur = float(obs.duration)
                fret = p - STANDARD_TUNING[si]
                if 0 <= fret <= MAX_FRET:
                    notes.append({
                        "pitch": p, "start": st, "duration": dur,
                        "gt_string": sn, "gt_fret": fret,
                    })
        if notes:
            notes.sort(key=lambda n: n["start"])
            all_data.append({"audio": audio, "basename": basename, "notes": notes})
    return all_data


def evaluate_fusion(data, w_cnn=1.0, w_human=0.0, w_open=0.0, w_low_fret=0.0):
    """
    CNN + Human Preference の融合スコアで弦を選択。
    
    score(s,f) = w_cnn * cnn_prob(s) 
               + w_human * human_pref_prob(pitch, s_idmt, f)
               + w_open * (1 if f==0 else 0)
               + w_low_fret * max(0, 1 - f/12)
    """
    pref = _load_human_preference()
    _load_string_classifier()

    correct = 0
    total = 0
    no_probs = 0
    per_string = Counter()
    per_string_total = Counter()

    for file_data in data:
        audio = file_data["audio"]
        for n in file_data["notes"]:
            pitch = n["pitch"]
            gt_s = n["gt_string"]

            # CNN確率
            cnn_probs = _predict_string_probs(audio, n["start"], pitch)
            if not cnn_probs:
                no_probs += 1
                continue

            # 物理的に可能なポジション
            candidates = []
            for si, op in enumerate(STANDARD_TUNING):
                s = 6 - si  # 標準形式: 6=低E, 1=高E
                f = pitch - op
                if 0 <= f <= 19:
                    # CNN score
                    cnn_score = cnn_probs.get(s, 0.0)

                    # Human preference score
                    human_score = 0.0
                    if pref and w_human != 0:
                        pitch_data = pref.get(str(pitch))
                        if pitch_data:
                            prob_map = pitch_data.get("prob", {})
                            # 弦番号変換: 標準形式 → IDMT形式
                            map_s = 7 - s
                            key = f"{map_s}_{f}"
                            human_score = prob_map.get(key, 0.0)

                    # Open string bonus
                    open_score = 1.0 if f == 0 else 0.0

                    # Low fret bonus
                    low_fret_score = max(0.0, 1.0 - f / 12.0)

                    total_score = (w_cnn * cnn_score
                                   + w_human * human_score
                                   + w_open * open_score
                                   + w_low_fret * low_fret_score)

                    candidates.append((s, f, total_score))

            if not candidates:
                continue

            best = max(candidates, key=lambda x: x[2])
            total += 1
            per_string_total[gt_s] += 1
            if best[0] == gt_s:
                correct += 1
                per_string[gt_s] += 1

    return correct, total, no_probs, per_string, per_string_total


def main():
    print("=== CNN + Human Preference Fusion Benchmark ===\n")

    data = load_solo_gt(max_files=60)
    total_notes = sum(len(d["notes"]) for d in data)
    print(f"Files: {len(data)}, Notes: {total_notes}")

    configs = [
        # (name, w_cnn, w_human, w_open, w_low_fret)
        ("CNN only",              1.0,  0.0,  0.0,  0.0),
        ("CNN + human 0.1",       1.0,  0.1,  0.0,  0.0),
        ("CNN + human 0.2",       1.0,  0.2,  0.0,  0.0),
        ("CNN + human 0.3",       1.0,  0.3,  0.0,  0.0),
        ("CNN + human 0.5",       1.0,  0.5,  0.0,  0.0),
        ("CNN + human 1.0",       1.0,  1.0,  0.0,  0.0),
        ("CNN + human 0.3 +open", 1.0,  0.3,  0.05, 0.0),
        ("CNN + human 0.3 +low",  1.0,  0.3,  0.0,  0.02),
        ("CNN + h0.3 +open +low", 1.0,  0.3,  0.05, 0.02),
    ]

    print(f"\n{'Config':<30s} {'Correct':>8s} {'Total':>6s} {'Acc':>8s}")
    print("-" * 55)

    best_acc = 0
    best_name = ""

    for name, wc, wh, wo, wl in configs:
        c, t, skip, ps, pst = evaluate_fusion(data, wc, wh, wo, wl)
        acc = c / t * 100 if t > 0 else 0
        marker = " ★" if acc > best_acc else ""
        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_ps = ps
            best_pst = pst
        print(f"{name:<30s} {c:8d} {t:6d} {acc:7.1f}%{marker}")

    # Best config detail
    print(f"\n=== Best: {best_name} ({best_acc:.1f}%) ===")
    print("Per-string:")
    for s in range(1, 7):
        sname = ["E4", "B3", "G3", "D3", "A2", "E2"][s - 1]
        c = best_ps.get(s, 0)
        t = best_pst.get(s, 0)
        a = c / t * 100 if t > 0 else 0
        print(f"  S{s}({sname}): {c}/{t} = {a:.1f}%")


if __name__ == "__main__":
    main()
