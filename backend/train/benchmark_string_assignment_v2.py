"""
benchmark_string_assignment_v2.py — 人間選好マップの重み最適化ベンチマーク
==========================================================================
重みをグリッドサーチして最適な人間選好ボーナスの強度を特定。

Usage:
    python backend/train/benchmark_string_assignment_v2.py
"""
import json, glob, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collections import Counter
import string_assigner

GUITARSET_DIR = r"D:\Music\datasets\GuitarSet\annotation"
from solotab_utils import STANDARD_TUNING
DS_TO_STRING = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}


def load_guitarset(max_files=50):
    jams_files = sorted(glob.glob(os.path.join(GUITARSET_DIR, "*.jams")))[:max_files]
    all_notes = []
    for jf in jams_files:
        with open(jf, "r") as f:
            data = json.load(f)
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
            string_num = DS_TO_STRING[ds_idx]
            open_pitch = STANDARD_TUNING[ds_idx]
            for obs in ann.get("data", []):
                pitch = round(obs.get("value", 0))
                if pitch <= 0:
                    continue
                fret = pitch - open_pitch
                if 0 <= fret <= 22:
                    all_notes.append({"pitch": pitch, "string": string_num, "fret": fret})
    return all_notes


def evaluate(notes, w_human, w_fret_height, w_sweet_spot):
    """指定重みでの弦割り当て精度を評価"""
    string_assigner.WEIGHTS["w_human_pref_bonus"] = w_human
    string_assigner.WEIGHTS["w_fret_height"] = w_fret_height
    string_assigner.WEIGHTS["w_sweet_spot_bonus"] = w_sweet_spot

    correct = 0
    total = 0
    for note in notes:
        candidates = string_assigner.get_possible_positions(note["pitch"])
        if not candidates:
            continue

        best_pos = None
        best_cost = float("inf")
        for s, f in candidates:
            cost = string_assigner._position_cost(s, f, pitch=note["pitch"])
            cost += string_assigner._timbre_cost(s, f, tuple(STANDARD_TUNING))
            if cost < best_cost:
                best_cost = cost
                best_pos = (s, f)

        if best_pos and best_pos[0] == note["string"]:
            correct += 1
        total += 1

    return correct / total * 100 if total > 0 else 0, correct, total


def main():
    print("=== Weight Optimization Benchmark ===")
    notes = load_guitarset(max_files=50)
    print(f"Notes: {len(notes)}")

    # 元の重みを保存
    orig_human = string_assigner.WEIGHTS["w_human_pref_bonus"]
    orig_fret = string_assigner.WEIGHTS["w_fret_height"]
    orig_sweet = string_assigner.WEIGHTS["w_sweet_spot_bonus"]

    print(f"\nOriginal weights: human={orig_human}, fret_height={orig_fret}, sweet_spot={orig_sweet}")

    # ベースライン（人間選好なし）
    acc, c, t = evaluate(notes, 0, orig_fret, orig_sweet)
    print(f"\nBaseline (no human pref): {acc:.1f}% ({c}/{t})")

    # グリッドサーチ
    print(f"\n--- Grid Search ---")
    best_acc = 0
    best_params = {}

    human_weights = [-5, -10, -15, -20, -30, -50, -80, -100]
    fret_weights = [0.0, 0.01, 0.05, 0.1]
    sweet_weights = [0.0, -0.5, -1.0]

    results = []
    for w_h in human_weights:
        for w_f in fret_weights:
            for w_s in sweet_weights:
                acc, c, t = evaluate(notes, w_h, w_f, w_s)
                results.append((acc, w_h, w_f, w_s))
                if acc > best_acc:
                    best_acc = acc
                    best_params = {"human": w_h, "fret": w_f, "sweet": w_s}

    # Top 10結果
    results.sort(reverse=True)
    print(f"\nTop 10 configurations:")
    for i, (acc, wh, wf, ws) in enumerate(results[:10]):
        marker = " <<<" if i == 0 else ""
        print(f"  {acc:5.1f}%  human={wh:5.0f} fret={wf:.2f} sweet={ws:.1f}{marker}")

    print(f"\n=== BEST ===")
    print(f"  Accuracy: {best_acc:.1f}%")
    print(f"  Params: {best_params}")

    # 最適パラメータでの弦別精度
    acc, _, _ = evaluate(notes, best_params["human"], best_params["fret"], best_params["sweet"])
    correct_by_s = Counter()
    total_by_s = Counter()
    for note in notes:
        candidates = string_assigner.get_possible_positions(note["pitch"])
        if not candidates:
            continue
        best_pos = None
        best_cost = float("inf")
        for s, f in candidates:
            cost = string_assigner._position_cost(s, f, pitch=note["pitch"])
            cost += string_assigner._timbre_cost(s, f, tuple(STANDARD_TUNING))
            if cost < best_cost:
                best_cost = cost
                best_pos = (s, f)
        if best_pos:
            total_by_s[note["string"]] += 1
            if best_pos[0] == note["string"]:
                correct_by_s[note["string"]] += 1

    print(f"\nPer-string accuracy (best config):")
    for s in range(1, 7):
        c = correct_by_s.get(s, 0)
        t = total_by_s.get(s, 0)
        a = c / t * 100 if t > 0 else 0
        print(f"  S{s}: {c}/{t} = {a:.1f}%")

    # 元に戻す
    string_assigner.WEIGHTS["w_human_pref_bonus"] = orig_human
    string_assigner.WEIGHTS["w_fret_height"] = orig_fret
    string_assigner.WEIGHTS["w_sweet_spot_bonus"] = orig_sweet


if __name__ == "__main__":
    main()
