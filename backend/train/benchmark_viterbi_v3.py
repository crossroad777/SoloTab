"""
benchmark_viterbi_v3.py — Viterbi DP全体での弦割り当てベンチマーク
=================================================================
単音ではなく、GuitarSetのフレーズをViterbi DPに通して
全体最適化での弦割り当て精度を測定する。

Usage:
    python backend/train/benchmark_viterbi_v3.py
"""
import json, glob, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collections import Counter
import string_assigner

GUITARSET_DIR = r"D:\Music\datasets\GuitarSet\annotation"
from solotab_utils import STANDARD_TUNING
DS_TO_STRING = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}


def load_guitarset_phrases(max_files=50):
    """GuitarSetからファイル単位のフレーズを構築"""
    jams_files = sorted(glob.glob(os.path.join(GUITARSET_DIR, "*.jams")))[:max_files]
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

            string_num = DS_TO_STRING[ds_idx]
            open_pitch = STANDARD_TUNING[ds_idx]

            for obs in ann.get("data", []):
                pitch = round(obs.get("value", 0))
                if pitch <= 0:
                    continue
                fret = pitch - open_pitch
                if 0 <= fret <= 22:
                    file_notes.append({
                        "pitch": pitch,
                        "string": string_num,
                        "fret": fret,
                        "start": obs.get("time", 0),
                        "duration": obs.get("duration", 0),
                    })

        if file_notes:
            file_notes.sort(key=lambda n: n["start"])
            phrases.append((os.path.basename(jf), file_notes))

    return phrases


def evaluate_viterbi(phrases, w_human):
    """Viterbi DPでの弦割り当て精度を評価"""
    string_assigner.WEIGHTS["w_human_pref_bonus"] = w_human

    total_correct = 0
    total_notes = 0

    for filename, notes in phrases:
        # Viterbi DPに渡す形式に変換
        # assign_strings_viterbi_dp は groups 形式を取る
        # 簡略化: 各ノートを個別groupとして渡す
        midi_notes = []
        for n in notes:
            midi_notes.append({
                "pitch": n["pitch"],
                "start": n["start"],
                "duration": n["duration"],
            })

        # Viterbi DPで割り当て
        try:
            assigned = string_assigner.assign_strings_dp(midi_notes)
        except Exception:
            continue

        # 正解と比較
        # assignedの各ノートにstring/fretが付与されている
        for assigned_note in assigned:
            pred_string = assigned_note.get("string", 0)
            pred_fret = assigned_note.get("fret", -1)
            pitch = assigned_note.get("pitch", 0)
            start = assigned_note.get("start", 0)

            # 対応する正解を探す
            best_match = None
            best_dist = float("inf")
            for gt in notes:
                if gt["pitch"] == pitch:
                    dist = abs(gt["start"] - start)
                    if dist < best_dist:
                        best_dist = dist
                        best_match = gt

            if best_match and best_dist < 0.1:  # 100ms以内
                total_notes += 1
                # GT string: IDMT形式(1=6弦) → 標準形式(6=6弦)に変換
                gt_string_std = 7 - best_match["string"]
                if pred_string == gt_string_std:
                    total_correct += 1

    return total_correct, total_notes


def main():
    print("=== Viterbi DP String Assignment Benchmark ===")
    phrases = load_guitarset_phrases(max_files=30)
    total_notes = sum(len(n) for _, n in phrases)
    print(f"Phrases: {len(phrases)}, Total notes: {total_notes}")

    # 元の重みを保存
    orig = string_assigner.WEIGHTS["w_human_pref_bonus"]

    # テスト
    configs = [
        ("No human pref", 0.0),
        ("Human pref -15 (current)", -15.0),
        ("Human pref -30", -30.0),
        ("Human pref -50", -50.0),
        ("Human pref -100", -100.0),
    ]

    print(f"\n{'Config':<30s} {'Correct':>8s} {'Total':>8s} {'Accuracy':>10s}")
    print("-" * 60)

    for name, w_h in configs:
        correct, total = evaluate_viterbi(phrases, w_h)
        acc = correct / total * 100 if total > 0 else 0
        print(f"{name:<30s} {correct:>8d} {total:>8d} {acc:>9.1f}%")

    # 元に戻す
    string_assigner.WEIGHTS["w_human_pref_bonus"] = orig


if __name__ == "__main__":
    main()
