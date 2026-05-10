"""
benchmark_string_assignment.py — 人間選好マップのベンチマーク
================================================================
GuitarSetの正解弦データ vs Viterbi DPの弦割り当てを比較。
人間選好マップ有/無で精度がどう変わるかを測定。

Usage:
    python backend/train/benchmark_string_assignment.py
"""
import json, glob, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collections import Counter, defaultdict
import string_assigner

GUITARSET_DIR = r"D:\Music\datasets\GuitarSet\annotation"
from solotab_utils import STANDARD_TUNING
DS_TO_STRING = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}  # IDMT形式


def load_guitarset_ground_truth(max_files=None):
    """GuitarSetから正解データをロード"""
    jams_files = sorted(glob.glob(os.path.join(GUITARSET_DIR, "*.jams")))
    if max_files:
        jams_files = jams_files[:max_files]

    all_notes = []
    for jf in jams_files:
        with open(jf, "r") as f:
            data = json.load(f)

        file_notes = []
        for ann in data.get("annotations", []):
            ns = ann.get("namespace", "")
            if ns != "note_midi":
                continue
            ann_meta = ann.get("annotation_metadata", {})
            ds = ann_meta.get("data_source", "")
            try:
                ds_idx = int(ds)
            except (ValueError, TypeError):
                continue
            if ds_idx not in DS_TO_STRING:
                continue

            string_num = DS_TO_STRING[ds_idx]
            open_pitch = STANDARD_TUNING[ds_idx]

            for obs in ann.get("data", []):
                midi_pitch = obs.get("value", 0)
                if not midi_pitch or midi_pitch <= 0:
                    continue
                pitch = round(midi_pitch)
                fret = pitch - open_pitch
                if fret < 0 or fret > 22:
                    continue
                file_notes.append({
                    "pitch": pitch,
                    "string": string_num,
                    "fret": fret,
                    "onset": obs.get("time", 0),
                    "file": os.path.basename(jf),
                })

        all_notes.append((os.path.basename(jf), file_notes))

    return all_notes


def run_viterbi_assignment(notes_list):
    """Viterbi DPで弦割り当てを実行"""
    results = []
    for pitch_info in notes_list:
        pitch = pitch_info["pitch"]
        # 候補ポジション取得
        candidates = string_assigner.get_possible_positions(pitch)
        if candidates:
            results.append({
                "pitch": pitch,
                "candidates": candidates,
                "gt_string": pitch_info["string"],
                "gt_fret": pitch_info["fret"],
            })
    return results


def evaluate_single_note_accuracy(ground_truth_files):
    """単音レベルでの弦割り当て精度を評価"""
    correct = 0
    total = 0
    correct_by_string = Counter()
    total_by_string = Counter()
    errors = []

    for filename, notes in ground_truth_files:
        for note in notes:
            pitch = note["pitch"]
            gt_string = note["string"]
            gt_fret = note["fret"]

            # Viterbiの_position_costで最もコストが低いポジションを選択
            candidates = string_assigner.get_possible_positions(pitch)
            if not candidates:
                continue

            # 各候補のposition_costを計算（人間選好込み）
            best_pos = None
            best_cost = float("inf")
            for s, f in candidates:
                cost = string_assigner._position_cost(s, f, pitch=pitch)
                cost += string_assigner._timbre_cost(s, f, tuple(STANDARD_TUNING))
                if cost < best_cost:
                    best_cost = cost
                    best_pos = (s, f)

            if best_pos:
                pred_string, pred_fret = best_pos
                total += 1
                total_by_string[gt_string] += 1

                if pred_string == gt_string:
                    correct += 1
                    correct_by_string[gt_string] += 1
                else:
                    errors.append({
                        "pitch": pitch,
                        "gt": f"S{gt_string}F{gt_fret}",
                        "pred": f"S{pred_string}F{pred_fret}",
                    })

    return correct, total, correct_by_string, total_by_string, errors


def evaluate_without_human_pref(ground_truth_files):
    """人間選好なしでの精度を評価（比較用）"""
    # 一時的に人間選好を無効化
    original_weight = string_assigner.WEIGHTS["w_human_pref_bonus"]
    string_assigner.WEIGHTS["w_human_pref_bonus"] = 0.0

    correct, total, correct_by_str, total_by_str, errors = \
        evaluate_single_note_accuracy(ground_truth_files)

    # 元に戻す
    string_assigner.WEIGHTS["w_human_pref_bonus"] = original_weight

    return correct, total, correct_by_str, total_by_str, errors


def main():
    print("=== Human Fingering Preference Benchmark ===")
    print(f"Loading GuitarSet ground truth...")
    gt_files = load_guitarset_ground_truth(max_files=50)

    total_notes = sum(len(notes) for _, notes in gt_files)
    print(f"Files: {len(gt_files)}, Total notes: {total_notes}")

    # --- 人間選好なし ---
    print(f"\n--- WITHOUT human preference map ---")
    c_no, t_no, cbs_no, tbs_no, err_no = evaluate_without_human_pref(gt_files)
    acc_no = c_no / t_no * 100 if t_no > 0 else 0
    print(f"  Accuracy: {c_no}/{t_no} = {acc_no:.1f}%")
    print(f"  Per string:")
    for s in range(1, 7):
        c = cbs_no.get(s, 0)
        t = tbs_no.get(s, 0)
        a = c / t * 100 if t > 0 else 0
        print(f"    S{s}: {c}/{t} = {a:.1f}%")

    # --- 人間選好あり ---
    print(f"\n--- WITH human preference map (116K notes) ---")
    c_yes, t_yes, cbs_yes, tbs_yes, err_yes = evaluate_single_note_accuracy(gt_files)
    acc_yes = c_yes / t_yes * 100 if t_yes > 0 else 0
    print(f"  Accuracy: {c_yes}/{t_yes} = {acc_yes:.1f}%")
    print(f"  Per string:")
    for s in range(1, 7):
        c = cbs_yes.get(s, 0)
        t = tbs_yes.get(s, 0)
        a = c / t * 100 if t > 0 else 0
        print(f"    S{s}: {c}/{t} = {a:.1f}%")

    # --- 比較 ---
    delta = acc_yes - acc_no
    print(f"\n=== RESULT ===")
    print(f"  Without: {acc_no:.1f}%")
    print(f"  With:    {acc_yes:.1f}%")
    print(f"  Delta:   {delta:+.1f}%")

    if delta > 0:
        print(f"  -> Human preference map IMPROVED accuracy!")
    elif delta < 0:
        print(f"  -> Human preference map DECREASED accuracy (weight tuning needed)")
    else:
        print(f"  -> No change")

    # エラー分析
    print(f"\n=== Error Analysis (WITH human pref) ===")
    print(f"  Total errors: {len(err_yes)}")
    if err_yes:
        # 最も多いエラーパターン
        error_patterns = Counter(f"{e['gt']}->{e['pred']}" for e in err_yes)
        print(f"  Top error patterns:")
        for pattern, count in error_patterns.most_common(10):
            print(f"    {pattern}: {count}")


if __name__ == "__main__":
    main()
