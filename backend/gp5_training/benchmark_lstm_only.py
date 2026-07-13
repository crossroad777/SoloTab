"""
LSTM単体精度 vs パイプライン精度 on GuitarSet
=============================================
LSTMの弦予測(argmax)だけで何%取れるかを測定する。
"""
import sys, json, os, time
import numpy as np

sys.path.insert(0, r'D:\Music\nextchord-solotab\backend')

import warnings
warnings.filterwarnings('ignore')

from string_assigner import (
    _predict_gp5_classifier, _predict_gp5_lstm,
    get_possible_positions, assign_strings_dp,
    _load_gp5_chord_dictionary
)
from solotab_utils import STANDARD_TUNING

GUITARSET_DIR = r"D:\Music\nextchord-solotab\datasets\GuitarSet"


def load_guitarset_by_file():
    import jams
    jams_files = sorted([f for f in os.listdir(GUITARSET_DIR) if f.endswith('.jams')])
    file_notes = {}
    for jf in jams_files:
        path = os.path.join(GUITARSET_DIR, jf)
        try:
            jam = jams.load(path)
        except Exception:
            continue
        string_annotations = [a for a in jam.annotations if a.namespace == 'note_midi']
        if len(string_annotations) != 6:
            continue
        notes = []
        for str_idx, ann in enumerate(string_annotations):
            string_num = 6 - str_idx
            for obs in ann.data:
                midi_pitch = int(round(obs.value))
                start_time = float(obs.time)
                duration = float(obs.duration)
                open_pitch = STANDARD_TUNING[6 - string_num]
                fret = midi_pitch - open_pitch
                if fret < 0 or fret > 22:
                    continue
                notes.append({
                    'pitch': midi_pitch,
                    'start': start_time,
                    'duration': duration,
                    'gt_string': string_num,
                    'gt_fret': fret,
                })
        if notes:
            notes.sort(key=lambda n: n['start'])
            file_notes[jf] = notes
    return file_notes


def eval_lstm_only(notes):
    """LSTM確率のargmaxで弦を決定した場合の精度"""
    correct = 0
    total = 0
    for note in notes:
        gt = note['gt_string']
        lstm_probs = note.get('gp5_lstm_probs')
        if lstm_probs:
            pred = max(lstm_probs, key=lstm_probs.get)
            total += 1
            if pred == gt:
                correct += 1
    return correct, total


def eval_classifier_only(notes):
    """GP5分類器のargmaxで弦を決定した場合の精度"""
    correct = 0
    total = 0
    for note in notes:
        gt = note['gt_string']
        clf_probs = note.get('gp5_string_probs')
        if clf_probs:
            pred = max(clf_probs, key=clf_probs.get)
            total += 1
            if pred == gt:
                correct += 1
    return correct, total


def eval_lowest_fret(notes):
    """最低フレット戦略（ベースライン）: 弾けるポジションの中で最もフレットが低いものを選ぶ"""
    correct = 0
    total = 0
    for note in notes:
        gt = note['gt_string']
        positions = get_possible_positions(note['pitch'], STANDARD_TUNING)
        if positions:
            # 最低フレットを選択
            best = min(positions, key=lambda p: p[1])
            total += 1
            if best[0] == gt:
                correct += 1
    return correct, total


def main():
    print("=" * 60)
    print("  LSTM単体 vs パイプライン vs ベースライン on GuitarSet")
    print("=" * 60)

    print("\n[1/3] Loading GuitarSet...")
    file_notes = load_guitarset_by_file()
    total_notes = sum(len(v) for v in file_notes.values())
    print(f"  {total_notes} notes from {len(file_notes)} files")

    print("\n[2/3] Running predictions...")
    _load_gp5_chord_dictionary()

    lstm_correct = 0
    lstm_total = 0
    clf_correct = 0
    clf_total = 0
    baseline_correct = 0
    baseline_total = 0

    t0 = time.time()
    processed = 0

    for filename, notes in file_notes.items():
        input_notes = [
            {'pitch': n['pitch'], 'start': n['start'], 'duration': n['duration'],
             'gt_string': n['gt_string'], 'gt_fret': n['gt_fret']}
            for n in notes
        ]

        # GP5分類器 + LSTM の確率注入（Viterbi DPは実行しない）
        input_notes = _predict_gp5_classifier(input_notes, STANDARD_TUNING)
        input_notes = _predict_gp5_lstm(input_notes, STANDARD_TUNING)

        # 各手法で評価
        c, t = eval_lstm_only(input_notes)
        lstm_correct += c
        lstm_total += t

        c, t = eval_classifier_only(input_notes)
        clf_correct += c
        clf_total += t

        c, t = eval_lowest_fret(input_notes)
        baseline_correct += c
        baseline_total += t

        processed += 1
        if processed % 60 == 0:
            print(f"    ...{processed}/{len(file_notes)} files")

    t1 = time.time()

    print(f"\n[3/3] Results ({t1-t0:.1f}s)")
    print(f"\n  {'Method':<30} {'Accuracy':>10} {'Correct':>10} {'Total':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")

    if lstm_total > 0:
        print(f"  {'GP5 LSTM v2 (argmax)':<30} {lstm_correct/lstm_total*100:>9.1f}% {lstm_correct:>10} {lstm_total:>10}")
    if clf_total > 0:
        print(f"  {'GP5 Classifier v2 (argmax)':<30} {clf_correct/clf_total*100:>9.1f}% {clf_correct:>10} {clf_total:>10}")
    if baseline_total > 0:
        print(f"  {'Lowest Fret (baseline)':<30} {baseline_correct/baseline_total*100:>9.1f}% {baseline_correct:>10} {baseline_total:>10}")
    print(f"  {'Full Pipeline (前回結果)':<30} {'61.7%':>10} {'38550':>10} {'62476':>10}")
    print(f"  {'論文目標':<30} {'95.2%':>10}")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
