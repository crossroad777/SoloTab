"""
Pure Viterbi DP (MLモデルなし) vs 現行パイプライン on GuitarSet
================================================================
CNN/LSTM/Transformer/GP5分類器を全て無効化し、
純粋なコスト関数(位置/遷移/音色/人間工学)のみで何%出るか測定。
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import sys, os, time
import numpy as np

sys.path.insert(0, r'D:\Music\nextchord-solotab\backend')

import warnings
warnings.filterwarnings('ignore')

# --- MONKEY PATCH: MLモデルを全て無効化 ---
import string_assigner as sa

# 各モデルのロードを常にFalseを返すように上書き
sa._GP5_STRING_CLASSIFIER = False
sa._GP5_CONTEXT_LSTM = False
sa._FINGERING_TRANSFORMER = False

# CNN弦分類器も無効化
original_load_clf = sa._load_string_classifier
sa._load_string_classifier = lambda: False

from string_assigner import assign_strings_dp, get_possible_positions
from solotab_utils import STANDARD_TUNING

GUITARSET_DIR = r"D:\Music\nextchord-solotab\datasets\GuitarSet"
NUM_STRINGS = 6


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


def main():
    print("=" * 60)
    print("  Pure Viterbi DP (MLモデル全無効) on GuitarSet")
    print("=" * 60)

    print("\n[1/2] Loading GuitarSet...")
    file_notes = load_guitarset_by_file()
    total_notes = sum(len(v) for v in file_notes.values())
    print(f"  {total_notes} notes from {len(file_notes)} files")

    print(f"\n[2/2] Running pure Viterbi DP...")
    t0 = time.time()

    total_correct = 0
    total_within_1 = 0
    total_notes = 0
    per_string_correct = [0] * NUM_STRINGS
    per_string_total = [0] * NUM_STRINGS

    processed = 0
    for filename, notes in file_notes.items():
        input_notes = [
            {'pitch': n['pitch'], 'start': n['start'], 'duration': n['duration']}
            for n in notes
        ]

        # Pure Viterbi DP (no audio_path = no CNN, MLモデルは上でmonkey-patched)
        result = assign_strings_dp(input_notes, tuning=STANDARD_TUNING)

        for i, note in enumerate(notes):
            if i >= len(result):
                break
            gt = note['gt_string']
            pred = result[i].get('string', 1)
            total_notes += 1
            if gt == pred:
                total_correct += 1
            if abs(gt - pred) <= 1:
                total_within_1 += 1
            if 1 <= gt <= 6 and 1 <= pred <= 6:
                per_string_total[gt - 1] += 1
                if gt == pred:
                    per_string_correct[gt - 1] += 1

        processed += 1
        if processed % 60 == 0:
            acc_so_far = total_correct / total_notes * 100 if total_notes > 0 else 0
            print(f"    ...{processed}/{len(file_notes)} files, accuracy={acc_so_far:.1f}%")

    t1 = time.time()

    acc = total_correct / total_notes * 100 if total_notes > 0 else 0
    within_1_pct = total_within_1 / total_notes * 100 if total_notes > 0 else 0

    print(f"\n  Results ({t1-t0:.1f}s):")
    print(f"\n  {'Method':<35} {'Accuracy':>10}")
    print(f"  {'-'*35} {'-'*10}")
    print(f"  {'Pure Viterbi DP (MLなし)':<35} {acc:>9.1f}%")
    print(f"  {'Full Pipeline (5モデル合議)':<35} {'61.7%':>10}")
    print(f"  {'GP5 LSTM v2 単体':<35} {'41.8%':>10}")
    print(f"  {'最低フレット ベースライン':<35} {'45.4%':>10}")
    print(f"  {'論文目標':<35} {'95.2%':>10}")

    print(f"\n  ±1 string accuracy: {within_1_pct:.1f}%")
    print(f"  Correct: {total_correct}/{total_notes}")

    print(f"\n  Per-string accuracy:")
    for s in range(NUM_STRINGS):
        if per_string_total[s] > 0:
            s_acc = per_string_correct[s] / per_string_total[s] * 100
            print(f"    String {s+1}: {s_acc:.1f}% ({per_string_correct[s]}/{per_string_total[s]})")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
