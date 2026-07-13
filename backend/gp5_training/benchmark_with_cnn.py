"""
CNN弦分類器付きベンチマーク on GuitarSet
==========================================
論文§8.7の通り、CNN弦分類器（CQT→6class CNN）を使って
音声特徴量ベースの弦割り当てを評価する。

論文の結果: CNN-first = 96.60% (solo 90トラック)
"""
import sys, os, json, time
import numpy as np

sys.path.insert(0, r'D:\Music\nextchord-solotab\backend')

import warnings
warnings.filterwarnings('ignore')

from string_assigner import assign_strings_dp, get_possible_positions
from solotab_utils import STANDARD_TUNING

GUITARSET_DIR = r"D:\Music\nextchord-solotab\datasets\GuitarSet"
AUDIO_DIR = os.path.join(GUITARSET_DIR, "audio_mono-mic")


def load_guitarset_with_audio():
    import jams
    jams_files = sorted([f for f in os.listdir(GUITARSET_DIR) if f.endswith('.jams')])
    file_data = {}
    for jf in jams_files:
        path = os.path.join(GUITARSET_DIR, jf)
        try:
            jam = jams.load(path)
        except Exception:
            continue
        string_annotations = [a for a in jam.annotations if a.namespace == 'note_midi']
        if len(string_annotations) != 6:
            continue

        # JAMSファイル名からオーディオパスを構築
        # JAMS: "00_BN1-129-Eb_comp.jams" → audio: "00_BN1-129-Eb_comp_mic.wav"
        base = jf.replace('.jams', '')
        audio_path = os.path.join(AUDIO_DIR, f"{base}_mic.wav")
        if not os.path.exists(audio_path):
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
            file_data[jf] = {'notes': notes, 'audio': audio_path}
    return file_data


def main():
    print("=" * 60)
    print("  CNN弦分類器付きベンチマーク on GuitarSet")
    print("=" * 60)

    print("\n[1/2] Loading GuitarSet + audio paths...")
    file_data = load_guitarset_with_audio()
    total_notes = sum(len(v['notes']) for v in file_data.values())
    print(f"  {total_notes} notes, {len(file_data)} files with audio")

    print(f"\n[2/2] Running assign_strings_dp WITH audio (CNN弦分類器)...")
    t0 = time.time()

    total_correct = 0
    total_within_1 = 0
    total_notes_eval = 0
    per_string_correct = [0] * 6
    per_string_total = [0] * 6

    processed = 0
    for filename, data in file_data.items():
        notes = data['notes']
        audio_path = data['audio']

        input_notes = [
            {'pitch': n['pitch'], 'start': n['start'], 'duration': n['duration']}
            for n in notes
        ]

        # CNN弦分類器を有効にするためaudio_pathを渡す
        result = assign_strings_dp(input_notes, tuning=STANDARD_TUNING, audio_path=audio_path)

        for i, note in enumerate(notes):
            if i >= len(result):
                break
            gt = note['gt_string']
            pred = result[i].get('string', 1)
            total_notes_eval += 1
            if gt == pred:
                total_correct += 1
            if abs(gt - pred) <= 1:
                total_within_1 += 1
            if 1 <= gt <= 6:
                per_string_total[gt - 1] += 1
                if gt == pred:
                    per_string_correct[gt - 1] += 1

        processed += 1
        if processed % 30 == 0:
            acc = total_correct / total_notes_eval * 100 if total_notes_eval > 0 else 0
            print(f"    ...{processed}/{len(file_data)} files, accuracy={acc:.1f}%")

    t1 = time.time()
    acc = total_correct / total_notes_eval * 100 if total_notes_eval > 0 else 0
    within_1 = total_within_1 / total_notes_eval * 100 if total_notes_eval > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"  RESULTS ({t1-t0:.1f}s)")
    print(f"{'=' * 60}")

    print(f"\n  {'手法':<40} {'精度':>10}")
    print(f"  {'-'*40} {'-'*10}")
    print(f"  {'CNN弦分類器 + Viterbi DP (今回)':<40} {acc:>9.1f}%")
    print(f"  {'Viterbi DP のみ (前回結果)':<40} {'61.7%':>10}")
    print(f"  {'論文 CNN-first (§8.7)':<40} {'96.6%':>10}")
    print(f"  {'論文 CNN LOO汎化 (§8.9)':<40} {'80.9%':>10}")

    print(f"\n  ±1弦精度: {within_1:.1f}%")
    print(f"  正解: {total_correct}/{total_notes_eval}")

    print(f"\n  弦別精度:")
    for s in range(6):
        if per_string_total[s] > 0:
            s_acc = per_string_correct[s] / per_string_total[s] * 100
            print(f"    String {s+1}: {s_acc:.1f}% ({per_string_correct[s]}/{per_string_total[s]})")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
