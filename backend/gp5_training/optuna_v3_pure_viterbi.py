"""
Optuna V3: Pure Viterbi DP Weight Optimization on GuitarSet
============================================================
MLモデル(CNN/LSTM/Transformer/GP5分類器)を完全に除外し、
Viterbi DPのコスト関数重みのみをGuitarSet全360ファイルで最適化。

処理時間: Pure Viterbi = ~8秒/360ファイル → 高速に多数trial実行可能
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import sys, os, json, time
import numpy as np

sys.path.insert(0, r'D:\Music\nextchord-solotab\backend')

import warnings
warnings.filterwarnings('ignore')

# --- MLモデルを全て無効化 ---
import string_assigner as sa
sa._GP5_STRING_CLASSIFIER = False
sa._GP5_CONTEXT_LSTM = False
sa._FINGERING_TRANSFORMER = False
sa._load_string_classifier = lambda: False

from string_assigner import assign_strings_dp, get_possible_positions
from solotab_utils import STANDARD_TUNING
import guitar_cost_functions as gcf

GUITARSET_DIR = r"D:\Music\nextchord-solotab\datasets\GuitarSet"

# --- GuitarSet読み込み (1回だけ) ---
def load_guitarset():
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


def evaluate_weights(weights_dict, file_notes):
    """重みを設定してGuitarSet全体の精度を評価"""
    # 重みを直接書き換え
    for k, v in weights_dict.items():
        if k in gcf.WEIGHTS:
            gcf.WEIGHTS[k] = v

    total_correct = 0
    total_notes = 0

    for filename, notes in file_notes.items():
        input_notes = [
            {'pitch': n['pitch'], 'start': n['start'], 'duration': n['duration']}
            for n in notes
        ]
        result = assign_strings_dp(input_notes, tuning=STANDARD_TUNING)
        for i, note in enumerate(notes):
            if i >= len(result):
                break
            if note['gt_string'] == result[i].get('string', 1):
                total_correct += 1
            total_notes += 1

    return total_correct / total_notes if total_notes > 0 else 0.0


def main():
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("=" * 60)
    print("  Optuna V3: Pure Viterbi Weight Optimization")
    print("=" * 60)

    print("\n[1/3] Loading GuitarSet...")
    file_notes = load_guitarset()
    total_notes = sum(len(v) for v in file_notes.values())
    print(f"  {total_notes} notes, {len(file_notes)} files")

    # 初期精度
    print("\n[2/3] Baseline accuracy...")
    baseline = evaluate_weights(dict(gcf._DEFAULT_WEIGHTS), file_notes)
    print(f"  Baseline: {baseline*100:.1f}%")

    # Optuna最適化
    print("\n[3/3] Running Optuna optimization (500 trials)...")

    best_acc = [baseline]
    best_weights = [dict(gcf._DEFAULT_WEIGHTS)]

    def objective(trial):
        w = {
            # 位置コスト
            "w_fret_height":          trial.suggest_float("w_fret_height", 0.1, 5.0),
            "w_high_fret_extra":      trial.suggest_float("w_high_fret_extra", 0.0, 30.0),
            "w_low_string_high_fret": trial.suggest_float("w_low_string_high_fret", 0.5, 5.0),
            "w_sweet_spot_bonus":     trial.suggest_float("w_sweet_spot_bonus", -20.0, 0.0),

            # 遷移コスト
            "w_movement":            trial.suggest_float("w_movement", 1.0, 80.0),
            "w_position_shift":      trial.suggest_float("w_position_shift", 5.0, 150.0),
            "w_string_switch":       trial.suggest_float("w_string_switch", 0.0, 20.0),
            "w_same_string_repeat":  trial.suggest_float("w_same_string_repeat", 0.0, 40.0),

            # 音色コスト
            "w_open_string_bonus":   trial.suggest_float("w_open_string_bonus", -30.0, 0.0),
            "w_open_match_bonus":    trial.suggest_float("w_open_match_bonus", -40.0, 0.0),

            # フィンガースタイル
            "w_bass_low_string":     trial.suggest_float("w_bass_low_string", -50.0, 0.0),
            "w_melody_high_string":  trial.suggest_float("w_melody_high_string", -50.0, 0.0),
            "w_bass_wrong_string":   trial.suggest_float("w_bass_wrong_string", 0.0, 60.0),
            "w_human_pref_bonus":    trial.suggest_float("w_human_pref_bonus", -40.0, 0.0),

            # ピッチ近接性
            "w_pitch_proximity_same_string": trial.suggest_float("w_pitch_prox_same", -25.0, 0.0),
            "w_pitch_proximity_adj_string":  trial.suggest_float("w_pitch_prox_adj", -15.0, 0.0),

            # PIMA
            "w_pima_natural_bonus":  trial.suggest_float("w_pima_natural", -15.0, 0.0),
            "w_pima_thumb_bass":     trial.suggest_float("w_pima_thumb_bass", -15.0, 0.0),
            "w_pima_thumb_wrong":    trial.suggest_float("w_pima_thumb_wrong", 0.0, 30.0),
            "w_pima_crossing":       trial.suggest_float("w_pima_crossing", 0.0, 60.0),
            "w_pima_same_finger":    trial.suggest_float("w_pima_same_finger", 0.0, 40.0),

            # Radicioni
            "w_radicioni_stretch":     trial.suggest_float("w_radicioni_stretch", 0.0, 50.0),
            "w_radicioni_independence": trial.suggest_float("w_radicioni_indep", 0.0, 10.0),
        }
        # 固定パラメータ (探索不要)
        w["w_fret_span"] = 100.0
        w["w_unplayable"] = 10000.0
        w["w_adjacent_stretch"] = 30.0
        w["w_too_many_fingers"] = 5000.0
        w["w_barre_bonus"] = -5.0
        w["w_pima_ama_avoid"] = 8.0

        acc = evaluate_weights(w, file_notes)

        if acc > best_acc[0]:
            best_acc[0] = acc
            best_weights[0] = dict(w)
            print(f"  ★ New best: {acc*100:.2f}% (trial {trial.number})")

        return acc

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
    t0 = time.time()
    study.optimize(objective, n_trials=500, show_progress_bar=False)
    t1 = time.time()

    print(f"\n{'=' * 60}")
    print(f"  OPTIMIZATION COMPLETE ({t1-t0:.0f}s)")
    print(f"{'=' * 60}")
    print(f"\n  Best accuracy: {best_acc[0]*100:.2f}%")
    print(f"  Improvement: {baseline*100:.1f}% → {best_acc[0]*100:.2f}%")

    # 保存
    output = {
        "weights": best_weights[0],
        "string_accuracy": best_acc[0],
        "baseline_accuracy": baseline,
        "n_trials": 500,
        "dataset": "GuitarSet",
        "method": "pure_viterbi_optuna_v3",
    }
    out_path = os.path.join(os.path.dirname(__file__), '..', 'optimized_weights_v3.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to: {out_path}")

    # Best weights
    print(f"\n  Best weights:")
    for k, v in sorted(best_weights[0].items()):
        default = gcf._DEFAULT_WEIGHTS.get(k, '?')
        if isinstance(default, (int, float)):
            print(f"    {k:<35} {v:>8.3f}  (was {default:.3f})")
        else:
            print(f"    {k:<35} {v:>8.3f}")


if __name__ == "__main__":
    main()
