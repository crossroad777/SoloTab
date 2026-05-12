"""
optimize_string_assignment.py — GuitarSetを使った弦割り当てパラメータ最適化
========================================================================
GuitarSetの正解(弦, ピッチ)データを使い、assign_strings_dpの
コスト関数パラメータをOptuna で自動最適化する。

Usage:
    python optimize_string_assignment.py
"""

import os
import sys
import json
import glob
import time
import numpy as np

# バックエンドのモジュールを使えるように
sys.path.insert(0, os.path.dirname(__file__))

# solotab_utils/guitar_cost_functions を先にimport
from solotab_utils import STANDARD_TUNING


def load_guitarset_groundtruth(annotation_dir: str, max_songs: int = None):
    """GuitarSetのJAMSアノテーションから正解データを読み込む。
    
    Returns: list of dict
        Each: {
            'name': str,
            'notes': [{'start': float, 'pitch': int, 'string': int, 'duration': float}, ...]
        }
    """
    import jams
    
    jams_files = sorted(glob.glob(os.path.join(annotation_dir, "*.jams")))
    if max_songs:
        jams_files = jams_files[:max_songs]
    
    songs = []
    for jams_path in jams_files:
        j = jams.load(jams_path)
        name = os.path.basename(jams_path).replace(".jams", "")
        
        # note_midi namespaces: 1つ目=6弦, 2つ目=5弦, ..., 6つ目=1弦
        midi_annotations = [ann for ann in j.annotations if ann.namespace == "note_midi"]
        if len(midi_annotations) != 6:
            continue
        
        notes = []
        for string_idx, ann in enumerate(midi_annotations):
            string_num = 6 - string_idx  # 6弦→1弦
            for obs in ann.data:
                pitch = int(round(obs.value))
                notes.append({
                    'start': float(obs.time),
                    'pitch': pitch,
                    'string': string_num,
                    'duration': float(obs.duration),
                })
        
        notes.sort(key=lambda n: n['start'])
        songs.append({'name': name, 'notes': notes})
    
    return songs


def evaluate_assignment(songs, params: dict = None):
    """パラメータで弦割り当てを実行し、正解と比較する。
    
    Returns: dict with metrics
    """
    import guitar_cost_functions as gcf
    from string_assigner import assign_strings_dp
    
    # パラメータを一時的に上書き
    original_weights = dict(gcf.WEIGHTS)
    if params:
        for k, v in params.items():
            if k in gcf.WEIGHTS:
                gcf.WEIGHTS[k] = v
    
    total_notes = 0
    string_correct = 0
    
    try:
        for song in songs:
            gt_notes = song['notes']
            
            # 入力: ピッチのみ（弦情報は除去）
            input_notes = [
                {'start': n['start'], 'pitch': n['pitch'], 
                 'duration': n['duration'], 'velocity': 0.9}
                for n in gt_notes
            ]
            
            # 弦割り当て実行 (CNN/Transformerなし=pure Viterbi DP)
            assigned = assign_strings_dp(input_notes, tuning=STANDARD_TUNING)
            
            # 正解と比較
            for gt, pred in zip(gt_notes, assigned):
                total_notes += 1
                if gt['string'] == pred.get('string'):
                    string_correct += 1
    finally:
        # パラメータを復元
        gcf.WEIGHTS.update(original_weights)
    
    accuracy = string_correct / total_notes if total_notes > 0 else 0
    return {
        'string_accuracy': accuracy,
        'total_notes': total_notes,
        'string_correct': string_correct,
    }


def run_optimization(songs, n_trials: int = 100):
    """Optunaでパラメータ最適化を実行。"""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def objective(trial):
        params = {
            'w_open_string_bonus': trial.suggest_float('w_open_string_bonus', -30.0, 0.0),
            'w_open_match_bonus': trial.suggest_float('w_open_match_bonus', -40.0, 0.0),
            'w_fret_height': trial.suggest_float('w_fret_height', 0.01, 1.0),
            'w_high_fret_extra': trial.suggest_float('w_high_fret_extra', 1.0, 10.0),
            'w_movement': trial.suggest_float('w_movement', 5.0, 30.0),
            'w_position_shift': trial.suggest_float('w_position_shift', 20.0, 80.0),
            'w_string_switch': trial.suggest_float('w_string_switch', 0.5, 10.0),
            'w_same_string_repeat': trial.suggest_float('w_same_string_repeat', 1.0, 15.0),
            'w_sweet_spot_bonus': trial.suggest_float('w_sweet_spot_bonus', -5.0, 0.0),
        }
        result = evaluate_assignment(songs, params)
        return result['string_accuracy']
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return study


if __name__ == "__main__":
    ANNOTATION_DIR = r"D:\Music\Datasets\GuitarSet\annotation"
    
    print("=" * 60)
    print("GuitarSet弦割り当てパラメータ最適化")
    print("=" * 60)
    
    # Step 1: データ読み込み
    print("\n[1/3] GuitarSetデータ読み込み中...")
    songs = load_guitarset_groundtruth(ANNOTATION_DIR, max_songs=20)
    total = sum(len(s['notes']) for s in songs)
    print(f"  → {len(songs)}曲, {total}ノート")
    
    # Step 2: 現在のパラメータでベースライン評価
    print("\n[2/3] 現在のパラメータでベースライン評価...")
    t0 = time.time()
    baseline = evaluate_assignment(songs)
    print(f"  → 弦一致率: {baseline['string_accuracy']:.4f} "
          f"({baseline['string_correct']}/{baseline['total_notes']}) "
          f"[{time.time()-t0:.1f}s]")
    
    # Step 3: Optuna最適化
    print(f"\n[3/3] Optuna最適化 (100 trials)...")
    t0 = time.time()
    study = run_optimization(songs, n_trials=100)
    
    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"最適化完了 ({time.time()-t0:.0f}s)")
    print(f"ベースライン弦一致率: {baseline['string_accuracy']:.4f}")
    print(f"最適パラメータ弦一致率: {best.value:.4f}")
    print(f"改善: +{(best.value - baseline['string_accuracy'])*100:.2f}%")
    print(f"\n最適パラメータ:")
    for k, v in sorted(best.params.items()):
        print(f"  {k}: {v:.4f}")
    
    # 結果をJSON保存
    result = {
        'baseline_accuracy': baseline['string_accuracy'],
        'best_accuracy': best.value,
        'best_params': best.params,
        'n_songs': len(songs),
        'n_notes': baseline['total_notes'],
    }
    out_path = os.path.join(os.path.dirname(__file__), 'optimized_weights.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n結果保存: {out_path}")
