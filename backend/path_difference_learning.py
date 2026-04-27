"""
path_difference_learning.py — Viterbiコスト関数の重み自動最適化
================================================================
Radisavljevic & Driessen (2004) "Path Difference Learning" に基づく。

GuitarSetの正解タブ譜（JAMS）を使って、
string_assigner.py の WEIGHTS を勾配降下法で自動チューニングする。

原理:
  1. 正解タブ譜 = 「正解パス」のコスト
  2. 現在の重みでViterbi実行 = 「予測パス」のコスト
  3. 正解パスのコスト < 予測パスのコスト になるよう重みを更新
"""

import json
import os
import sys
import glob
import copy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from string_assigner import (
    WEIGHTS, STANDARD_TUNING, MAX_FRET,
    _position_cost, _transition_cost, _timbre_cost,
    assign_strings_dp, get_possible_positions
)


def load_guitarset_annotations(annotation_dir: str, max_files: int = 50):
    """
    GuitarSetのJAMSファイルから正解ノート（弦+フレット情報付き）を読み込む。
    solo tracks のみ使用（ソロギター特化）。
    """
    import jams

    jams_files = sorted(glob.glob(os.path.join(annotation_dir, "*_solo.jams")))
    if not jams_files:
        jams_files = sorted(glob.glob(os.path.join(annotation_dir, "*.jams")))
    
    jams_files = jams_files[:max_files]
    print(f"Loading {len(jams_files)} JAMS files...")
    
    all_tracks = []
    
    for jams_path in jams_files:
        try:
            jam = jams.load(jams_path)
        except Exception as e:
            print(f"  Skip {os.path.basename(jams_path)}: {e}")
            continue
        
        notes = []
        # GuitarSetでは note_midi アノテーションが弦ごとに順番に並んでいる
        # ann[1]=6弦, ann[3]=5弦, ann[5]=4弦, ann[7]=3弦, ann[9]=2弦, ann[11]=1弦
        note_midi_idx = 0  # note_midiアノテーションの出現順
        for ann in jam.annotations:
            if ann.namespace != "note_midi":
                continue
            
            # 出現順で弦番号を決定: 0→6弦, 1→5弦, ..., 5→1弦
            string_num = 6 - note_midi_idx
            note_midi_idx += 1
            
            if string_num < 1 or string_num > 6:
                continue
            
            string_idx = 6 - string_num  # 配列インデックス
            
            for obs in ann.data:
                midi_pitch = int(round(obs.value))
                start = float(obs.time)
                duration = float(obs.duration)
                
                # フレット計算
                gt_fret = midi_pitch - STANDARD_TUNING[string_idx]
                if gt_fret < 0 or gt_fret > MAX_FRET:
                    continue  # 不正なフレットはスキップ
                
                note = {
                    "pitch": midi_pitch,
                    "start": start,
                    "duration": duration,
                    "end": start + duration,
                    "gt_string": string_num,
                    "gt_fret": gt_fret,
                }
                notes.append(note)
        
        if notes:
            # 時間順にソート
            notes.sort(key=lambda n: (n["start"], n["pitch"]))
            all_tracks.append({
                "file": os.path.basename(jams_path),
                "notes": notes,
            })
    
    print(f"Loaded {len(all_tracks)} tracks, total {sum(len(t['notes']) for t in all_tracks)} notes")
    return all_tracks


def compute_path_cost(notes, weights_dict):
    """ノート列の総コスト（遷移+位置+音色）を計算。"""
    if len(notes) < 2:
        return 0.0
    
    total = 0.0
    for i in range(len(notes)):
        s = notes[i].get("string", notes[i].get("gt_string", 1))
        f = notes[i].get("fret", notes[i].get("gt_fret", 0))
        
        total += _position_cost(s, f)
        
        if i > 0:
            prev_s = notes[i-1].get("string", notes[i-1].get("gt_string", 1))
            prev_f = notes[i-1].get("fret", notes[i-1].get("gt_fret", 0))
            total += _transition_cost(s, f, prev_s, prev_f)
    
    return total


def evaluate_accuracy(tracks, tuning=None):
    """
    全トラックでViterbiを実行し、正解との弦一致率を計算。
    """
    if tuning is None:
        tuning = STANDARD_TUNING
    
    correct_string = 0
    correct_fret = 0
    total = 0
    
    for track in tracks:
        notes_input = []
        for n in track["notes"]:
            if "gt_string" in n and "gt_fret" in n:
                notes_input.append({
                    "pitch": n["pitch"],
                    "start": n["start"],
                    "duration": n.get("duration", 0.5),
                })
                total += 1
        
        if not notes_input:
            continue
        
        # Viterbi実行
        assigned = assign_strings_dp(copy.deepcopy(notes_input), tuning=tuning)
        
        # 正解と比較
        gt_notes = [n for n in track["notes"] if "gt_string" in n]
        for pred, gt in zip(assigned, gt_notes):
            if pred.get("string") == gt.get("gt_string"):
                correct_string += 1
            if (pred.get("string") == gt.get("gt_string") and 
                pred.get("fret") == gt.get("gt_fret")):
                correct_fret += 1
    
    if total == 0:
        return 0.0, 0.0
    
    return correct_string / total, correct_fret / total


def path_difference_learning(tracks, n_epochs: int = 30, lr: float = 0.1):
    """
    Path Difference Learning で WEIGHTS を最適化。
    
    勾配降下法ベースのアプローチ:
    各重みを微小に変化させ、正解パスのコストが予測パスのコストより
    小さくなるように更新する。
    """
    print("=" * 60)
    print("Path Difference Learning — WEIGHTS自動最適化")
    print("=" * 60)
    
    # 最適化対象の重み
    tunable_keys = [
        "w_movement", "w_position_shift", "w_string_switch",
        "w_same_string_repeat", "w_fret_height", "w_high_fret_extra",
        "w_sweet_spot_bonus", "w_open_string_bonus",
    ]
    
    # 初期評価
    str_acc, fret_acc = evaluate_accuracy(tracks)
    print(f"\n初期精度: 弦一致={str_acc:.4f}, 弦+フレット一致={fret_acc:.4f}")
    print(f"初期重み: { {k: WEIGHTS[k] for k in tunable_keys} }")
    
    best_acc = fret_acc
    best_weights = {k: WEIGHTS[k] for k in tunable_keys}
    
    # 物理的制約: 各重みの上下限（意味的に妥当な範囲）
    weight_bounds = {
        "w_movement":          (1.0, 30.0),    # 常に正
        "w_position_shift":    (5.0, 100.0),   # 常に正・大
        "w_string_switch":     (0.5, 10.0),    # 常に正
        "w_same_string_repeat":(0.0, 15.0),    # 0以上
        "w_fret_height":       (0.0, 5.0),     # 0以上（ハイフレット = 高コスト）
        "w_high_fret_extra":   (0.0, 20.0),    # 0以上
        "w_sweet_spot_bonus":  (-5.0, 0.0),    # 常に負（ボーナス）
        "w_open_string_bonus": (-10.0, 0.0),   # 常に負（ボーナス）
    }
    
    for epoch in range(n_epochs):
        improved = False
        
        for key in tunable_keys:
            original_val = WEIGHTS[key]
            lo, hi = weight_bounds.get(key, (-100, 100))
            
            # 正の方向に微小変化（上限チェック）
            val_plus = min(hi, original_val + lr)
            WEIGHTS[key] = val_plus
            _, fret_acc_plus = evaluate_accuracy(tracks)
            
            # 負の方向に微小変化（下限チェック）
            val_minus = max(lo, original_val - lr)
            WEIGHTS[key] = val_minus
            _, fret_acc_minus = evaluate_accuracy(tracks)
            
            # 最も良い方向を選択
            if fret_acc_plus > best_acc and val_plus != original_val:
                WEIGHTS[key] = val_plus
                best_acc = fret_acc_plus
                best_weights[key] = WEIGHTS[key]
                improved = True
                print(f"  [{key}] {original_val:.2f} → {WEIGHTS[key]:.2f} (acc={best_acc:.4f})")
            elif fret_acc_minus > best_acc and val_minus != original_val:
                WEIGHTS[key] = val_minus
                best_acc = fret_acc_minus
                best_weights[key] = WEIGHTS[key]
                improved = True
                print(f"  [{key}] {original_val:.2f} → {WEIGHTS[key]:.2f} (acc={best_acc:.4f})")
            else:
                WEIGHTS[key] = original_val
        
        if not improved:
            lr *= 0.5
            print(f"Epoch {epoch+1}: 改善なし → 学習率を {lr:.4f} に縮小")
            if lr < 0.01:
                print("収束しました。")
                break
        else:
            print(f"Epoch {epoch+1}: best fret_acc={best_acc:.4f}")
    
    # 最終結果
    print("\n" + "=" * 60)
    print("最適化完了")
    print("=" * 60)
    for k, v in best_weights.items():
        WEIGHTS[k] = v
    
    str_acc, fret_acc = evaluate_accuracy(tracks)
    print(f"最終精度: 弦一致={str_acc:.4f}, 弦+フレット一致={fret_acc:.4f}")
    print(f"\n最適化された重み:")
    for k in tunable_keys:
        print(f"  {k}: {WEIGHTS[k]:.4f}")
    
    # 結果をJSONで保存
    output = {
        "string_accuracy": str_acc,
        "fret_accuracy": fret_acc,
        "weights": {k: WEIGHTS[k] for k in tunable_keys},
    }
    output_path = os.path.join(os.path.dirname(__file__), "optimized_weights.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n保存先: {output_path}")
    
    return output


if __name__ == "__main__":
    annotation_dir = r"D:\Music\Datasets\GuitarSet\annotation"
    
    if not os.path.exists(annotation_dir):
        print(f"GuitarSet not found: {annotation_dir}")
        sys.exit(1)
    
    tracks = load_guitarset_annotations(annotation_dir, max_files=20)
    
    if not tracks:
        print("No tracks loaded.")
        sys.exit(1)
    
    result = path_difference_learning(tracks, n_epochs=30, lr=0.5)
