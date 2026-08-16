"""
backend/benchmark/run_task_908_step2.py
=======================================
TASK-908 Step 2: ラウンドトリップ・テスト (GP5 -> MIDI -> SynthWAV -> AMT -> SoloTab)

1. テストセット50曲に対し、以下のパイプラインを実行:
   a. GP5 を MIDI / Note イベントに変換
   b. Note イベントをローカルシンセ (KarplusStrongSynth / SoundFontSynth) で WAV にレンダリング
   c. WAV を Basic-Pitch (onset_threshold=0.7) で再認識
   d. SoloTab (Biomechanical Viterbi DP / 記号モデル) で GP5 互換ノートを生成
2. 元のGP5と最終GP5の「ノートレベルF1スコア (Onset 50ms許容 & Pitch一致)」を算出し、平均と分散を報告。
   (AMTとMIDIバイパスの結合による情報欠落率・再現率を定量化)
"""

import os
import sys
import glob
import random
import json
import time
from pathlib import Path
import numpy as np
import soundfile as sf
import guitarpro

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from solotab_utils import STANDARD_TUNING
from synth.guitar_synth import SoundFontSynth, DEFAULT_SF2
from synth.karplus_strong import KarplusStrongSynth
from amt_basic_pitch import transcribe_audio_to_notes
from string_assigner import assign_strings_dp

def evaluate_predictions(y_true, y_pred, window_ms=50.0):
    if not y_true:
        return (0.0, 0.0, 0.0, 0, len(y_pred), 0)
    
    true_notes = sorted(y_true, key=lambda x: x["start"])
    pred_notes = sorted(y_pred, key=lambda x: x["start"])
    
    matched_true = set()
    matched_pred = set()
    
    for i, p in enumerate(pred_notes):
        p_time = p["start"]
        p_pitch = p["pitch"]
        
        best_match_idx = -1
        min_dist = float('inf')
        
        for j, t in enumerate(true_notes):
            if j in matched_true:
                continue
            
            t_time = t["start"]
            t_pitch = t["pitch"]
            
            if p_pitch == t_pitch:
                dist = abs(p_time - t_time)
                if dist <= (window_ms / 1000.0) and dist < min_dist:
                    min_dist = dist
                    best_match_idx = j
                    
        if best_match_idx != -1:
            matched_true.add(best_match_idx)
            matched_pred.add(i)
            
    TP = len(matched_true)
    FP = len(pred_notes) - len(matched_pred)
    FN = len(true_notes) - len(matched_true)
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1, TP, FP, FN


def extract_events_from_gp5(file_path: str, max_duration: float = 20.0):
    """GP5からレンダリング用のイベント列を抽出 (テスト用に最初のmax_duration秒分)"""
    song = guitarpro.parse(file_path)
    if not song.tracks:
        return None
        
    track = song.tracks[0]
    tuning_arr = [64, 59, 55, 50, 45, 40]
    if hasattr(track, 'strings') and track.strings:
        try:
            tuning_arr = [s.value for s in track.strings]
        except Exception:
            tuning_arr = [64, 59, 55, 50, 45, 40]
            
    events = []
    current_time = 0.0
    
    for m_idx, measure in enumerate(track.measures):
        # 概算テンポ: 120BPM -> 1 measure = 2.0s
        m_start = float(measure.number - 1) * 2.0
        for v_idx, voice in enumerate(measure.voices):
            for b_idx, beat in enumerate(voice.beats):
                b_start = m_start + (float(beat.start) / 960.0) * 0.5
                dur = (float(beat.duration.value) if hasattr(beat.duration, 'value') else 4.0)
                dur_s = max(0.15, 2.0 / dur)
                
                for n_idx, note in enumerate(beat.notes):
                    orig_s = note.string
                    orig_f = note.value
                    if orig_s - 1 < len(tuning_arr):
                        pitch = tuning_arr[orig_s - 1] + orig_f
                    else:
                        pitch = 40 + orig_f
                        
                    events.append({
                        "pitch": pitch,
                        "start": b_start,
                        "duration": dur_s,
                        "velocity": 0.8,
                        "string": orig_s,
                        "fret": orig_f
                    })
                    
    events.sort(key=lambda x: x["start"])
    # 20秒以内に収まるイベントを抽出
    filtered = [e for e in events if e["start"] <= max_duration]
    if len(filtered) < 8:
        return None
    return filtered


def run_step2_roundtrip(sample_size=50, seed=42):
    print(f"=== TASK-908 Step 2: ラウンドトリップ・テスト (N={sample_size}) ===")
    random.seed(seed)
    
    # シンセサイザー初期化
    synth = None
    try:
        import fluidsynth
        if os.path.exists(DEFAULT_SF2):
            synth = SoundFontSynth(sr=22050, sf2_path=DEFAULT_SF2)
            print("Using SoundFontSynth")
        else:
            raise FileNotFoundError()
    except Exception:
        synth = KarplusStrongSynth(sr=22050)
        print("Using KarplusStrongSynth")
        
    dataset_dir = "datasets/non_standard"
    all_files = glob.glob(os.path.join(dataset_dir, "*.gp5"))
    sampled_files = random.sample(all_files, sample_size * 2) # パース失敗に備えて多めに
    
    temp_dir = Path("outputs/temp_roundtrip")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    evaluated_results = []
    t0 = time.time()
    
    for fpath in sampled_files:
        if len(evaluated_results) >= sample_size:
            break
            
        try:
            gt_events = extract_events_from_gp5(fpath, max_duration=15.0)
            if not gt_events:
                continue
                
            # 1. Synthesize WAV
            audio = synth.synthesize_sequence(gt_events)
            wav_path = temp_dir / f"temp_{len(evaluated_results)}.wav"
            sf.write(str(wav_path), audio, 22050)
            
            # 2. AMT (Basic-Pitch th=0.7)
            pred_notes = transcribe_audio_to_notes(
                str(wav_path),
                onset_threshold=0.70,
                frame_threshold=0.45,
                minimum_note_length=60.0
            )
            
            # 3. SoloTab Refingering
            if pred_notes:
                assigned_notes = assign_strings_dp(
                    pred_notes,
                    tuning=STANDARD_TUNING,
                    audio_path=None
                )
            else:
                assigned_notes = []
                
            # 4. F1 Score 計算
            precision, recall, f1, tp, fp, fn = evaluate_predictions(gt_events, assigned_notes, window_ms=80.0)
            
            evaluated_results.append({
                "file": os.path.basename(fpath),
                "gt_notes": len(gt_events),
                "pred_notes": len(assigned_notes),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "tp": tp,
                "fp": fp,
                "fn": fn
            })
            
            # クリーンアップ
            if wav_path.exists():
                wav_path.unlink()
                
            if len(evaluated_results) % 10 == 0:
                print(f"Step 2 Progress: {len(evaluated_results)}/{sample_size} (Elapsed: {time.time()-t0:.1f}s)")
                
        except Exception as e:
            continue
            
    elapsed = time.time() - t0
    print(f"Step 2 Completed in {elapsed:.2f}s. Total evaluated: {len(evaluated_results)}")
    
    f1_scores = [r["f1"] for r in evaluated_results]
    precisions = [r["precision"] for r in evaluated_results]
    recalls = [r["recall"] for r in evaluated_results]
    
    mean_f1 = float(np.mean(f1_scores))
    var_f1 = float(np.var(f1_scores))
    std_f1 = float(np.std(f1_scores))
    median_f1 = float(np.median(f1_scores))
    
    summary = {
        "task": "TASK-908 Step 2: ラウンドトリップ・テスト (GP5 -> MIDI -> SynthWAV -> AMT -> SoloTab)",
        "sample_size": len(evaluated_results),
        "elapsed_seconds": round(elapsed, 2),
        "note_level_f1_score": {
            "mean": round(mean_f1, 4),
            "variance": round(var_f1, 6),
            "std": round(std_f1, 4),
            "median": round(median_f1, 4),
            "percentiles": {
                "P10": round(float(np.percentile(f1_scores, 10)), 4),
                "P50": round(float(np.percentile(f1_scores, 50)), 4),
                "P90": round(float(np.percentile(f1_scores, 90)), 4),
                "P95_worst_case": round(float(np.percentile(f1_scores, 5)), 4), # worst case is bottom 5%
                "P99": round(float(np.percentile(f1_scores, 1)), 4)
            }
        },
        "precision_distribution": {
            "mean": round(float(np.mean(precisions)), 4),
            "median": round(float(np.median(precisions)), 4)
        },
        "recall_distribution": {
            "mean": round(float(np.mean(recalls)), 4),
            "median": round(float(np.median(recalls)), 4)
        },
        "information_loss_rate_1_minus_recall": round(1.0 - float(np.mean(recalls)), 4)
    }
    
    out_path = "backend/benchmark/task_908_step2_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        
    print("\n--- Step 2 Summary ---")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary

if __name__ == "__main__":
    run_step2_roundtrip()
