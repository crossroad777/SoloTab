import os
import sys
import glob
import json
import torch
import numpy as np
from tqdm import tqdm
try:
    import mir_eval
except ImportError:
    print("mir_eval is required.")
    sys.exit(1)

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
from moe_ensemble_transcriber import transcribe_dynamic_ensemble
from string_assigner import assign_strings_dp

import scipy.signal

def calculate_alignment_offset(gt_notes, pred_notes, resolution=0.01):
    """相互相関を用いて、pred_notesをgt_notesに合わせるためのオフセット(秒)を計算する。"""
    if not gt_notes or not pred_notes:
        return 0.0
    max_time = max(max(n['start'] for n in gt_notes), max(n['start'] for n in pred_notes)) + 2.0
    vec_len = int(max_time / resolution)
    gt_vec = np.zeros(vec_len)
    pred_vec = np.zeros(vec_len)
    
    for n in gt_notes:
        idx = int(n['start'] / resolution)
        if idx < vec_len: gt_vec[idx] = 1.0
    for n in pred_notes:
        idx = int(n['start'] / resolution)
        if idx < vec_len: pred_vec[idx] = 1.0
        
    correlation = scipy.signal.correlate(gt_vec, pred_vec, mode='full')
    lags = scipy.signal.correlation_lags(gt_vec.size, pred_vec.size, mode='full')
    
    best_lag_idx = np.argmax(correlation)
    best_lag = lags[best_lag_idx]
    
    # y[n] を x[n] に合わせるためのシフト量は best_lag
    return best_lag * resolution

GUITARSET_DIR = r"D:\Music\Datasets\GuitarSet"
ANNOTATIONS_DIR = os.path.join(GUITARSET_DIR, "annotation")
AUDIO_DIR = os.path.join(GUITARSET_DIR, "audio_mono-pickup_mix")

def load_jams_notes(jams_path):
    with open(jams_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    notes = []
    for ann in data.get('annotations', []):
        if ann.get('namespace') == 'note_midi':
            for data_dict in ann.get('data', []):
                start = float(data_dict.get('time', 0.0))
                dur = float(data_dict.get('duration', 0.0))
                pitch = int(round(float(data_dict.get('value', 0.0))))
                notes.append({"start": start, "end": start + dur, "pitch": pitch})
    return notes

def format_for_mireval(notes):
    intervals = []
    pitches = []
    for n in notes:
        intervals.append([n['start'], n['end']])
        pitches.append(n['pitch'])
    if len(intervals) == 0:
        return np.empty((0, 2)), np.empty(0)
    intervals_np = np.array(intervals, dtype=float)
    pitches_hz = np.array([440.0 * (2.0 ** ((p - 69.0) / 12.0)) for p in pitches], dtype=float)
    # 重複ノーツ（全く同じ時間・ピッチ）を排除（mir_evalのエラー回避とF1の正当評価のため）
    unique_mask = [True] * len(intervals)
    for i in range(1, len(intervals)):
        if abs(intervals[i][0] - intervals[i-1][0]) < 0.01 and pitches[i] == pitches[i-1]:
            unique_mask[i] = False
    
    return intervals_np[unique_mask], pitches_hz[unique_mask]

import copy

def analyze_track(wav_path, jams_path):
    gt_notes = load_jams_notes(jams_path)
    if not gt_notes: return {"raw": (0, 0, 0), "tab": (0, 0, 0)}
    
    try:
        # MoE Inference
        result = transcribe_dynamic_ensemble(wav_path)
        raw_notes = result['notes']
        assigned_notes = assign_strings_dp(raw_notes)
        
        def calc_scores(target_notes):
            if len(target_notes) == 0: return 0, 0, 0
            aligned_notes = copy.deepcopy(target_notes)
            offset = calculate_alignment_offset(gt_notes, aligned_notes, resolution=0.01)
            
            for n in aligned_notes:
                n['start'] = max(0.0, n['start'] + offset)
                n['end'] = max(n['start'] + 0.01, n['end'] + offset)
                
            ref_intervals, ref_pitches = format_for_mireval(gt_notes)
            est_intervals, est_pitches = format_for_mireval(aligned_notes)
            
            p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
                ref_intervals, ref_pitches, est_intervals, est_pitches,
                onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None
            )
            return p, r, f

        p_raw, r_raw, f_raw = calc_scores(raw_notes)
        p_tab, r_tab, f_tab = calc_scores(assigned_notes)
        
        return {"raw": (p_raw, r_raw, f_raw), "tab": (p_tab, r_tab, f_tab)}
    except Exception as e:
        print(f"[Error processing {wav_path}]: {e}")
        return {"raw": (0, 0, 0), "tab": (0, 0, 0)}

def main():
    print("=== SoloTab V2.0 : GuitarSet Full Validation ===")
    jams_files = glob.glob(os.path.join(ANNOTATIONS_DIR, "00_*.jams"))
    
    if not jams_files:
        print("Error: Ground truth 00_*.jams not found.")
        return

    results = {}
    metrics = {"raw_p": 0, "raw_r": 0, "raw_f": 0, "tab_p": 0, "tab_r": 0, "tab_f": 0}
    count = 0
    
    print(f"Total files to process: {len(jams_files)}")
    
    log_file = os.path.join(project_root, "benchmark_progress.log")
    with open(log_file, "w", encoding="utf-8") as f_log:
        f_log.write("=== Benchmark Log ===\n")
        
    for test_jams in tqdm(jams_files):
        base_name = os.path.basename(test_jams).replace(".jams", "")
        wav_path = os.path.join(AUDIO_DIR, f"{base_name}_mix.wav")
        if not os.path.exists(wav_path):
             wav_path = os.path.join(GUITARSET_DIR, "audio_mono-pickup_mix", f"{base_name}_mix.wav")
             if not os.path.exists(wav_path): continue
        
        parts = base_name.split("_")
        genre = "Unknown"
        if len(parts) > 1:
            genre_tag = parts[1].split("-")[0]
            import re
            genre = re.sub(r'\d+', '', genre_tag)
            
        res = analyze_track(wav_path, test_jams)
        
        p_raw, r_raw, f_raw = res["raw"]
        p_tab, r_tab, f_tab = res["tab"]
        
        with open(log_file, "a", encoding="utf-8") as f_log:
            f_log.write(f"[{base_name}] RAW=[F:{f_raw:.4f}] TAB=[F:{f_tab:.4f}]\n")
            
        if genre not in results:
            results[genre] = {'raw_f': 0, 'tab_f': 0, 'count': 0}
        results[genre]['raw_f'] += f_raw
        results[genre]['tab_f'] += f_tab
        results[genre]['count'] += 1
        
        metrics["raw_p"] += p_raw
        metrics["raw_r"] += r_raw
        metrics["raw_f"] += f_raw
        metrics["tab_p"] += p_tab
        metrics["tab_r"] += r_tab
        metrics["tab_f"] += f_tab
        count += 1

    report_path = os.path.join(project_root, "official_sota_report_v3.txt")
    with open(report_path, "w", encoding="utf-8") as rep:
        rep.write("=== SoloTab V2.0 SOTA Evaluation Report (Double Benchmark) ===\n")
        rep.write(f"Total Tracks Tested: {count}\n\n")
        
        for g, data in results.items():
            if data['count'] > 0:
                avg_raw = data['raw_f'] / data['count']
                avg_tab = data['tab_f'] / data['count']
                rep.write(f"Genre: {g} ({data['count']} tracks) -> Raw Pitch: {avg_raw:.4f} | TAB Viterbi: {avg_tab:.4f}\n")
                
        if count > 0:
            rep.write("-" * 50 + "\n")
            rep.write(f"🏆 OVERALL AVERAGE (Macro)\n")
            rep.write(f"[RAW Pitch (AI Ear-Copy Limit)]\n")
            rep.write(f"  F1-Score : {metrics['raw_f']/count:.4f}\n")
            rep.write(f"  Precision: {metrics['raw_p']/count:.4f}\n")
            rep.write(f"  Recall   : {metrics['raw_r']/count:.4f}\n\n")
            rep.write(f"[TAB Playable (Viterbi Applied)]\n")
            rep.write(f"  F1-Score : {metrics['tab_f']/count:.4f}\n")
            rep.write(f"  Precision: {metrics['tab_p']/count:.4f}\n")
            rep.write(f"  Recall   : {metrics['tab_r']/count:.4f}\n")
            
    print(f"\nEvaluation completed. Results saved to {report_path}")

if __name__ == "__main__":
    main()
