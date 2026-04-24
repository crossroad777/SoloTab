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
from guitar_transcriber import _frames_to_notes
import scipy.signal

def calculate_alignment_offset(gt_notes, pred_notes, resolution=0.01):
    if not gt_notes or not pred_notes: return 0.0
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
    best_lag = lags[np.argmax(correlation)]
    return best_lag * resolution

GUITARSET_DIR = r"D:\Music\Datasets\GuitarSet"
ANNOTATIONS_DIR = os.path.join(GUITARSET_DIR, "annotation")
AUDIO_DIR = os.path.join(GUITARSET_DIR, "audio_mono-pickup_mix")

def load_jams_notes(jams_path):
    with open(jams_path, 'r', encoding='utf-8') as f: data = json.load(f)
    notes = []
    for ann in data.get('annotations', []):
        if ann.get('namespace') == 'note_midi':
            for d in ann.get('data', []):
                s = float(d.get('time', 0.0))
                notes.append({"start": s, "end": s + float(d.get('duration', 0.0)), "pitch": int(round(float(d.get('value', 0.0))))})
    return notes

def format_for_mireval(notes):
    intervals, pitches = [], []
    for n in notes:
        intervals.append([n['start'], n['end']])
        pitches.append(n['pitch'])
    if not intervals: return np.empty((0, 2)), np.empty(0)
    intervals_np = np.array(intervals, dtype=float)
    pitches_hz = np.array([440.0 * (2.0 ** ((p - 69.0) / 12.0)) for p in pitches], dtype=float)
    unique_mask = [True] * len(intervals)
    for i in range(1, len(intervals)):
        if abs(intervals[i][0] - intervals[i-1][0]) < 0.01 and pitches[i] == pitches[i-1]:
            unique_mask[i] = False
    return intervals_np[unique_mask], pitches_hz[unique_mask]

def main():
    print("=== SoloTab V2.0 : Threshold Tuning (Grid Search) ===")
    jams_files = glob.glob(os.path.join(ANNOTATIONS_DIR, "00_*.jams"))
    # 時間短縮のため、検証に使う曲を15曲程度に絞ることも可能ですが、ここでは全曲を対象とします。
    # jams_files = jams_files[:15]
    
    thresholds_to_test = np.arange(0.30, 0.85, 0.05)
    # thresholds_to_test = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    
    # { threshold: { 'total_f': 0, 'count': 0 } }
    results_grid = { round(th, 2): {'total_f': 0.0, 'total_p': 0.0, 'total_r': 0.0, 'count': 0} for th in thresholds_to_test }
    
    for test_jams in tqdm(jams_files):
        base_name = os.path.basename(test_jams).replace(".jams", "")
        wav_path = os.path.join(AUDIO_DIR, f"{base_name}_mix.wav")
        if not os.path.exists(wav_path): continue
        gt_notes = load_jams_notes(test_jams)
        if not gt_notes: continue
        
        try:
            # 1. 重いニューラルネット推論は1曲につき1回だけ実行し、生の確率行列を得る
            result = transcribe_dynamic_ensemble(wav_path, return_raw_logits=True)
            onset_logits = result.get('raw_onset_logits')
            fret_logits = result.get('raw_fret_logits')
            if onset_logits is None: continue
            
            fret_preds_full = np.argmax(fret_logits, axis=-1)
            
            # 2. 複数の閾値でデコードし、それぞれのP/R/Fを計測
            for th in thresholds_to_test:
                th_key = round(th, 2)
                raw_notes = _frames_to_notes(onset_logits, fret_preds_full, onset_threshold=th_key)
                assigned_notes = assign_strings_dp(raw_notes)
                
                if not assigned_notes: 
                    # 音符ゼロの場合は0点加算としてカウント
                    results_grid[th_key]['count'] += 1
                    continue
                
                offset = calculate_alignment_offset(gt_notes, assigned_notes, resolution=0.01)
                for n in assigned_notes:
                    n['start'] = max(0.0, n['start'] + offset)
                    n['end'] = max(n['start'] + 0.01, n['end'] + offset)
                    
                ref_int, ref_p = format_for_mireval(gt_notes)
                est_int, est_p = format_for_mireval(assigned_notes)
                
                p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
                    ref_int, ref_p, est_int, est_p, onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None
                )
                
                results_grid[th_key]['total_p'] += p
                results_grid[th_key]['total_r'] += r
                results_grid[th_key]['total_f'] += f
                results_grid[th_key]['count'] += 1
                
        except Exception as e:
            print(f"[Error] {base_name}: {e}")

    # 結果出力
    print("\n--- Tuning Results ---")
    best_th = 0
    best_f1 = -1
    for th in sorted(results_grid.keys()):
        data = results_grid[th]
        if data['count'] > 0:
            avg_p = data['total_p'] / data['count']
            avg_r = data['total_r'] / data['count']
            avg_f = data['total_f'] / data['count']
            print(f"Threshold = {th:.2f} | F1: {avg_f:.4f} (P: {avg_p:.4f}, R: {avg_r:.4f})")
            if avg_f > best_f1:
                best_f1 = avg_f
                best_th = th
                
    print(f"\n=> Optimal Onset Threshold is {best_th:.2f} with F1 = {best_f1:.4f}")

if __name__ == "__main__":
    main()
