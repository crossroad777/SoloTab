import os
import sys
import glob
import json
import torch
import numpy as np

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# mir_eval for standard MIR metrics
try:
    import mir_eval
except ImportError:
    print("mir_eval is required. Install it with: pip install mir_eval")
    sys.exit(1)

from moe_ensemble_transcriber import transcribe_dynamic_ensemble
from string_assigner import assign_strings_dp

GUITARSET_DIR = r"D:\Music\Datasets\GuitarSet"
ANNOTATIONS_DIR = os.path.join(GUITARSET_DIR, "annotation")
AUDIO_DIR = os.path.join(GUITARSET_DIR, "audio_mono-mic")

def load_jams_notes(jams_path):
    """
    JAMSファイル（JSON）からGT（Ground Truth）のノートを抽出。
    各弦ごとの Note MIDI と Start/End 時間を取得する。
    """
    with open(jams_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print("Loaded JAMS file.")
    notes = []
    # GuitarSet の jams には sandbox -> ... -> annotations の中に弦ごとの note annotation がある
    for ann in data.get('annotations', []):
        if ann.get('namespace') == 'note_midi':
            for data_dict in ann.get('data', []):
                start = float(data_dict.get('time', 0.0))
                dur = float(data_dict.get('duration', 0.0))
                pitch = int(round(float(data_dict.get('value', 0.0))))
                
                # 全弦を統合して純粋なオンセット・オフセット・ピッチを抽出
                notes.append({
                    "start": start,
                    "end": start + dur,
                    "pitch": pitch
                })
    return notes

def format_for_mireval(notes):
    """ mir_eval が求める numpy array に変換 """
    intervals = []
    pitches = []
    for n in notes:
        intervals.append([n['start'], n['end']])
        pitches.append(n['pitch'])
    
    if len(intervals) == 0:
        return np.empty((0, 2)), np.empty(0)
    
    # Hz に変換 (mir_eval は Hzベースのピッチを好む)
    intervals_np = np.array(intervals, dtype=float)
    pitches_hz = np.array([440.0 * (2.0 ** ((p - 69.0) / 12.0)) for p in pitches], dtype=float)
    
    return intervals_np, pitches_hz

def main():
    print("=== SoloTab V2.0 : GuitarSet Benchmark ===")
    
    # 1. 曲の選択 (テストとして "00_" プレフィックスの曲を1曲選ぶ)
    jams_files = glob.glob(os.path.join(ANNOTATIONS_DIR, "00_BN1-129-Eb_comp.jams"))
    if not jams_files:
        print("[Error] No JAMS files found. Checking default directory...")
        jams_files = glob.glob(os.path.join(ANNOTATIONS_DIR, "*.jams"))
        if not jams_files:
             print("GuitarSet directory not found or empty.")
             sys.exit(1)
            
    test_jams = jams_files[0]
    base_name = os.path.basename(test_jams).replace(".jams", "")
    wav_path = os.path.join(AUDIO_DIR, f"{base_name}_mic.wav")
    
    if not os.path.exists(wav_path):
        print(f"[Error] Audio file not found: {wav_path}")
        # fallback to hex pickup mix if mic is missing
        wav_path = os.path.join(GUITARSET_DIR, "audio_mono-pickup_mix", f"{base_name}_mix.wav")
        if not os.path.exists(wav_path):
             sys.exit(1)
             
    print(f"Testing on: {base_name}")
    
    # 2. GT(正解)ロード
    gt_notes = load_jams_notes(test_jams)
    ref_intervals, ref_pitches = format_for_mireval(gt_notes)
    print(f"Ground Truth Notes Count: {len(gt_notes)}")
    
    # 3. SoloTab (MoE) 推論
    print("Running SoloTab MoE Inference...")
    result = transcribe_dynamic_ensemble(wav_path)
    raw_notes = result['notes']
    
    # 4. Finger Optimization
    print("Applying Viterbi Fingering Optimization...")
    assigned_notes = assign_strings_dp(raw_notes)
    
    # 時間軸オフセット補正 (アライメント)
    if len(gt_notes) > 0 and len(assigned_notes) > 0:
        first_gt_start = sorted(gt_notes, key=lambda x: x['start'])[0]['start']
        first_est_start = sorted(assigned_notes, key=lambda x: x['start'])[0]['start']
        time_offset = first_gt_start - first_est_start
        print(f"\nApplying Time Offset Correction: {time_offset:+.3f} sec")
        for n in assigned_notes:
            n['start'] += time_offset
            n['end'] += time_offset

    est_intervals, est_pitches = format_for_mireval(assigned_notes)
    print(f"Predicted Notes Count: {len(assigned_notes)}")
    
    # --- Debugging Output ---
    print("\n--- Raw Data Check (First 5 notes) ---")
    print("GT Notes (Expected):")
    for n in sorted(gt_notes, key=lambda x: x['start'])[:5]:
        print(f"  t={n['start']:.3f}-{n['end']:.3f} MIDI={n['pitch']}")
    print("Pred Notes (Actual):")
    for n in sorted(assigned_notes, key=lambda x: x['start'])[:5]:
        print(f"  t={n['start']:.3f}-{n['end']:.3f} MIDI={n['pitch']}")
    print("--------------------------------------\n")
    
    # 5. mir_eval を用いた Onset / F-measure 評価
    # Onset tolerance 50ms, Pitch tolerance 50 Cents
    # offsetはギターの性質上正確な取得が難しいため、オフセット評価は参考値とし、Onset F-measureをメインとする
    precision, recall, f_measure, avg_overlap_ratio = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches, est_intervals, est_pitches,
        onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None # None deactivates offset check if supported, else leave as default
    )
    
    print("\n" + "="*50)
    print(" === GuitarSet Benchmark Results (Onset Note-Level) ===")
    print("="*50)
    print(f"Precision : {precision:.4f} (鳴っていない音をノイズとして出していないか)")
    print(f"Recall    : {recall:.4f} (鳴っている音を漏らさず拾えているか)")
    print(f"F1-Score  : {f_measure:.4f} (総合評価: 1.0が完全一致)")
    print("="*50)
    print("※これは1曲の抜き打ちテストによる結果です。")

if __name__ == "__main__":
    main()
