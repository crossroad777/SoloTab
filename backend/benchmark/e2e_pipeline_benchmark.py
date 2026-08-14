import os
import sys
import glob
import json
import time
import shutil
import tempfile
from pathlib import Path
import numpy as np

# backend ディレクトリへのパスを追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from pipeline import run_pipeline
except ImportError as e:
    print(f"Error importing pipeline: {e}")
    sys.exit(1)

try:
    import mir_eval
except ImportError:
    print("mir_eval required: pip install mir_eval")
    sys.exit(1)

GUITARSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mini_dataset")
ANNOTATIONS_DIR = os.path.join(GUITARSET_DIR, "annotation")
AUDIO_DIR = os.path.join(GUITARSET_DIR, "audio_mono-mic")

TARGET_TRACKS = [
    "05_Jazz2-187-F#_comp",
    "00_SS1-68-E_comp",
    "03_BN1-147-Gb_comp",
    "01_Funk1-97-C_comp",
    "05_Jazz2-187-F#_solo",
    "00_Rock2-142-D_solo",
    "04_Rock1-130-A_solo",
    "05_SS2-88-F_solo",
    "02_Funk2-119-G_comp"
]

def load_jams_notes_with_string(jams_path):
    """JAMSから正解ノートと弦番号（1〜6）を抽出する"""
    with open(jams_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    notes = []
    for ann in data.get('annotations', []):
        if ann.get('namespace') == 'note_midi':
            # GuitarSetの data_source: 0=6弦(Low E) ... 5=1弦(High E)
            data_source = int(ann.get('annotation_metadata', {}).get('data_source', 0))
            # SoloTabの弦番号: 1=1弦(High E) ... 6=6弦(Low E)
            # つまり、1弦なら data_source=5 -> string=1。 6弦なら data_source=0 -> string=6。
            # 変換式: string = 6 - data_source
            string_num = 6 - data_source
            
            for d in ann.get('data', []):
                start = float(d.get('time', 0.0))
                dur = float(d.get('duration', 0.0))
                pitch = int(round(float(d.get('value', 0.0))))
                notes.append({
                    "start": start, 
                    "end": start + dur, 
                    "pitch": pitch,
                    "string": string_num
                })
    notes.sort(key=lambda x: x['start'])
    return notes

def to_mireval(notes):
    """mir_eval形式の intervals, pitches_hz を作成"""
    if not notes:
        return np.empty((0, 2)), np.empty(0)
    intervals = []
    for n in notes:
        start = n.get('start', n.get('start_time', 0.0))
        end = n.get('end', n.get('end_time', start + n.get('duration', 0.1)))
        intervals.append([start, end])
    intervals = np.array(intervals, dtype=float)
    pitches_hz = np.array([440.0 * (2.0 ** ((n['pitch'] - 69.0) / 12.0)) for n in notes], dtype=float)
    return intervals, pitches_hz

def evaluate_string_fret(ref_notes, est_notes, ref_intervals, ref_pitches, est_intervals, est_pitches):
    """mir_evalのマッチングを利用して、String+FretのF1を算出する"""
    if not ref_notes or not est_notes:
        return 0.0, 0.0, 0.0
        
    # Pitchがマッチしているペアを取得 (onset_tolerance=0.05s, pitch_tolerance=50cents)
    matching = mir_eval.transcription.match_notes(
        ref_intervals, ref_pitches, est_intervals, est_pitches,
        onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None
    )
    
    correct_string_fret = 0
    for ref_idx, est_idx in matching:
        ref_n = ref_notes[ref_idx]
        est_n = est_notes[est_idx]
        
        # Stringが一致しているか判定（ピッチが合っていてStringが合っていればFretも自ずと一致する）
        if ref_n.get('string') == est_n.get('string'):
            correct_string_fret += 1
            
    precision = correct_string_fret / len(est_notes) if est_notes else 0.0
    recall = correct_string_fret / len(ref_notes) if ref_notes else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def main():
    print("="*50)
    print(" E2E PIPELINE BENCHMARK (GuitarSet Mini)")
    print("="*50)
    start_time = time.time()
    
    jams_files = []
    for track in TARGET_TRACKS:
        p = os.path.join(ANNOTATIONS_DIR, f"{track}.jams")
        if os.path.exists(p):
            jams_files.append(p)
        else:
            print(f"Warning: Track {track} not found in {ANNOTATIONS_DIR}")
            
    if not jams_files:
        print("No target tracks found. Aborting.")
        sys.exit(1)

    print(f"Total target tracks: {len(jams_files)}")
    results_pitch = {}
    results_tab = {}

    for i, jams_path in enumerate(jams_files):
        base = os.path.basename(jams_path).replace(".jams", "")
        wav_path = os.path.join(AUDIO_DIR, f"{base}_mic.wav")
        print(f"\n[{i+1}/{len(jams_files)}] {base}")
        
        gt_notes = load_jams_notes_with_string(jams_path)
        if not gt_notes:
            print("-> No GT, skip")
            continue

        # E2E Pipeline 実行用の一時ディレクトリ
        temp_dir = tempfile.mkdtemp(prefix="solotab_e2e_")
        session_id = f"benchmark_{base}"
        
        try:
            t0 = time.time()
            
            # pipeline.py を直接呼び出す
            run_pipeline(
                session_id=session_id,
                session_dir=Path(temp_dir),
                wav_path=Path(wav_path),
                tuning_name="standard",
                skip_demucs=True, # GuitarSetはギター単一トラックのため
                fast_moe=True     # 本番同様にBasic Pitch + MoEの融合を利用
            )
            
            t1 = time.time()
            
            # 出力結果の読み込み
            out_json = os.path.join(temp_dir, "notes_assigned.json")
            if not os.path.exists(out_json):
                print("  [FAIL] notes_assigned.json was not generated.")
                continue
                
            with open(out_json, "r") as f:
                pred_notes = json.load(f)
                
            # --- 評価 ---
            ref_intervals, ref_pitches = to_mireval(gt_notes)
            est_intervals, est_pitches = to_mireval(pred_notes)
            
            if not pred_notes:
                p_pitch, r_pitch, f1_pitch = 0.0, 0.0, 0.0
                p_tab, r_tab, f1_tab = 0.0, 0.0, 0.0
            else:
                p_pitch, r_pitch, f1_pitch, _ = mir_eval.transcription.precision_recall_f1_overlap(
                    ref_intervals, ref_pitches, est_intervals, est_pitches,
                    onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None
                )
                p_tab, r_tab, f1_tab = evaluate_string_fret(
                    gt_notes, pred_notes, ref_intervals, ref_pitches, est_intervals, est_pitches
                )
                
            results_pitch[base] = f1_pitch
            results_tab[base] = f1_tab
            print(f"  [Pitch] F1={f1_pitch:.4f} P={p_pitch:.4f} R={r_pitch:.4f}")
            print(f"  [Tab]   F1={f1_tab:.4f} P={p_tab:.4f} R={r_tab:.4f} | Time: {t1-t0:.1f}s")
            
        except Exception as e:
            print(f"  [ERROR] Processing {base} failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    print("\n" + "="*50)
    print(" FINAL RESULTS (PIPELINE vs PURE MOE BASELINE)")
    print("="*50)
    
    elapsed = time.time() - start_time
    
    all_pitch_f1 = list(results_pitch.values())
    mean_pitch_f1 = np.mean(all_pitch_f1) if all_pitch_f1 else 0.0
    
    all_tab_f1 = list(results_tab.values())
    mean_tab_f1 = np.mean(all_tab_f1) if all_tab_f1 else 0.0
    
    print(f"Total Elapsed Time: {elapsed:.1f}s")
    print(f"Overall Pitch F1 (E2E Pipeline) : {mean_pitch_f1:.4f}")
    print(f"Overall Tab F1   (E2E Pipeline) : {mean_tab_f1:.4f}")
    
    print("\n[Track Breakdown - Pitch F1]")
    # Baseline scores typically recorded from mini_benchmark (Pure MoE): around 0.84-0.90 depending on threshold
    for track in TARGET_TRACKS:
        if track in results_pitch:
            print(f"  {track:25s} : {results_pitch[track]:.4f}")

if __name__ == '__main__':
    main()
