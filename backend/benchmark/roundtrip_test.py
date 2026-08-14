import sys
import os
import json
import time
import shutil
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from synth.guitar_synth import SoundFontSynth, DEFAULT_SF2
from synth.karplus_strong import KarplusStrongSynth
from pipeline import run_pipeline
import soundfile as sf

def evaluate_predictions(y_true, y_pred, window_ms=50.0):
    if not y_true:
        return (0.0, 0.0, 0.0, 0, len(y_pred))
    
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

def run_roundtrip_test():
    print("=== SoloTab Round-Trip Recall Upper Bound Test ===")
    
    # 1. 動作可能なシンセサイザーの判定 (Fallbackロジック)
    synth = None
    synth_name = ""
    
    try:
        import fluidsynth
        if os.path.exists(DEFAULT_SF2):
            print("=> SoundFontSynth (FluidSynth + FluidR3_GM.sf2) が利用可能です。")
            synth = SoundFontSynth(sr=22050, sf2_path=DEFAULT_SF2)
            synth_name = "SoundFont"
        else:
            raise FileNotFoundError(f"SoundFont not found: {DEFAULT_SF2}")
    except Exception as e:
        print(f"=> SoundFontSynth は利用できません ({e})。")
        print("=> フォールバック: KarplusStrongSynth (物理モデル合成) を使用します。")
        synth = KarplusStrongSynth(sr=22050)
        synth_name = "Karplus-Strong"
    
    # --- Pattern 1: Single note scale (C major) ---
    p1_events = []
    t = 0.0
    bpm = 100.0
    beat = 60.0 / bpm
    scale_pitches = [48, 50, 52, 53, 55, 57, 59, 60, 62, 64, 65, 67, 69, 71, 72]
    for p in scale_pitches:
        p1_events.append({"pitch": p, "start": t, "duration": beat * 0.9, "velocity": 0.8})
        t += beat
        
    # --- Pattern 2: Open Chords (C, G, Am, F) ---
    p2_events = []
    t = 0.0
    bpm = 100.0
    beat = 60.0 / bpm
    chords = [
        [48, 52, 55, 60, 64], # C
        [43, 47, 50, 55, 59, 67], # G
        [45, 52, 57, 60, 64], # Am
        [41, 48, 53, 57, 60, 65]  # F
    ]
    for c in chords:
        for p in c:
            p2_events.append({"pitch": p, "start": t, "duration": beat * 3.8, "velocity": 0.85})
        t += beat * 4
        
    # --- Pattern 3: High-density 16th notes ---
    p3_events = []
    t = 0.0
    bpm = 120.0
    sub_beat = (60.0 / bpm) / 4.0
    for i in range(32):
        pitch = 50 + (i % 12)
        p3_events.append({"pitch": pitch, "start": t, "duration": sub_beat * 0.9, "velocity": 0.9})
        t += sub_beat
        
    # --- Pattern 4: Arpeggio (Romance style) ---
    p4_events = []
    t = 0.0
    bpm = 80.0
    sub_beat = (60.0 / bpm) / 3.0
    for i in range(4):
        p4_events.append({"pitch": 40, "start": t, "duration": sub_beat * 2.8, "velocity": 0.8})
        for _ in range(3):
            p4_events.append({"pitch": 64, "start": t, "duration": sub_beat * 0.9, "velocity": 0.7})
            t += sub_beat
            p4_events.append({"pitch": 59, "start": t, "duration": sub_beat * 0.9, "velocity": 0.6})
            t += sub_beat
            p4_events.append({"pitch": 55, "start": t, "duration": sub_beat * 0.9, "velocity": 0.6})
            t += sub_beat

    # --- Pattern 5: 2-voice (Blackbird style) ---
    p5_events = []
    t = 0.0
    bpm = 90.0
    beat = 60.0 / bpm
    for i in range(8):
        p5_events.append({"pitch": 43 + (i%5), "start": t, "duration": beat * 0.9, "velocity": 0.8})
        p5_events.append({"pitch": 67 + (i%5), "start": t + beat*0.5, "duration": beat * 0.4, "velocity": 0.7})
        t += beat

    patterns = [
        {"name": "Pattern 1: Single note scale", "events": p1_events},
        {"name": "Pattern 2: Chords", "events": p2_events},
        {"name": "Pattern 3: High-density fast picking", "events": p3_events},
        {"name": "Pattern 4: Arpeggio", "events": p4_events},
        {"name": "Pattern 5: 2-voice melody", "events": p5_events},
    ]

    temp_dir = Path(os.path.dirname(__file__)) / "temp_sessions"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    total_true = 0
    total_pred = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    results_report = f"# ラウンドトリップテスト結果 (Recall 上限検証)\n\n"
    results_report += f"**使用合成方式**: {synth_name}\n\n"
    
    for i, pat in enumerate(patterns):
        name = pat["name"]
        events = pat["events"]
        
        session_id = f"test_session_{i}"
        session_dir = temp_dir / session_id
        os.makedirs(session_dir, exist_ok=True)
        
        # 1. Synthesize WAV
        audio = synth.synthesize_sequence(events)
        wav_path = session_dir / "input.wav"
        sf.write(str(wav_path), audio, 22050)
        
        # 2. Transcribe using the pipeline
        print(f"\nProcessing {name}...")
        t0 = time.time()
        
        # pipeline.py uses its internal methods for Demucs, BasicPitch, etc.
        # we will set skip_demucs=True to avoid separating a perfectly clean solo guitar track!
        # (This also speeds up testing significantly)
        run_pipeline(session_id, session_dir, wav_path, skip_demucs=True, fast_moe=True, progress_cb=lambda s, m: None)
        
        t_elapsed = time.time() - t0
        
        # Read the pipeline output
        notes_assigned_path = session_dir / "notes_assigned.json"
        pred_normalized = []
        if notes_assigned_path.exists():
            with open(notes_assigned_path, "r", encoding="utf-8") as f:
                pipeline_data = json.load(f)
            
            # pipeline_data might be a list or dict depending on pipeline version
            if isinstance(pipeline_data, dict):
                pred_notes = pipeline_data.get("notes", [])
            else:
                pred_notes = pipeline_data
                
            pred_normalized = [{"pitch": n["pitch"], "start": n.get("start", n.get("start_time", 0.0))} for n in pred_notes]
        
        # 3. Evaluate
        precision, recall, f1, tp, fp, fn = evaluate_predictions(events, pred_normalized)
        
        print(f"[{name}] GT: {len(events)} | Pred: {len(pred_normalized)} | P: {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f}")
        
        results_report += f"### {name}\n"
        results_report += f"- Ground Truth ノート数: {len(events)}\n"
        results_report += f"- 検出ノート数: {len(pred_normalized)}\n"
        results_report += f"- Metrics: Precision={precision:.4f} | Recall={recall:.4f} | F1={f1:.4f}\n\n"
        
        total_true += len(events)
        total_pred += len(pred_normalized)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
    print("\n=======================================================")
    print("=== OVERALL ROUND-TRIP RESULTS ===")
    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0
    
    print(f"Total Ground Truth Notes: {total_true}")
    print(f"Total Detected Notes:     {total_pred}")
    print(f"Overall Precision: {overall_p:.4f}")
    print(f"Overall Recall:    {overall_r:.4f}")
    print(f"Overall F1-Score:  {overall_f1:.4f}")
    
    diff = overall_r - 0.8430
    print(f"\n[Comparison] Overall Recall vs GuitarSet Benchmark (0.8430)")
    print(f"Difference: {'+' if diff > 0 else ''}{diff:.4f}")
    
    results_report += f"## 総合結果 (Overall)\n"
    results_report += f"- Total Ground Truth: {total_true}\n"
    results_report += f"- Total Detected: {total_pred}\n"
    results_report += f"- **Overall Recall: {overall_r:.4f}**\n"
    results_report += f"- (GuitarSet Recall: 0.8430)\n"
    results_report += f"- 差分: {'+' if diff > 0 else ''}{diff:.4f}\n\n"
    
    # Write report
    report_path = os.path.join(os.path.dirname(__file__), "roundtrip_results.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(results_report)
    print(f"\nReport written to: {report_path}")

if __name__ == "__main__":
    run_roundtrip_test()
