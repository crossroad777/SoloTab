"""
End-to-End評価: MoE音検出 → CNN弦分類器 → GuitarSet正解比較

検証内容:
1. MoE が検出した音を CNN弦分類器で弦割り当て
2. GuitarSetの正解(弦+フレット)と照合
3. ノートマッチング: onset±50ms, pitch完全一致
"""
import sys, os, glob, copy, json
import numpy as np
sys.path.insert(0, r'D:\Music\nextchord-solotab\backend')

import jams
from string_assigner import assign_strings_dp, STANDARD_TUNING
from pure_moe_transcriber import transcribe_pure_moe


ANN_DIR = r'D:\Music\Datasets\GuitarSet\annotation'
MIC_DIR = r'D:\Music\Datasets\GuitarSet\audio_mono-mic'

# GuitarSet Test split: Player 05 (36 tracks = 未学習データ)
TEST_PLAYERS = ['05']

ONSET_TOLERANCE = 0.05  # 50ms


def load_gt_notes(jams_path):
    """JAMSから正解ノート (弦+フレット+pitch+onset) を読み込む"""
    jam = jams.load(jams_path)
    gt_notes = []
    note_midi_idx = 0
    for ann in jam.annotations:
        if ann.namespace != 'note_midi':
            continue
        string_num = 6 - note_midi_idx
        note_midi_idx += 1
        if string_num < 1 or string_num > 6:
            continue
        string_idx = 6 - string_num
        for obs in ann.data:
            midi_pitch = int(round(obs.value))
            gt_fret = midi_pitch - STANDARD_TUNING[string_idx]
            if gt_fret < 0 or gt_fret > 19:
                continue
            gt_notes.append({
                'pitch': midi_pitch,
                'start': float(obs.time),
                'duration': float(obs.duration),
                'string': string_num,
                'fret': gt_fret,
            })
    gt_notes.sort(key=lambda n: (n['start'], n['pitch']))
    return gt_notes


def match_notes(pred_notes, gt_notes, onset_tol=ONSET_TOLERANCE):
    """
    Pred vs GT ノートマッチング。
    Returns: (TP_pitch, TP_string_fret, FP, FN)
    - TP_pitch: onset+pitch一致
    - TP_string_fret: onset+pitch+string+fret一致
    """
    gt_matched = [False] * len(gt_notes)
    tp_pitch = 0
    tp_sf = 0
    fp = 0

    for pred in pred_notes:
        p_onset = pred.get('start', 0)
        p_pitch = pred.get('pitch', 0)
        p_string = pred.get('string', 0)
        p_fret = pred.get('fret', 0)

        best_idx = None
        best_dt = float('inf')
        for i, gt in enumerate(gt_notes):
            if gt_matched[i]:
                continue
            dt = abs(p_onset - gt['start'])
            if dt <= onset_tol and p_pitch == gt['pitch'] and dt < best_dt:
                best_dt = dt
                best_idx = i

        if best_idx is not None:
            gt_matched[best_idx] = True
            tp_pitch += 1
            if p_string == gt_notes[best_idx]['string'] and p_fret == gt_notes[best_idx]['fret']:
                tp_sf += 1
        else:
            fp += 1

    fn = sum(1 for m in gt_matched if not m)
    return tp_pitch, tp_sf, fp, fn


def run_e2e_benchmark(players=None, max_tracks=None):
    """End-to-End ベンチマーク"""
    jams_files = sorted(glob.glob(os.path.join(ANN_DIR, '*.jams')))
    
    if players:
        jams_files = [f for f in jams_files if os.path.basename(f)[:2] in players]
    if max_tracks:
        jams_files = jams_files[:max_tracks]
    
    print(f"E2E Benchmark: {len(jams_files)} tracks")
    print(f"Players: {players or 'all'}")
    print()
    
    total_tp_pitch = 0
    total_tp_sf = 0
    total_fp = 0
    total_fn = 0
    total_gt = 0
    
    for jams_path in jams_files:
        basename = os.path.basename(jams_path).replace('.jams', '')
        mic_path = os.path.join(MIC_DIR, basename + '_mic.wav')
        if not os.path.exists(mic_path):
            continue
        
        # Step 1: MoE音検出
        print(f"--- {basename} ---")
        moe_notes = transcribe_pure_moe(mic_path)
        
        # Step 2: CNN弦割り当て
        assigned = assign_strings_dp(
            copy.deepcopy(moe_notes),
            tuning=STANDARD_TUNING,
            audio_path=mic_path
        )
        
        # Step 3: 正解データ
        gt_notes = load_gt_notes(jams_path)
        
        # Step 4: マッチング
        tp_pitch, tp_sf, fp, fn = match_notes(assigned, gt_notes)
        total_tp_pitch += tp_pitch
        total_tp_sf += tp_sf
        total_fp += fp
        total_fn += fn
        total_gt += len(gt_notes)
        
        # トラック別
        p = tp_pitch / max(tp_pitch + fp, 1)
        r = tp_pitch / max(tp_pitch + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-8)
        sf_rate = tp_sf / max(tp_pitch, 1)
        print(f"  Notes: pred={len(assigned)}, gt={len(gt_notes)}")
        print(f"  Pitch: P={p:.3f} R={r:.3f} F1={f1:.3f} (TP={tp_pitch})")
        print(f"  String+Fret: {tp_sf}/{tp_pitch} = {sf_rate:.3f} ({sf_rate*100:.1f}%) of matched notes")
        print()
    
    # 全体結果
    print("=" * 60)
    print(f"=== End-to-End 結果 ({len(jams_files)} tracks) ===")
    print()
    
    p_all = total_tp_pitch / max(total_tp_pitch + total_fp, 1)
    r_all = total_tp_pitch / max(total_tp_pitch + total_fn, 1)
    f1_all = 2 * p_all * r_all / max(p_all + r_all, 1e-8)
    sf_rate_all = total_tp_sf / max(total_tp_pitch, 1)
    
    print(f"音検出:")
    print(f"  Precision: {p_all:.4f}")
    print(f"  Recall:    {r_all:.4f}")
    print(f"  F1:        {f1_all:.4f}")
    print(f"  TP(pitch): {total_tp_pitch}")
    print(f"  FP:        {total_fp}")
    print(f"  FN:        {total_fn}")
    print()
    print(f"弦+フレット (正しく検出された音のうち):")
    print(f"  一致率:    {total_tp_sf}/{total_tp_pitch} = {sf_rate_all:.4f} ({sf_rate_all*100:.2f}%)")
    print()
    print(f"End-to-End (全GT音に対する弦+フレット完全一致):")
    e2e = total_tp_sf / max(total_gt, 1)
    print(f"  完全一致率: {total_tp_sf}/{total_gt} = {e2e:.4f} ({e2e*100:.2f}%)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--players", nargs="+", default=None, help="Player IDs to test")
    parser.add_argument("--max", type=int, default=None, help="Max tracks")
    args = parser.parse_args()
    
    run_e2e_benchmark(players=args.players, max_tracks=args.max)
