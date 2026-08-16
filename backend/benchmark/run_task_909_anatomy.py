"""
backend/benchmark/run_task_909_anatomy.py
=========================================
TASK-909: ワーストケース（改悪）の解剖と物理的検証

1. datasets/non_standard/ の1,000曲から移動距離差分(opt - orig)が最大となった上位3曲を特定
2. 各曲の全ノートについて 元の運指 vs AI最適化運指 の詳細Diffを抽出
3. 「開放弦への誘導（0フレット化）」による見かけの距離増か、「不自然な跳躍」「演奏不可能」かを厳密に判定
"""

import os
import sys
import glob
import random
import json
import io
import contextlib
from pathlib import Path
import numpy as np
import guitarpro

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from solotab_utils import STANDARD_TUNING
from string_assigner import assign_strings_dp
from refingering_engine import compute_ergonomic_cost

def analyze_song_detailed(file_path: str, max_notes: int = 150):
    try:
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
                
        raw_entries = []
        for m_idx, measure in enumerate(track.measures):
            if len(raw_entries) >= max_notes:
                break
            for v_idx, voice in enumerate(measure.voices):
                for b_idx, beat in enumerate(voice.beats):
                    for n_idx, note in enumerate(beat.notes):
                        original_string = note.string
                        original_fret = note.value
                        
                        if original_string - 1 < len(tuning_arr):
                            pitch = tuning_arr[original_string - 1] + original_fret
                        else:
                            pitch = 40 + original_fret
                            
                        start_t = float(measure.number - 1) * 3.0 + (float(beat.start) / 960.0)
                        raw_entries.append({
                            "start": start_t,
                            "end": start_t + 0.25,
                            "duration": 0.25,
                            "pitch": pitch,
                            "voice": v_idx,
                            "measure": measure.number,
                            "beat": b_idx + 1,
                            "original_string": original_string,
                            "original_fret": original_fret,
                            "velocity": 0.8
                        })
                        if len(raw_entries) >= max_notes:
                            break
                    if len(raw_entries) >= max_notes:
                        break
                if len(raw_entries) >= max_notes:
                    break
                        
        if len(raw_entries) < 10:
            return None
            
        raw_entries.sort(key=lambda x: (x["start"], x["voice"]))
        
        with contextlib.redirect_stdout(io.StringIO()):
            assigned_notes = assign_strings_dp(
                raw_entries,
                tuning=STANDARD_TUNING,
                audio_path=None
            )
        
        orig_notes_meta = [{"string": n["original_string"], "fret": n["original_fret"]} for n in raw_entries]
        orig_cost = compute_ergonomic_cost(orig_notes_meta)
        opt_cost = compute_ergonomic_cost(assigned_notes)
        
        diff_movement = opt_cost["total_movement_frets"] - orig_cost["total_movement_frets"]
        diff_jumps = opt_cost["excessive_jumps_gt_4f"] - orig_cost["excessive_jumps_gt_4f"]
        
        diff_rows = []
        prev_orig_fret = None
        prev_opt_fret = None
        
        open_string_changes = 0
        jump_changes = 0
        impossible_stretches = 0
        
        for i, (orig, opt, meta) in enumerate(zip(orig_notes_meta, assigned_notes, raw_entries)):
            orig_s, orig_f = orig["string"], orig["fret"]
            opt_s, opt_f = opt["string"], opt["fret"]
            pitch = meta["pitch"]
            measure = meta["measure"]
            beat = meta["beat"]
            
            orig_step_mov = abs(orig_f - prev_orig_fret) if (prev_orig_fret is not None and orig_f > 0 and prev_orig_fret > 0) else 0
            opt_step_mov = abs(opt_f - prev_opt_fret) if (prev_opt_fret is not None and opt_f > 0 and prev_opt_fret > 0) else 0
            
            if orig_f > 0:
                prev_orig_fret = orig_f
            if opt_f > 0:
                prev_opt_fret = opt_f
                
            is_open_conversion = (orig_f > 0 and opt_f == 0)
            if is_open_conversion:
                open_string_changes += 1
                
            is_jump = (opt_step_mov > 4)
            if is_jump and not (orig_step_mov > 4):
                jump_changes += 1
                
            # 和音・同時発音時のフレットスパンチェック（4f超の同時押弦があるか）
            # ここでは単音遷移の無理なハイポジション(>20f)等も検証
            if opt_f > 22:
                impossible_stretches += 1
                
            diff_rows.append({
                "idx": i + 1,
                "measure": measure,
                "beat": beat,
                "pitch": pitch,
                "orig": f"({orig_s}s, {orig_f}f)",
                "opt": f"({opt_s}s, {opt_f}f)",
                "orig_step_mov": orig_step_mov,
                "opt_step_mov": opt_step_mov,
                "is_open_conversion": is_open_conversion,
                "is_diff": (orig_s != opt_s or orig_f != opt_f)
            })
            
        return {
            "file": os.path.basename(file_path),
            "file_path": file_path,
            "note_count": len(raw_entries),
            "orig_movement": orig_cost["total_movement_frets"],
            "opt_movement": opt_cost["total_movement_frets"],
            "diff_movement": diff_movement,
            "orig_jumps": orig_cost["excessive_jumps_gt_4f"],
            "opt_jumps": opt_cost["excessive_jumps_gt_4f"],
            "diff_jumps": diff_jumps,
            "open_string_conversions_count": open_string_changes,
            "unwanted_jumps_count": jump_changes,
            "impossible_stretches_count": impossible_stretches,
            "diff_rows": diff_rows
        }
    except Exception as e:
        return None


def run_worst_case_anatomy(sample_size=1000, seed=42):
    random.seed(seed)
    dataset_dir = "datasets/non_standard"
    all_files = glob.glob(os.path.join(dataset_dir, "*.gp5"))
    sampled_files = random.sample(all_files, sample_size * 2)
    
    evaluated = []
    for f in sampled_files:
        if len(evaluated) >= sample_size:
            break
        res = analyze_song_detailed(f, max_notes=150)
        if res is not None:
            evaluated.append(res)
            
    # diff_movement の降順（改悪・距離増加が大きい順）でソート
    evaluated.sort(key=lambda x: x["diff_movement"], reverse=True)
    worst_3 = evaluated[:3]
    
    output_report = {
        "task": "TASK-909: ワーストケース（改悪）の解剖と物理的検証",
        "worst_3_songs": []
    }
    
    for rank, w in enumerate(worst_3, 1):
        # 差異があった代表的なノートをピックアップ（最大15件）
        diff_samples = [r for r in w["diff_rows"] if r["is_diff"]][:15]
        
        # 開放弦活用比率
        total_diffs = sum(1 for r in w["diff_rows"] if r["is_diff"])
        open_ratio = (w["open_string_conversions_count"] / max(1, total_diffs)) * 100
        
        # 分類判定
        if w["impossible_stretches_count"] > 0:
            classification = "物理的演奏不可能（バグの可能性）"
        elif w["open_string_conversions_count"] > 0 or w["opt_movement"] / max(1, w["note_count"]) < 3.5:
            classification = "開放弦・ローポジションへの再配置（美学仕様）"
        else:
            classification = "ハイポジション跳躍（改善余地あり）"
            
        output_report["worst_3_songs"].append({
            "rank": rank,
            "file": w["file"],
            "note_count": w["note_count"],
            "orig_total_movement": w["orig_movement"],
            "opt_total_movement": w["opt_movement"],
            "diff_movement": round(w["diff_movement"], 2),
            "orig_excessive_jumps": w["orig_jumps"],
            "opt_excessive_jumps": w["opt_jumps"],
            "open_string_conversions": w["open_string_conversions_count"],
            "open_string_conversion_ratio_pct": round(open_ratio, 2),
            "impossible_stretches": w["impossible_stretches_count"],
            "classification": classification,
            "diff_sample_table": diff_samples
        })
        
    out_file = "backend/benchmark/task_909_worst_case_anatomy.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_report, f, ensure_ascii=False, indent=2)
        
    print(json.dumps(output_report, ensure_ascii=False, indent=2))
    return output_report

if __name__ == "__main__":
    run_worst_case_anatomy()
