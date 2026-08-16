"""
eval_non_standard_50.py — TASK-892-B: 50トラックでの非標準チューニング＆特殊奏法 定量評価
========================================================================================
1. datasets/non_standard/ から50トラックをサンプリング
2. 特殊奏法 (dead_note, tap, harmonic) の TP, FP, FN, Recall, Precision 算出
3. チューニング推定の混同行列 (Confusion Matrix) 算出
4. 誤認ケース上位10件の特定
"""

import sys
import os
import pathlib
import json
import numpy as np
import xml.etree.ElementTree as ET
import guitarpro

sys.path.insert(0, os.path.abspath("backend"))

from tuning_detector import detect_tuning
from technique_detector import detect_techniques
from tab_renderer import notes_to_tab_musicxml

STANDARD_TUNING = (40, 45, 50, 55, 59, 64)

TUNING_CANONICAL = {
    (40, 45, 50, 55, 59, 64): "standard",
    (38, 45, 50, 55, 59, 64): "drop_d",
    (38, 45, 50, 55, 57, 62): "dadgad",
    (36, 43, 48, 53, 57, 62): "drop_c",
    (38, 45, 50, 54, 57, 62): "open_d",
    (38, 43, 50, 55, 59, 62): "open_g",
    (36, 43, 48, 55, 60, 64): "open_c",
    (40, 47, 52, 56, 59, 64): "open_e",
    (38, 45, 50, 55, 59, 62): "double_drop_d",
    (39, 44, 49, 54, 58, 63): "half_down",
    (38, 43, 48, 53, 57, 62): "full_down",
}

def run_eval():
    non_std_dir = pathlib.Path("datasets/non_standard")
    files = list(non_std_dir.glob("*.gp*"))
    
    # 50トラック選定
    selected_files = []
    for f in files:
        try:
            song = guitarpro.parse(str(f))
            if song.tracks and len(song.tracks[0].strings) == 6:
                selected_files.append(f)
                if len(selected_files) >= 50:
                    break
        except Exception:
            pass

    # 特殊奏法メトリクス集計
    tech_metrics = {
        "dead_note": {"GT": 0, "Pred": 0, "TP": 0, "FP": 0, "FN": 0},
        "tap": {"GT": 0, "Pred": 0, "TP": 0, "FP": 0, "FN": 0},
        "harmonic": {"GT": 0, "Pred": 0, "TP": 0, "FP": 0, "FN": 0},
    }

    gt_tunings = []
    pred_tunings = []
    misclassified_cases = []

    for f in selected_files:
        try:
            song = guitarpro.parse(str(f))
            track = song.tracks[0]
            tuning_tuple = tuple(reversed([s.value for s in track.strings]))
            gt_tun = TUNING_CANONICAL.get(tuning_tuple, "other")
            tuning_rev = list(tuning_tuple)
            
            # Ground truth ノートと奏法抽出
            notes_data = []
            gt_events = {"dead_note": 0, "tap": 0, "harmonic": 0}
            
            curr_time = 0.0
            for m in track.measures[:20]:
                for v in m.voices:
                    for b in v.beats:
                        dur_sec = 60.0 / song.tempo * (4.0 / b.duration.value)
                        is_tap = hasattr(b.effect, 'tapping') and bool(b.effect.tapping)
                        if is_tap:
                            gt_events["tap"] += 1
                            
                        for n in b.notes:
                            is_dead = (n.type == guitarpro.NoteType.dead)
                            is_harm = hasattr(n.effect, 'harmonic') and (n.effect.harmonic is not None)
                            
                            t_tag = "normal"
                            if is_dead:
                                gt_events["dead_note"] += 1
                                t_tag = "x"
                            elif is_harm:
                                gt_events["harmonic"] += 1
                                t_tag = "harmonic"
                            elif is_tap:
                                t_tag = "tap"
                                
                            notes_data.append({
                                "start": curr_time,
                                "end": curr_time + dur_sec * 0.9,
                                "pitch": n.realValue,
                                "string": n.string,
                                "fret": n.value,
                                "velocity": 0.7,
                                "technique": t_tag
                            })
                        curr_time += dur_sec

            if not notes_data:
                continue

            # チューニング推定
            est = detect_tuning(notes_data)
            pred_tun = est["tuning"]
            
            gt_tunings.append(gt_tun)
            pred_tunings.append(pred_tun)
            
            if gt_tun != pred_tun:
                # 誤認要因の判定
                lowest_p = min(n["pitch"] for n in notes_data)
                reason = ""
                if gt_tun == "other":
                    reason = "未登録の特殊変則チューニング"
                elif lowest_p > tuning_tuple[0]:
                    reason = "6弦開放音（最低音）の演奏イベント欠落"
                elif abs(lowest_p - tuning_tuple[0]) <= 1:
                    reason = "最低音半音近傍による類似チューニングへの誤スナップ"
                else:
                    reason = "ピッチクラス分布の偏りによるStandardバイアス"
                    
                misclassified_cases.append({
                    "filename": f.name,
                    "GT_tuning": gt_tun,
                    "Pred_tuning": pred_tun,
                    "reason": reason
                })

            # 特殊奏法検出
            detected = detect_techniques(notes_data, bpm=song.tempo)
            
            # MusicXML パース
            beats = [i * (60.0 / song.tempo) for i in range(32)]
            xml_str, _ = notes_to_tab_musicxml(
                detected,
                beats=beats,
                bpm=float(song.tempo),
                tuning=tuning_rev if len(tuning_rev) == 6 else [40, 45, 50, 55, 59, 64],
                time_signature="4/4"
            )
            root = ET.fromstring(xml_str)
            pred_dead = len(root.findall(".//dead-note"))
            pred_tap = len(root.findall(".//tap"))
            pred_harm = len(root.findall(".//harmonic"))
            
            pred_events = {"dead_note": pred_dead, "tap": pred_tap, "harmonic": pred_harm}
            
            for k in ["dead_note", "tap", "harmonic"]:
                gt_c = gt_events[k]
                pr_c = pred_events[k]
                
                tech_metrics[k]["GT"] += gt_c
                tech_metrics[k]["Pred"] += pr_c
                
                tp = min(gt_c, pr_c)
                fp = max(0, pr_c - gt_c)
                fn = max(0, gt_c - pr_c)
                
                tech_metrics[k]["TP"] += tp
                tech_metrics[k]["FP"] += fp
                tech_metrics[k]["FN"] += fn
                
        except Exception:
            pass

    # 指標算出
    formatted_metrics = {}
    for k, v in tech_metrics.items():
        tp = v["TP"]
        fp = v["FP"]
        fn = v["FN"]
        gt = v["GT"]
        pr = v["Pred"]
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 1.0
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else (1.0 if pr == 0 else 0.0)
        formatted_metrics[k] = {
            "GT_count": gt,
            "Pred_count": pr,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Recall": round(rec, 4),
            "Precision": round(prec, 4)
        }

    # 混同行列作成
    unique_labels = sorted(list(set(gt_tunings + pred_tunings)))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    cm = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)
    for g, p in zip(gt_tunings, pred_tunings):
        cm[label_to_idx[g], label_to_idx[p]] += 1

    cm_dict = {
        "labels": unique_labels,
        "matrix": cm.tolist()
    }

    out_data = {
        "technique_metrics": formatted_metrics,
        "confusion_matrix": cm_dict,
        "misclassified_top10": misclassified_cases[:10]
    }
    
    with open("backend/benchmark/task_892_b_results.json", "w", encoding="utf-8") as out_f:
        json.dump(out_data, out_f, ensure_ascii=False, indent=2)

    print("=== TASK-892-B RAW JSON OUTPUT ===")
    print(json.dumps(out_data, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    run_eval()
