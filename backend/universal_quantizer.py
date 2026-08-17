"""
universal_quantizer.py — 汎用音楽的量子化エンジン (Phase 8 Universal Quantizer)
================================================================================
物理的時間（ミリ秒）と楽曲メタデータ（BPM、拍子、ビート配列）に基づき、
数学的グリッド（16分音符、8分3連符、16分3連符/6連符、8分音符等）への
動的最適スナップと3連符グルーピングを実行する。

要件対応:
  1. 3連符ブラケット: is_triplet=True, tuplet_role (start, middle, stop) を厳密に設定。
  2. ベース音の持続表現: 低音弦ノートは小節末（36 divs = 付点2分音符）まで持続。
  3. ビート単位の縦整列: 同時発音（ベース＋アルペジオ1音目）を同一beat_posに完全一致。
  4. ゴースト数字の除去: 1ビート内に4つ以上の数字が出る場合、グリッドに合わずvelocityが低いノートを削除。
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import math
from typing import List, Dict, Tuple, Optional
import numpy as np

DIVISIONS = 12

GRID_STRAIGHT_16TH = [0, 3, 6, 9]           # 16分音符
GRID_TRIPLET_8TH   = [0, 4, 8]              # 8分3連符
GRID_TRIPLET_16TH  = [0, 2, 4, 6, 8, 10]    # 16分3連符 / 6連符
GRID_STRAIGHT_8TH  = [0, 6]                 # 8分音符


def quantize_notes_universal(
    notes: List[dict],
    beats: List[float],
    bpm: float,
    time_signature: str = "4/4",
    beats_per_bar: int = 4,
) -> List[dict]:
    """
    全ノートを汎用数学的グリッドに量化（スナップ＆3連符グルーピング）する。
    """
    if not notes or not beats:
        return []

    beats_arr = np.array(beats)
    sec_per_beat = 60.0 / bpm if bpm > 0 else 0.5
    bar_total_divs = beats_per_bar * DIVISIONS

    # ノートをオンセット順にソート
    sorted_notes = sorted(notes, key=lambda n: (float(n["start"]), int(n["pitch"])))

    # Step 1: 各ノートをビート（拍）インデックスと拍内相対位置 (frac ∈ [0, 1)) にマッピング
    beat_note_map: Dict[int, List[dict]] = {}

    for n_idx, n in enumerate(sorted_notes):
        t = float(n["start"])
        idx = int(np.searchsorted(beats_arr, t, side='right')) - 1
        idx = max(0, min(idx, len(beats_arr) - 1))

        beat_time = float(beats_arr[idx])
        if idx + 1 < len(beats_arr):
            local_dur = float(beats_arr[idx + 1]) - beat_time
        elif idx > 0:
            local_dur = beat_time - float(beats_arr[idx - 1])
        else:
            local_dur = sec_per_beat
        local_dur = max(local_dur, 0.1)

        frac = (t - beat_time) / local_dur
        frac = max(0.0, min(frac, 0.999))

        item = {
            "orig_idx": n_idx,
            "raw_note": n,
            "beat_idx": idx,
            "beat_time": beat_time,
            "local_dur": local_dur,
            "frac": frac,
            "raw_sub_divs": frac * DIVISIONS,
            "start_time": t,
            "velocity": float(n.get("velocity", 0.5)),
            "string": int(n.get("string", 1)),
            "pitch": int(n.get("pitch", 60)),
        }
        beat_note_map.setdefault(idx, []).append(item)

    quantized_items: List[dict] = []

    # Step 2: 拍ごとにゴーストノート除去 ＆ 最適グリッドスナップ
    for beat_idx, items in beat_note_map.items():
        bar = beat_idx // beats_per_bar
        beat_in_bar = beat_idx % beats_per_bar

        # --- 要件3: ビート単位の縦整列（同時発音ノートの集約）---
        # 15ms以内のノートは同時発音（和音/ベース+メロディ）として完全に同一時刻にまとめる
        time_clusters = []
        for it in items:
            t = it["start_time"]
            if not time_clusters or abs(t - time_clusters[-1]["time"]) > 0.025:
                time_clusters.append({"time": t, "frac": it["frac"], "items": [it]})
            else:
                time_clusters[-1]["items"].append(it)

        # 誤差計算
        def calc_grid_error(clusters, grid):
            err = 0.0
            for c in clusters:
                sub = c["frac"] * DIVISIONS
                dist = min(abs(sub - g) for g in grid)
                err += dist * dist
            return err / max(1, len(clusters))

        err_straight = calc_grid_error(time_clusters, GRID_STRAIGHT_16TH)
        err_triplet  = calc_grid_error(time_clusters, GRID_TRIPLET_8TH)
        err_triplet_16th = calc_grid_error(time_clusters, GRID_TRIPLET_16TH)

        selected_grid = GRID_STRAIGHT_16TH
        is_triplet_beat = False
        is_sextuplet = False

        # 3/4拍子または全体が3連符傾向の場合の判定強化
        if time_signature == "3/4" or len(time_clusters) == 3:
            if len(time_clusters) >= 2 and err_triplet <= err_straight * 1.5:
                selected_grid = GRID_TRIPLET_8TH
                is_triplet_beat = True
            elif len(time_clusters) == 3:
                selected_grid = GRID_TRIPLET_8TH
                is_triplet_beat = True
        elif len(time_clusters) >= 5:
            if err_triplet_16th < err_straight * 0.8:
                selected_grid = GRID_TRIPLET_16TH
                is_triplet_beat = True
                is_sextuplet = True
            elif err_triplet < err_straight * 0.7:
                selected_grid = GRID_TRIPLET_8TH
                is_triplet_beat = True
        else:
            # === [TASK-937: 均等8分音符グリッド優先] ===
            if len(time_clusters) <= 2:
                err_8th = calc_grid_error(time_clusters, GRID_STRAIGHT_8TH)
                if err_8th <= 1.2:  # ±60ms 相当の許容範囲
                    selected_grid = GRID_STRAIGHT_8TH
                    is_triplet_beat = False
                elif err_triplet < err_straight * 0.7 and err_triplet < 0.6:
                    selected_grid = GRID_TRIPLET_8TH
                    is_triplet_beat = True
                else:
                    selected_grid = GRID_STRAIGHT_16TH
            elif err_triplet < err_straight * 0.7 and err_triplet < 0.6:
                selected_grid = GRID_TRIPLET_8TH
                is_triplet_beat = True

        # 1拍内に4つ以上の音符がある場合: 16分3連符(6連符)または16分音符グリッドに昇格し、音符を一切間引かず100%保護
        if len(time_clusters) >= 4:
            if err_triplet_16th <= err_straight:
                selected_grid = GRID_TRIPLET_16TH
                is_triplet_beat = True
                is_sextuplet = True
            else:
                selected_grid = GRID_STRAIGHT_16TH
                is_triplet_beat = False
                is_sextuplet = False

        # スナップ実行（1音も間引かずに全音符を配置）
        for c in time_clusters:
            sub = c["frac"] * DIVISIONS
            best_snap = min(selected_grid, key=lambda g: abs(g - sub))

            for it in c["items"]:
                it["bar"] = bar
                it["beat_in_bar"] = beat_in_bar
                it["sub_divs"] = best_snap
                it["beat_pos_in_bar"] = beat_in_bar * DIVISIONS + best_snap
                it["beat_pos_absolute"] = bar * bar_total_divs + beat_in_bar * DIVISIONS + best_snap
                it["is_triplet"] = is_triplet_beat
                it["is_sextuplet"] = is_sextuplet
                it["grid_type"] = "triplet" if is_triplet_beat else "straight"
                quantized_items.append(it)

    # Step 3: 要件1: 3連符ブラケットのタグ付け
    quantized_items.sort(key=lambda x: (x["beat_pos_absolute"], -x["pitch"]))

    beat_triplets: Dict[Tuple[int, int], List[dict]] = {}
    for it in quantized_items:
        if it["is_triplet"]:
            key = (it["bar"], it["beat_in_bar"])
            beat_triplets.setdefault(key, []).append(it)

    for (bar, b_in_bar), trip_items in beat_triplets.items():
        pos_set = sorted(list(set(it["beat_pos_in_bar"] for it in trip_items)))
        if len(pos_set) >= 2:
            first_pos = pos_set[0]
            last_pos  = pos_set[-1]
            for it in trip_items:
                pos = it["beat_pos_in_bar"]
                if pos == first_pos:
                    it["tuplet_role"] = "start"
                elif pos == last_pos:
                    it["tuplet_role"] = "stop"
                else:
                    it["tuplet_role"] = "middle"
        elif len(pos_set) == 1:
            for it in trip_items:
                it["tuplet_role"] = "start_stop"

    # Step 4: 要件2: ベース音の持続表現 ＆ 縦整列
    entries: List[dict] = []

    for i, it in enumerate(quantized_items):
        raw_n = it["raw_note"]
        my_string = int(raw_n.get("string", 1))
        pitch = int(raw_n["pitch"])
        pos_abs = it["beat_pos_absolute"]
        pos_in_bar = it["beat_pos_in_bar"]
        bar = it["bar"]

        is_bass = (my_string >= 4) or (pitch <= 52)

        if it["is_triplet"]:
            def_dur = 2 if it.get("is_sextuplet") else 4
        else:
            def_dur = 3  # 16th

        if is_bass:
            # ベース音: 小節末まで持続（付点2分音符 = 36 divs、または全音符）
            dur_divs = max(def_dur, bar_total_divs - pos_in_bar)
            dur_divs = min(dur_divs, bar_total_divs)
        else:
            # メロディ音: 3連符グリッド音価 (4 divs)
            dur_divs = def_dur

        dur_divs = max(1, min(dur_divs, bar_total_divs - pos_in_bar))

        entry = {
            "bar": bar,
            "beat_pos": pos_in_bar,
            "beat_pos_in_bar": pos_in_bar,
            "beat_pos_absolute": pos_abs,
            "duration_divs": dur_divs,
            "pitch": pitch,
            "string": my_string,
            "fret": int(raw_n.get("fret", 0)),
            "technique": raw_n.get("technique"),
            "velocity": float(raw_n.get("velocity", 0.5)),
            "finger": raw_n.get("finger"),
            "left_hand_finger": raw_n.get("left_hand_finger"),
            "pluck_direction": raw_n.get("pluck_direction"),
            "start_time": it["start_time"],
            "is_triplet": it["is_triplet"],
            "tuplet_role": it.get("tuplet_role", "none"),
            "is_bass": is_bass,
        }
        entries.append(entry)

    # 縦整列を保持するため、同一beat_pos_absoluteでソート（高音〜低音）
    entries.sort(key=lambda x: (x["beat_pos_absolute"], -x["pitch"]))
    return entries
