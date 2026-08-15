"""
universal_quantizer.py — 汎用音楽的量子化エンジン (Phase 8 Universal Quantizer)
================================================================================
物理的時間（ミリ秒）と楽曲メタデータ（BPM、拍子、ビート配列）に基づき、
数学的グリッド（16分音符、8分3連符、16分3連符/6連符、8分音符等）への
動的最適スナップと3連符グルーピングを実行する。

設計思想:
  - AIモデルは「物理時間(秒)」と「音高・弦」の推定に徹する。
  - Quantizer が拍ごとに誤差（Straight vs Triplet）を最小化する数学的グリッドを選択。
  - 3連符（8分3連: 4divs, 16分3連: 2divs）を自動グルーピングし、MusicXML/GP5に渡す。
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import math
from typing import List, Dict, Tuple, Optional
import numpy as np

# 1拍あたりの基本解像度 (LCM(4, 3) = 12)
# 4分音符 = 12 divs
# 8分音符 = 6 divs
# 16分音符 = 3 divs
# 8分3連符 = 4 divs
# 16分3連符 = 2 divs
DIVISIONS = 12

# グリッド定義 (拍内オフセット: 0〜11 divs)
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

    Parameters
    ----------
    notes : list[dict]
        生ノート配列 (start, end, pitch, string, fret, velocity, technique, ...)
    beats : list[float]
        ビート位置（秒）
    bpm : float
        テンポ (BPM)
    time_signature : str
        拍子 ("3/4", "4/4", "6/8" 等)
    beats_per_bar : int
        1小節あたりの拍数 (3/4なら3, 4/4なら4)

    Returns
    -------
    list[dict]
        量化済みノートエントリリスト
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
        # t 以下の最大のビートを探す
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
        }
        beat_note_map.setdefault(idx, []).append(item)

    # Step 2: 拍ごとに最適グリッド（Straight vs Triplet）を数学的に判定
    quantized_items: List[dict] = []

    for beat_idx, items in beat_note_map.items():
        bar = beat_idx // beats_per_bar
        beat_in_bar = beat_idx % beats_per_bar

        # 拍内のノート数とオンセット分布からグリッドを評価
        # 同時発音ノート（chord / 15ms以内）をグループとして集約
        time_clusters = []
        for it in items:
            t = it["start_time"]
            if not time_clusters or abs(t - time_clusters[-1]["time"]) > 0.025:
                time_clusters.append({"time": t, "frac": it["frac"], "items": [it]})
            else:
                time_clusters[-1]["items"].append(it)

        n_clusters = len(time_clusters)

        # 誤差計算関数
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

        # グリッド決定ロジック:
        # 1. クラスター数が3個で 3連符誤差が小さい場合 → 8分3連符 (0, 4, 8)
        # 2. クラスター数が5〜6個で 16分3連符誤差が小さい場合 → 16分3連符 (0, 2, 4, 6, 8, 10)
        # 3. それ以外は 誤差の小さい方（Triplet vs Straight）
        selected_grid = GRID_STRAIGHT_16TH
        is_triplet_beat = False
        is_sextuplet = False

        if n_clusters == 3:
            # 3音の場合、3連符の可能性が極めて高い (例: アルペジオ)
            if err_triplet <= err_straight * 1.2 or err_triplet < 0.8:
                selected_grid = GRID_TRIPLET_8TH
                is_triplet_beat = True
        elif n_clusters >= 5:
            if err_triplet_16th < err_straight * 0.8:
                selected_grid = GRID_TRIPLET_16TH
                is_triplet_beat = True
                is_sextuplet = True
            elif err_triplet < err_straight * 0.7:
                selected_grid = GRID_TRIPLET_8TH
                is_triplet_beat = True
        else:
            if err_triplet < err_straight * 0.6 and err_triplet < 0.5:
                selected_grid = GRID_TRIPLET_8TH
                is_triplet_beat = True

        # スナップ実行
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

    # Step 3: 3連符グルーピング（tuplet_role: start, middle, stop）のタグ付け
    quantized_items.sort(key=lambda x: (x["beat_pos_absolute"], float(x["start_time"])))

    # 拍ごとに 3連符ノートを整理
    beat_triplets: Dict[Tuple[int, int], List[dict]] = {}
    for it in quantized_items:
        if it["is_triplet"]:
            key = (it["bar"], it["beat_in_bar"])
            beat_triplets.setdefault(key, []).append(it)

    for (bar, b_in_bar), trip_items in beat_triplets.items():
        # ユニークな beat_pos_in_bar を取得
        pos_set = sorted(list(set(it["beat_pos_in_bar"] for it in trip_items)))
        if len(pos_set) >= 2:  # 少なくとも2つ以上の異なる位置がある場合
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
                it["tuplet_role"] = "start_stop"  # 1音のみ

    # Step 4: 音価（Duration）の計算と持続・ベース音ポリフォニー処理
    # ギター特性: 同弦の次のノートまで持続、異弦は同時に鳴り続ける
    entries: List[dict] = []

    for i, it in enumerate(quantized_items):
        raw_n = it["raw_note"]
        my_string = int(raw_n.get("string", 1))
        pitch = int(raw_n["pitch"])
        pos_abs = it["beat_pos_absolute"]
        pos_in_bar = it["beat_pos_in_bar"]
        bar = it["bar"]

        is_bass = (my_string >= 4) or (pitch <= 52)

        # デフォルト音価:
        # 8分3連符なら 4 divs, 16分音符なら 3 divs, 8分音符なら 6 divs
        if it["is_triplet"]:
            def_dur = 2 if it.get("is_sextuplet") else 4
        else:
            def_dur = 3  # 16th

        # 同一弦の次のノートを探す
        gap_same_string = bar_total_divs
        for j in range(i + 1, len(quantized_items)):
            other = quantized_items[j]
            gap = other["beat_pos_absolute"] - pos_abs
            if gap <= 0:
                continue
            if int(other["raw_note"].get("string", 1)) == my_string:
                gap_same_string = gap
                break

        # 全弦の次のノートを探す
        gap_all_string = bar_total_divs
        for j in range(i + 1, len(quantized_items)):
            other = quantized_items[j]
            gap = other["beat_pos_absolute"] - pos_abs
            if gap > 0:
                gap_all_string = gap
                break

        # 持続時間決定
        if is_bass:
            # ベース音: 小節境界または次のベース音まで長く持続（最低1拍=12divs、最大1小節）
            dur_divs = max(def_dur, min(gap_same_string, bar_total_divs - pos_in_bar))
            dur_divs = max(dur_divs, min(12, bar_total_divs - pos_in_bar))
        else:
            # メロディ音 / アルペジオ音: 基本はグリッド単位（3連なら4divs）または同弦次ノートまで
            dur_divs = min(gap_same_string, max(def_dur, gap_all_string))
            dur_divs = min(dur_divs, bar_total_divs - pos_in_bar)
            dur_divs = max(1, dur_divs)

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
        }
        entries.append(entry)

    # Step 5: 同一弦のグリッド衝突防止（同一弦で同位置になった後発音を押し出し）
    for i in range(len(entries)):
        s_i = entries[i]["string"]
        for j in range(i + 1, len(entries)):
            if entries[j]["string"] == s_i:
                if entries[j]["beat_pos_absolute"] <= entries[i]["beat_pos_absolute"]:
                    if entries[j]["start_time"] > entries[i]["start_time"] + 0.020:
                        shift = 4 if entries[i]["is_triplet"] else 3
                        entries[j]["beat_pos_absolute"] = entries[i]["beat_pos_absolute"] + shift
                        entries[j]["beat_pos_in_bar"] = min(bar_total_divs - 1, entries[i]["beat_pos_in_bar"] + shift)
                        entries[j]["beat_pos"] = entries[j]["beat_pos_in_bar"]
                break

    entries.sort(key=lambda x: (x["beat_pos_absolute"], x["string"]))
    return entries
