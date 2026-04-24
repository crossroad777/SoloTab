"""
rhythm_quantizer.py — リズム量子化（3連符対応版）
====================================
ノートのタイミングをビートグリッドにスナップし、
TAB譜のリズム表記を自然にする。

16分音符グリッドと3連符グリッドの両方を生成し、
各ノートに最も近いグリッドにスナップする混合量子化。
"""

import numpy as np
from typing import List


def _build_mixed_grid(beats: np.ndarray, straight_subdivisions: int = 4,
                       triplet: bool = True) -> np.ndarray:
    """
    16分音符と3連符の混合グリッドを生成。
    
    Args:
        beats: ビート位置の配列
        straight_subdivisions: ストレート分割数 (4=16分音符, 2=8分音符)
        triplet: 3連符グリッドを含めるか
    """
    grid_set = set()
    
    for i in range(len(beats) - 1):
        beat_start = beats[i]
        beat_end = beats[i + 1]
        beat_duration = beat_end - beat_start
        
        # ストレートグリッド (4分割=16分音符)
        for j in range(straight_subdivisions):
            t = beat_start + beat_duration * j / straight_subdivisions
            grid_set.add(round(t, 6))
        
        # 3連符グリッド (3分割)
        if triplet:
            for j in range(3):
                t = beat_start + beat_duration * j / 3
                grid_set.add(round(t, 6))
    
    # 最後のビートも追加
    grid_set.add(round(beats[-1], 6))
    
    return np.array(sorted(grid_set))


def quantize_notes(notes: List[dict], beats: List[float], 
                   grid_subdivisions: int = 4,
                   snap_threshold_ratio: float = 0.4,
                   enable_triplet: bool = True) -> List[dict]:
    """
    ノートのstart/endをビートグリッド（16分音符+3連符混合）にスナップする。

    Parameters
    ----------
    notes : list[dict]
        ノートリスト (start, end, pitch, string, fret, ...)
    beats : list[float]
        ビート位置の時刻リスト (秒)
    grid_subdivisions : int
        1ビートあたりのストレート分割数 (4=16分音符)
    snap_threshold_ratio : float
        グリッド間隔の何割以内ならスナップするか (0.4=40%以内)
    enable_triplet : bool
        3連符グリッドを有効にするか

    Returns
    -------
    list[dict]
        量子化されたノートリスト
    """
    if not notes or len(beats) < 2:
        return notes

    beats_arr = np.array(beats)

    # 混合グリッド生成（16分音符 + 3連符）
    grid_arr = _build_mixed_grid(beats_arr, grid_subdivisions, triplet=enable_triplet)

    # 平均グリッド間隔
    avg_grid_interval = np.mean(np.diff(grid_arr)) if len(grid_arr) > 1 else 0.1
    snap_threshold = avg_grid_interval * snap_threshold_ratio

    # === 前処理: 微小な時間ズレの和音クラスタリング ===
    # アコースティックギターのストロークやアルペジオにおける物理的な発音ズレ（通常0.01〜0.08秒程度）を、
    # 楽譜上では「同時の和音」として綺麗に縦に並べるために、互いに近いノートの開始時刻を統一する。
    cluster_threshold = 0.06  # 60ms以内のズレは同じ和音とみなす
    notes_sorted = sorted(notes, key=lambda n: n["start"])
    clustered_notes = []
    
    if notes_sorted:
        current_cluster = [notes_sorted[0]]
        for i in range(1, len(notes_sorted)):
            n = notes_sorted[i]
            # 直前のノートとの差ではなく、クラスタ最初のノートとの差で判定する
            if n["start"] - current_cluster[0]["start"] <= cluster_threshold:
                current_cluster.append(n)
            else:
                # クラスタ内のノートの開始時刻を、最初のノートの時刻に統一
                cluster_start = current_cluster[0]["start"]
                for cn in current_cluster:
                    cn_copy = dict(cn)
                    cn_copy["start"] = cluster_start
                    clustered_notes.append(cn_copy)
                current_cluster = [n]
        
        # 最後のクラスタの処理
        if current_cluster:
            cluster_start = current_cluster[0]["start"]
            for cn in current_cluster:
                cn_copy = dict(cn)
                cn_copy["start"] = cluster_start
                clustered_notes.append(cn_copy)
    else:
        clustered_notes = []

    quantized = []
    for note in clustered_notes:
        n = dict(note)  # コピー

        # start をスナップ
        start = n["start"]
        idx = int(np.searchsorted(grid_arr, start))
        idx = max(0, min(idx, len(grid_arr) - 1))

        # 前後のグリッドポイントで近い方を選択
        candidates = []
        if idx > 0:
            candidates.append(grid_arr[idx - 1])
        candidates.append(grid_arr[idx])
        if idx < len(grid_arr) - 1:
            candidates.append(grid_arr[idx + 1])

        snapped_start = min(candidates, key=lambda g: abs(g - start))
        dist = abs(snapped_start - start)

        if dist <= snap_threshold:
            n["start"] = float(snapped_start)
        # else: 閾値超過 → 元のまま (原曲原理主義)

        # end もスナップ (ただしstart以降のグリッドに)
        end = n.get("end", n["start"] + 0.1)
        idx_end = int(np.searchsorted(grid_arr, end))
        idx_end = max(0, min(idx_end, len(grid_arr) - 1))

        candidates_end = []
        if idx_end > 0:
            candidates_end.append(grid_arr[idx_end - 1])
        candidates_end.append(grid_arr[idx_end])
        if idx_end < len(grid_arr) - 1:
            candidates_end.append(grid_arr[idx_end + 1])

        snapped_end = min(candidates_end, key=lambda g: abs(g - end))
        # endはstartより後であること保証
        if snapped_end > n["start"]:
            n["end"] = float(snapped_end)
        else:
            # 最低1グリッド分の長さを確保
            n["end"] = float(n["start"] + avg_grid_interval)

        # 最小ノート長チェック
        if n["end"] - n["start"] < 0.02:
            n["end"] = n["start"] + avg_grid_interval

        quantized.append(n)

    # 重複解消: 同じグリッド位置+同ピッチのノートを統合
    quantized.sort(key=lambda n: (n["start"], n["pitch"]))
    deduped = []
    for n in quantized:
        if deduped and (abs(n["start"] - deduped[-1]["start"]) < 0.001 and
                        n["pitch"] == deduped[-1]["pitch"]):
            # 長い方を残す
            if n["end"] > deduped[-1]["end"]:
                deduped[-1]["end"] = n["end"]
            continue
        deduped.append(n)

    return deduped

