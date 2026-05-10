"""
music_theory.py — 音楽理論エンジン
===================================
ML出力の生ノートデータを音楽的に正しい楽譜表記に変換する。

機能:
  1. 音価推論 (Duration inference) — 実測値を音楽的音価にスナップ
  2. 休符判定 — 意図的な休符 vs レガート持続の区別
  3. タイ検出 — 拍/小節をまたぐ音の持続
  4. 付点音符検出
  5. 調号推定 — ノートのピッチ分布から調を推定
  6. 強弱マッピング — velocity → ダイナミクス記号
  7. 声部分離 — ベースラインとメロディの分離
  8. リズムパターン検出 — 3連符 vs ストレート
"""
from typing import List, Optional, Tuple
import math


# ─── 定数 ───

DIVISIONS = 12  # 1拍あたりの分割数 (LCM of 4 and 3)

# 有効な音価 (divisions単位)
# whole=48, half=24, quarter=12, 8th=6, 16th=3, triplet-8th=4, triplet-quarter=8
VALID_DURATIONS_STRAIGHT = [3, 6, 9, 12, 18, 24, 36, 48]   # 16th, 8th, dot-8th, quarter, dot-quarter, half, dot-half, whole
VALID_DURATIONS_TRIPLET = [4, 8, 12, 24, 48]                 # trip-8th, trip-quarter, quarter, half, whole
VALID_DURATIONS_ALL = sorted(set(VALID_DURATIONS_STRAIGHT + VALID_DURATIONS_TRIPLET))

# 付点音価のマッピング: base_dur → dotted_dur
DOTTED_MAP = {
    # 3: 4.5 は実用上 3連8分(4)と衝突するため除外
    6: 9,    # dotted 8th = 9
    12: 18,  # dotted quarter = 18
    24: 36,  # dotted half = 36
}

# ダイナミクスマッピング
DYNAMICS_MAP = [
    (0.15, "pp"),
    (0.35, "p"),
    (0.50, "mp"),
    (0.65, "mf"),
    (0.80, "f"),
    (0.95, "ff"),
    (1.00, "fff"),
]

# 調号の判定用: 各調のピッチクラスセット
# major keys - pitch classes (0=C, 1=C#, ... 11=B)
KEY_SIGNATURES = {
    "C":  [0, 2, 4, 5, 7, 9, 11],
    "G":  [0, 2, 4, 6, 7, 9, 11],
    "D":  [1, 2, 4, 6, 7, 9, 11],
    "A":  [1, 2, 4, 6, 8, 9, 11],
    "E":  [1, 3, 4, 6, 8, 9, 11],
    "B":  [1, 3, 4, 6, 8, 10, 11],
    "F":  [0, 2, 4, 5, 7, 9, 10],
    "Bb": [0, 2, 3, 5, 7, 9, 10],
    "Eb": [0, 2, 3, 5, 7, 8, 10],
    "Ab": [0, 1, 3, 5, 7, 8, 10],
    # minor keys (natural minor = relative major)
    "Am": [0, 2, 4, 5, 7, 9, 11],  # same as C
    "Em": [0, 2, 4, 6, 7, 9, 11],  # same as G
    "Dm": [0, 2, 3, 5, 7, 9, 10],  # same as F
    "Bm": [1, 2, 4, 6, 7, 9, 11],  # same as D
}


# ─── 1. 音価推論 ───

def snap_duration(raw_divs: int, is_triplet: bool = False, beat_total: int = 12) -> Tuple[int, bool, bool]:
    """
    生のdivision値を最も近い音楽的音価にスナップする。
    
    Returns: (snapped_divs, is_dotted, is_triplet_note)
    """
    if raw_divs <= 0:
        return (1, False, False)
    
    if is_triplet:
        valid = VALID_DURATIONS_TRIPLET
    else:
        valid = VALID_DURATIONS_STRAIGHT
    
    # 最近傍スナップ
    best = min(valid, key=lambda v: abs(v - raw_divs))
    
    # 付点判定: bestが付点音価に一致するか
    is_dotted = False
    is_trip = best in [4, 8]  # 4=triplet-8th, 8=triplet-quarter
    
    for base, dotted in DOTTED_MAP.items():
        if best == dotted:
            is_dotted = True
            break
    
    return (best, is_dotted, is_trip)


def quantize_note_durations(entries: List[dict], is_triplet_mode: bool = False,
                             beats_per_bar: int = 4) -> List[dict]:
    """
    ノートリストの duration_divs を音楽的音価にスナップする。
    同時に is_dotted, is_triplet フラグを設定。
    """
    bar_total = beats_per_bar * DIVISIONS
    
    for entry in entries:
        raw = int(entry.get("duration_divs", DIVISIONS))
        snapped, dotted, triplet = snap_duration(raw, is_triplet=is_triplet_mode)
        snapped = min(snapped, bar_total)  # 小節を超えない
        entry["duration_divs"] = snapped
        entry["is_dotted"] = dotted
        entry["is_triplet"] = triplet
    
    return entries


# ─── 2. 休符判定 ───

def classify_gaps(entries: List[dict], beat_interval: float, 
                  min_rest_fraction: float = 0.2) -> List[dict]:
    """
    ノート間のギャップを分類:
    - 短いギャップ (< min_rest_fraction beat) → レガート（ノートを延長）
    - 長いギャップ → 意図的な休符
    
    min_rest_fraction: 1拍の何割以下なら休符と見なさないか (0.2 = 16分音符未満)
    """
    if not entries:
        return entries
    
    min_gap_sec = beat_interval * min_rest_fraction
    sorted_entries = sorted(entries, key=lambda e: float(e.get("start_time", 0)))
    
    for i in range(len(sorted_entries)):
        e = sorted_entries[i]
        note_end = float(e.get("start_time", 0)) + float(e.get("end_time", float(e.get("start_time", 0)) + 0.5)) - float(e.get("start_time", 0))
        
        if i + 1 < len(sorted_entries):
            next_start = float(sorted_entries[i + 1].get("start_time", 0))
            gap = next_start - float(e.get("start_time", 0))
            
            if gap < min_gap_sec:
                # レガート: このノートの実効デュレーションを次のノートまで延長
                e["_legato"] = True
            else:
                e["_legato"] = False
        else:
            e["_legato"] = False
    
    return sorted_entries


# ─── 3. タイ検出 ───

def detect_ties(entries: List[dict], beats_per_bar: int = 4) -> List[dict]:
    """
    ノートの音価が拍境界を超える場合、タイとしてマーク。
    MusicXML生成時に <tie> 要素として出力するためのフラグを設定。
    """
    bar_total = beats_per_bar * DIVISIONS
    
    for entry in entries:
        beat_pos = int(entry.get("beat_pos_in_bar", 0))
        dur = int(entry.get("duration_divs", DIVISIONS))
        
        if beat_pos + dur > bar_total:
            # 小節をまたぐ → タイが必要
            entry["_tie_start"] = True
            entry["_overflow_divs"] = beat_pos + dur - bar_total
        else:
            entry["_tie_start"] = False
            entry["_overflow_divs"] = 0
    
    return entries


# ─── 4. 調号推定 ───

def detect_key_signature(notes: List[dict]) -> str:
    """
    ノートのピッチ分布から最も可能性の高い調を推定。
    Krumhansl-Schmuckler アルゴリズムの簡易版。
    """
    if not notes:
        return "C"
    
    # ピッチクラスのヒストグラム
    pitch_hist = [0] * 12
    for n in notes:
        pc = int(n.get("pitch", 60)) % 12
        pitch_hist[pc] += 1
    
    total = sum(pitch_hist)
    if total == 0:
        return "C"
    
    # 各調とのマッチングスコア
    best_key = "C"
    best_score = -1
    
    for key_name, scale_pcs in KEY_SIGNATURES.items():
        score = sum(pitch_hist[pc] for pc in scale_pcs) / total
        if score > best_score:
            best_score = score
            best_key = key_name
    
    return best_key


# ─── 5. 強弱マッピング ───

def velocity_to_dynamic(velocity: float) -> Optional[str]:
    """velocity値 (0.0-1.0) をダイナミクス記号に変換"""
    if velocity > 1.0:
        velocity /= 127.0
    for threshold, label in DYNAMICS_MAP:
        if velocity <= threshold:
            return label
    return "f"


def assign_dynamics_to_bars(entries: List[dict], beats_per_bar: int = 4) -> dict:
    """
    各小節の平均velocityからダイナミクスを割り当て。
    Returns: {bar_number: dynamic_marking}
    """
    bar_velocities = {}
    for e in entries:
        bar = int(e.get("bar", 0))
        vel = float(e.get("velocity", 0.5))
        if vel > 1.0:
            vel /= 127.0
        if bar not in bar_velocities:
            bar_velocities[bar] = []
        bar_velocities[bar].append(vel)
    
    bar_dynamics = {}
    prev_dyn = None
    for bar in sorted(bar_velocities.keys()):
        avg_vel = sum(bar_velocities[bar]) / len(bar_velocities[bar])
        dyn = velocity_to_dynamic(avg_vel)
        # 前の小節と同じなら繰り返さない
        if dyn != prev_dyn:
            bar_dynamics[bar] = dyn
            prev_dyn = dyn
    
    return bar_dynamics


# ─── 6. 声部分離 ───

def separate_voices(entries: List[dict], split_pitch: int = 52) -> Tuple[List[dict], List[dict]]:
    """
    ピッチに基づいて2声部に分離。
    split_pitch以下 → voice 2 (ベース)
    split_pitch超   → voice 1 (メロディ)
    
    split_pitch=52 → E3 (ギターの3弦開放付近)
    """
    voice1 = []  # メロディ
    voice2 = []  # ベース
    
    for e in entries:
        pitch = int(e.get("pitch", 60))
        if pitch <= split_pitch:
            e["voice"] = 2
            voice2.append(e)
        else:
            e["voice"] = 1
            voice1.append(e)
    
    return voice1, voice2


# ─── 7. リズムパターン検出 ───

def detect_rhythm_pattern(notes: List[dict], beats: List[float]) -> dict:
    """
    ノートのonsetタイミングからリズムパターンを分析。
    
    Returns: {
        "subdivision": "triplet" | "straight" | "mixed",
        "triplet_ratio": float (0.0-1.0),
        "swing_ratio": float,
        "dominant_duration": int (divisions)
    }
    """
    import numpy as np
    
    result = {
        "subdivision": "straight",
        "triplet_ratio": 0.0,
        "swing_ratio": 0.0,
        "dominant_duration": DIVISIONS,
    }
    
    if not notes or not beats or len(beats) < 3:
        return result
    
    beats_arr = np.array(beats)
    triplet_count = 0
    straight_count = 0
    onset_fracs = []
    
    for n in notes:
        t = float(n["start"])
        bidx = max(0, int(np.searchsorted(beats_arr, t, side='right')) - 1)
        if bidx < len(beats_arr) - 1:
            bt = float(beats_arr[bidx])
            nbt = float(beats_arr[bidx + 1])
            if nbt > bt:
                frac = (t - bt) / (nbt - bt)
                frac = frac % 1.0
                onset_fracs.append(frac)
                
                # 3連符グリッド: 0, 1/3, 2/3
                triplet_dist = min(abs(frac - g) for g in [0, 1/3, 2/3, 1.0])
                # ストレートグリッド: 0, 1/4, 1/2, 3/4
                straight_dist = min(abs(frac - g) for g in [0, 0.25, 0.5, 0.75, 1.0])
                
                if triplet_dist < straight_dist:
                    triplet_count += 1
                else:
                    straight_count += 1
    
    total = triplet_count + straight_count
    if total > 0:
        result["triplet_ratio"] = triplet_count / total
    
    if result["triplet_ratio"] > 0.4:
        result["subdivision"] = "triplet"
        result["dominant_duration"] = 4  # triplet 8th
    elif result["triplet_ratio"] > 0.2:
        result["subdivision"] = "mixed"
    else:
        result["subdivision"] = "straight"
        # 最頻の音価を推定
        if onset_fracs:
            # 隣接onset間の差分から最頻の分割を推定
            intervals = []
            sorted_fracs = sorted(onset_fracs)
            for i in range(1, len(sorted_fracs)):
                diff = sorted_fracs[i] - sorted_fracs[i-1]
                if diff > 0.05:
                    intervals.append(diff)
            if intervals:
                avg_interval = np.median(intervals)
                # fraction → divisions
                dom_dur = max(1, int(round(avg_interval * DIVISIONS)))
                dom_dur_snapped, _, _ = snap_duration(dom_dur, is_triplet=False)
                result["dominant_duration"] = dom_dur_snapped
    
    # スウィング検出 (3連符の最初と2番目の比率)
    if onset_fracs and result["subdivision"] == "triplet":
        # 1/3付近のonsetと2/3付近のonsetの比率からswing度合いを推定
        near_third = [f for f in onset_fracs if 0.2 < f < 0.45]
        near_two_third = [f for f in onset_fracs if 0.55 < f < 0.78]
        if near_third and near_two_third:
            avg_first = np.mean(near_third)
            avg_second = np.mean(near_two_third)
            result["swing_ratio"] = avg_first / avg_second if avg_second > 0 else 0.5
    
    return result


# ─── 8. 統合パイプライン ───

def apply_music_theory(notes: List[dict], beats: List[float], 
                        bpm: float, beats_per_bar: int = 4,
                        time_signature: str = "4/4") -> dict:
    """
    全ての音楽理論処理を統合的に適用。
    
    Returns: {
        "rhythm_info": dict,
        "key_signature": str,
        "bar_dynamics": dict,
        "notes": List[dict] (処理済み),
    }
    """
    # 1. リズムパターン検出
    rhythm_info = detect_rhythm_pattern(notes, beats)
    
    # 2. 調号推定
    key_sig = detect_key_signature(notes)
    
    # 3. 強弱マッピング (entriesに変換前の段階で)
    # bar_dynamicsは後でentries化した後に計算
    
    return {
        "rhythm_info": rhythm_info,
        "key_signature": key_sig,
        "notes": notes,
    }
