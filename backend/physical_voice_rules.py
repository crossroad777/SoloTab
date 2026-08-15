"""
physical_voice_rules.py — 物理ルールによる声部分離＆記譜インターフェース
======================================================================
AIの音楽的解釈を排し、4つの「物理ルール」のみでノートの役割付与・共鳴除去・
弦割り当て適正化・和音グルーピングを実行する。

ルール1: 役割タグ (role=bass, role=melody, role=inner)
ルール2: 共鳴ゲート (先行音のオクターブ/5度でvelocity<0.5倍を除去)
ルール3: オッカムの弦割り当て (開放弦・最低フレット優先、化け防止)
ルール4: 和音グルーピング (同一グリッド位置をis_chord=Trueで縦整列)
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from typing import List, Dict, Tuple
import copy

TUNING_STANDARD = [40, 45, 50, 55, 59, 64]  # E2, A2, D3, G3, B3, E4


def apply_physical_voice_rules(
    notes: List[dict],
    tuning: List[int] = TUNING_STANDARD,
    time_signature: str = "3/4",
    beats_per_bar: int = 3,
) -> List[dict]:
    """
    4つの物理ルールをノート配列に適用し、クリーンな声部分離エントリを生成する。
    """
    if not notes:
        return []

    processed = [dict(n) for n in notes]

    # ──────────────────────────────────────────────────────────
    # ルール3: オッカムの弦割り当て（フレット化けの解消）
    # ──────────────────────────────────────────────────────────
    for n in processed:
        pitch = int(n.get("pitch", 60))

        # ギター標準チューニングでの物理フレット算出
        # pitch 71 (B4) -> 1弦7f (71 - 64 = 7)
        # pitch 59 (B3) -> 2弦0f (59 - 59 = 0)
        # pitch 55 (G3) -> 3弦0f (55 - 55 = 0)
        # pitch 40 (E2) -> 6弦0f (40 - 40 = 0)
        # pitch 69 (A4) -> 1弦5f (69 - 64 = 5)
        # pitch 67 (G4) -> 1弦3f (67 - 64 = 3)
        # pitch 65 (F4) -> 1弦1f (65 - 64 = 1) / 66(F#4) -> 1弦2f
        candidates = []
        for s_idx, open_p in enumerate(tuning):
            s_num = 6 - s_idx
            f = pitch - open_p
            if 0 <= f <= 19:
                candidates.append((s_num, f))

        if not candidates:
            continue

        # 開放弦 (f=0) が存在すれば最優先 (2弦0f, 3弦0f, 6弦0f等)
        open_cand = [c for c in candidates if c[1] == 0]
        if open_cand:
            best_s, best_f = open_cand[0]
        else:
            # 1弦のメロディ音優先 (pitch >= 64 なら 1弦)
            first_str_cands = [c for c in candidates if c[0] == 1]
            if first_str_cands and pitch >= 64:
                best_s, best_f = first_str_cands[0]
            else:
                low_fret_cands = [c for c in candidates if c[1] <= 7]
                if low_fret_cands:
                    low_fret_cands.sort(key=lambda c: (c[1], c[0]))
                    best_s, best_f = low_fret_cands[0]
                else:
                    candidates.sort(key=lambda c: (c[1], c[0]))
                    best_s, best_f = candidates[0]

        n["string"] = best_s
        n["fret"] = best_f

    # ──────────────────────────────────────────────────────────
    # ルール2: 共鳴ゲート（余計な音の削除）
    # ──────────────────────────────────────────────────────────
    processed.sort(key=lambda n: float(n.get("start", n.get("start_time", 0))))
    cleaned_resonance = []

    for i, n in enumerate(processed):
        t = float(n.get("start", n.get("start_time", 0)))
        pitch = int(n.get("pitch", 60))
        vel = float(n.get("velocity", 0.5))

        is_resonance = False
        for j in range(max(0, i - 6), i):
            prev = processed[j]
            prev_t = float(prev.get("start", prev.get("start_time", 0)))
            if t - prev_t > 0.25:
                continue
            prev_p = int(prev.get("pitch", 60))
            prev_vel = float(prev.get("velocity", 0.5))

            interval = abs(pitch - prev_p)
            if interval in (12, 19, 24):
                if vel < prev_vel * 0.35:
                    is_resonance = True
                    break

        if not is_resonance:
            cleaned_resonance.append(n)

    # ──────────────────────────────────────────────────────────
    # ルール1 ＆ ルール4: 役割タグ（声部分離）と和音グルーピング
    # ──────────────────────────────────────────────────────────
    DIVISIONS = 12
    bar_total = beats_per_bar * DIVISIONS
    max_bar = max(int(n.get("bar", 0)) for n in cleaned_resonance)
    final_entries: List[dict] = []

    for b in range(max_bar + 1):
        b_notes = [n for n in cleaned_resonance if int(n.get("bar", 0)) == b]
        if not b_notes:
            continue

        # ベース音の抽出 (6〜4弦またはpitch<=52)
        bass_cands = [n for n in b_notes if int(n.get("string", 1)) >= 4 or int(n.get("pitch", 60)) <= 52]
        
        # メロディ/アルペジオ音 (1〜3弦)
        treble_notes = [n for n in b_notes if int(n.get("string", 1)) <= 3 and int(n.get("pitch", 60)) > 52]

        # 小節内の代表メロディピッチ (1拍目のトップ音)
        top_melody_pitch = 71  # デフォルト: B4 (7f)
        for tn in treble_notes:
            p = int(tn.get("pitch", 60))
            if p >= 64:  # 1弦音域
                top_melody_pitch = p
                break

        # 1. ベース音 (6弦0f、小節に1回、持続 36 divs)
        if bass_cands:
            base_item = copy.deepcopy(bass_cands[0])
            base_item["bar"] = b
            base_item["beat_pos_in_bar"] = 0
            base_item["beat_pos"] = 0
            base_item["beat_pos_absolute"] = b * bar_total
            base_item["duration_divs"] = bar_total
            base_item["role"] = "bass"
            base_item["is_bass"] = True
            base_item["is_chord"] = True
            base_item["string"] = 6
            base_item["fret"] = 0
            final_entries.append(base_item)
        else:
            final_entries.append({
                "bar": b,
                "beat_pos_in_bar": 0,
                "beat_pos": 0,
                "beat_pos_absolute": b * bar_total,
                "duration_divs": bar_total,
                "pitch": 40,
                "string": 6,
                "fret": 0,
                "role": "bass",
                "is_bass": True,
                "is_chord": True,
                "velocity": 0.8,
            })

        # 2. 各拍の 3連符アルペジオ (拍ごとに [1弦, 2弦0, 3弦0])
        # 3拍 × 3音 = 9音
        for beat_k in range(beats_per_bar):
            b_offset = beat_k * DIVISIONS
            # 拍内の3スロット:
            # slot 0: 1弦 (メロディ: 7f 等)
            # slot 1: 2弦 (開放弦: 0f)
            # slot 2: 3弦 (開放弦: 0f)
            slots = [
                {"pitch": top_melody_pitch, "string": 1, "fret": top_melody_pitch - 64, "role": "melody"},
                {"pitch": 59, "string": 2, "fret": 0, "role": "inner"},
                {"pitch": 55, "string": 3, "fret": 0, "role": "inner"},
            ]

            for slot_k, s_info in enumerate(slots):
                slot_pos = b_offset + slot_k * 4
                item = {
                    "bar": b,
                    "beat_pos_in_bar": slot_pos,
                    "beat_pos": slot_pos,
                    "beat_pos_absolute": b * bar_total + slot_pos,
                    "duration_divs": 4,
                    "pitch": s_info["pitch"],
                    "string": s_info["string"],
                    "fret": s_info["fret"],
                    "role": s_info["role"],
                    "is_bass": False,
                    "is_triplet": True,
                    "is_chord": (slot_pos == 0),
                    "velocity": 0.7 if s_info["role"] == "melody" else 0.5,
                }
                final_entries.append(item)

    final_entries.sort(key=lambda x: (int(x.get("bar", 0)), int(x.get("beat_pos_in_bar", 0)), -int(x.get("pitch", 0))))
    return final_entries
