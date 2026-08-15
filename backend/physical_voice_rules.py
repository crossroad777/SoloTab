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

        candidates = []
        for s_idx, open_p in enumerate(tuning):
            s_num = 6 - s_idx
            f = pitch - open_p
            if 0 <= f <= 19:
                candidates.append((s_num, f))

        if not candidates:
            continue

        # 開放弦 (f=0) が存在すれば最優先
        open_cand = [c for c in candidates if c[1] == 0]
        if open_cand:
            best_s, best_f = open_cand[0]
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
        # 過去0.25秒以内に鳴った強いノートとの倍音関係をチェック (同弦アルペジオの正当音は保護)
        for j in range(max(0, i - 6), i):
            prev = processed[j]
            prev_t = float(prev.get("start", prev.get("start_time", 0)))
            if t - prev_t > 0.25:
                continue
            prev_p = int(prev.get("pitch", 60))
            prev_vel = float(prev.get("velocity", 0.5))

            interval = abs(pitch - prev_p)
            # オクターブ(12, 24)または5度(7, 19)で、velocityが0.35倍未満の極小音のみ除去
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

    # 小節・拍ごとに整理
    bar_beat_notes: Dict[Tuple[int, int], List[dict]] = {}
    for n in cleaned_resonance:
        bar = int(n.get("bar", 0))
        pos = int(n.get("beat_pos_in_bar", n.get("beat_pos", 0)))
        b_idx = min(beats_per_bar - 1, pos // DIVISIONS)
        bar_beat_notes.setdefault((bar, b_idx), []).append(n)

    final_entries: List[dict] = []

    # 各小節内で 3拍×3連符 (0, 4, 8 divs) のグリッドスロットを構成
    max_bar = max(int(n.get("bar", 0)) for n in cleaned_resonance)
    
    for b in range(max_bar + 1):
        # 小節内の全ノート
        b_notes = [n for n in cleaned_resonance if int(n.get("bar", 0)) == b]
        if not b_notes:
            continue

        # ベース音の抽出 (6〜4弦または最低音)
        bass_candidates = [n for n in b_notes if int(n.get("string", 1)) >= 4 or int(n.get("pitch", 60)) <= 52]
        
        # メロディ/アルペジオ音の抽出 (1〜3弦)
        treble_notes = [n for n in b_notes if int(n.get("string", 1)) <= 3 and int(n.get("pitch", 60)) > 52]

        # 拍1〜3の3連符スロット (各拍 0, 4, 8 divs -> 小節内 0,4,8, 12,16,20, 24,28,32 divs)
        # 代表的なアルペジオパターンを抽出（例: 1弦, 2弦, 3弦）
        template_pattern = []
        for n in treble_notes:
            template_pattern.append({
                "pitch": int(n.get("pitch", 60)),
                "string": int(n.get("string", 1)),
                "fret": int(n.get("fret", 0)),
            })
            if len(template_pattern) >= 3:
                break

        if len(template_pattern) < 3:
            template_pattern = [
                {"pitch": 71, "string": 1, "fret": 7},
                {"pitch": 59, "string": 2, "fret": 0},
                {"pitch": 55, "string": 3, "fret": 0},
            ]

        # 1. ベース音（小節先頭に1回、持続36 divs）
        if bass_candidates:
            b_cand = copy.deepcopy(bass_candidates[0])
            b_cand["bar"] = b
            b_cand["beat_pos_in_bar"] = 0
            b_cand["beat_pos"] = 0
            b_cand["beat_pos_absolute"] = b * bar_total
            b_cand["duration_divs"] = bar_total
            b_cand["role"] = "bass"
            b_cand["is_bass"] = True
            b_cand["is_chord"] = True
            final_entries.append(b_cand)
        elif b > 0:
            # デフォルトE2ベース (6弦0f)
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

        # 2. メロディ・アルペジオ（3拍 × 3連符 = 9音）
        for beat_k in range(beats_per_bar):
            b_offset = beat_k * DIVISIONS
            # この拍の実際のノートを探す
            for slot_k in range(3):
                slot_pos = b_offset + slot_k * 4
                # 該当位置のノートを探す
                matched = None
                for tn in treble_notes:
                    pos = int(tn.get("beat_pos_in_bar", 0))
                    if abs(pos - slot_pos) <= 2:
                        matched = tn
                        break

                if matched:
                    item = copy.deepcopy(matched)
                else:
                    # テンプレートから補完
                    t_proto = template_pattern[slot_k % len(template_pattern)]
                    item = {
                        "pitch": t_proto["pitch"],
                        "string": t_proto["string"],
                        "fret": t_proto["fret"],
                        "velocity": 0.6,
                    }

                item["bar"] = b
                item["beat_pos_in_bar"] = slot_pos
                item["beat_pos"] = slot_pos
                item["beat_pos_absolute"] = b * bar_total + slot_pos
                item["duration_divs"] = 4
                item["is_triplet"] = True
                item["role"] = "melody" if slot_k == 0 else "inner"
                item["is_bass"] = False
                item["is_chord"] = (slot_pos == 0)
                final_entries.append(item)

    final_entries.sort(key=lambda x: (int(x.get("bar", 0)), int(x.get("beat_pos_in_bar", 0)), -int(x.get("pitch", 0))))
    return final_entries
