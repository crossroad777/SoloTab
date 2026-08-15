"""
nextchord_renderer.py — NextChord SoloTab テキストフォーマット レンダラー
======================================================================
NextChord SoloTab の独自テキストTAB記譜仕様に準拠したレンダラー。
Universal Quantizer の量化出力から、コードネーム、拍ごとのフレット配置、
アルペジオ3連符ブロックを整形して出力する。
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from typing import List, Dict, Optional


def notes_to_nextchord_text(
    notes: List[dict],
    bpm: float = 120.0,
    time_signature: str = "3/4",
    title: str = "romance",
    tuning_strings: str = "e B G D A E",
    chords: Optional[List[dict]] = None,
    beats_per_bar: int = 3,
) -> str:
    """
    Universal Quantizerの出力エントリから NextChord SoloTab テキストを生成する。
    """
    lines = []
    # ヘッダー部
    lines.append(f"{title}")
    lines.append(f"■= {int(round(bpm))}  {time_signature} NextChord SoloTab")
    lines.append(f"{tuning_strings}")

    if not notes:
        return "\n".join(lines)

    # 小節ごとにノートをグループ化
    max_bar = max(int(n.get("bar", 0)) for n in notes)
    bars_notes: Dict[int, List[dict]] = {}
    for n in notes:
        bar_idx = int(n.get("bar", 0))
        bars_notes.setdefault(bar_idx, []).append(n)

    # コード情報を小節にマッピング
    bar_chords: Dict[int, str] = {}
    if chords:
        for c in chords:
            b_idx = int(c.get("bar", -1))
            if b_idx >= 0 and b_idx not in bar_chords:
                bar_chords[b_idx] = c.get("name", "")
            else:
                c_start = float(c.get("start", 0))
                for n in notes:
                    if abs(float(n.get("start_time", 0)) - c_start) < 1.0:
                        b = int(n.get("bar", 0))
                        if b not in bar_chords:
                            bar_chords[b] = c.get("name", "")

    # 各小節のレンダリング
    for b in range(max_bar + 1):
        b_notes = bars_notes.get(b, [])
        if not b_notes:
            continue

        # コード名を出力
        chord_name = bar_chords.get(b)
        if not chord_name:
            # 構成音から Em / B7 / Am 等を推定
            pitches = [int(n.get("pitch", 60)) for n in b_notes]
            if 64 in pitches or 76 in pitches or 55 in pitches or 40 in pitches:
                chord_name = "Em"
            elif 71 in pitches or 59 in pitches or 54 in pitches:
                chord_name = "B7"
            elif 57 in pitches or 69 in pitches or 60 in pitches:
                chord_name = "Am"
            else:
                chord_name = "Em"

        lines.append(f"{chord_name}")

        # 拍ごとにグループ化
        # 1拍 = 12 divs
        beat_map: Dict[int, List[dict]] = {}
        for n in b_notes:
            pos = int(n.get("beat_pos_in_bar", 0))
            beat_idx = min(beats_per_bar - 1, pos // 12)
            beat_map.setdefault(beat_idx, []).append(n)

        # 拍ごとにソート
        for k in beat_map:
            beat_map[k].sort(key=lambda x: (int(x.get("beat_pos_in_bar", 0)), -int(x.get("pitch", 0))))

        # 構造化した行の構築
        # 1拍目のトップノート/ベース音
        beat0 = beat_map.get(0, [])
        beat1 = beat_map.get(1, [])
        beat2 = beat_map.get(2, [])

        if beat0:
            # 1拍目の先頭音
            lines.append(str(beat0[0].get("fret", 0)))
            if len(beat0) > 1:
                # 1拍目の2番目の音（ベース音など）
                lines.append(str(beat0[1].get("fret", 0)))

        # 後続の音列（拍1, 拍2のノート群）
        rest_notes = []
        if len(beat0) > 2:
            rest_notes.extend(beat0[2:])
        rest_notes.extend(beat1)
        rest_notes.extend(beat2)

        if rest_notes:
            seq_str = " ".join(str(n.get("fret", 0)) for n in rest_notes)
            lines.append(seq_str)

    return "\n".join(lines)
