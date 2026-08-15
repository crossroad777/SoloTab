"""
nextchord_renderer.py — NextChord SoloTab テキストフォーマット レンダラー
======================================================================
物理ルール (physical_voice_rules.py) に基づき、
role=bass, role=melody, role=inner を構造化してテキスト出力する。

romance.wav 正解仕様:
  - bass = 6弦0 (小節に1回だけ、持続)
  - 3連符 = 各拍 [1弦(メロディ), 2弦(0), 3弦(0)] × 3拍 = 9音
  - 第1〜3小節 = [1弦7, 2弦0, 3弦0] × 3拍 + ベース6弦0 = 計10音
  - 第4小節 = メロディが1弦5 (または 1弦7->5->3->2) に変化
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from typing import List, Dict, Optional
from physical_voice_rules import apply_physical_voice_rules


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
    物理ルールを適用したノートエントリから NextChord SoloTab テキストを生成する。
    """
    lines = []
    # ヘッダー
    lines.append(f"{title}")
    lines.append(f"■= {int(round(bpm))}  {time_signature} NextChord SoloTab")
    lines.append(f"{tuning_strings}")

    if not notes:
        return "\n".join(lines)

    # 物理ルールの適用
    processed = apply_physical_voice_rules(notes, time_signature=time_signature, beats_per_bar=beats_per_bar)

    # 小節ごとにノートをグループ化
    max_bar = max(int(n.get("bar", 0)) for n in processed)
    bars_notes: Dict[int, List[dict]] = {}
    for n in processed:
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
                for n in processed:
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

        bass_notes = [n for n in b_notes if n.get("role") == "bass"]
        melody_notes = [n for n in b_notes if n.get("role") != "bass"]

        # 1拍目トップノート
        if melody_notes:
            lines.append(str(melody_notes[0].get("fret", 0)))

        # 1拍目ベース音 (小節に1回だけ)
        if bass_notes:
            lines.append(str(bass_notes[0].get("fret", 0)))

        # アルペジオの後続音列（2音目〜9音目）
        if len(melody_notes) > 1:
            seq_str = " ".join(str(n.get("fret", 0)) for n in melody_notes[1:])
            lines.append(seq_str)

    return "\n".join(lines)
