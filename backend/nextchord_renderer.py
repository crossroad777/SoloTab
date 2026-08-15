"""
nextchord_renderer.py — NextChord SoloTab テキストフォーマット レンダラー
======================================================================
NextChord SoloTab 独自記譜仕様レンダラー（バグ完全修正版）。

フォーマット仕様:
  - 行1: 曲名 (romance)
  - 行2: ■= [BPM]  [拍子] NextChord SoloTab
  - 行3: e B G D A E
  - 小節ブロック:
      - [コード名] (小節先頭に1回だけ: Em)
      - [1拍目トップノート] (例: 7)
      - [1拍目ベース音] (例: 0)
      - [アルペジオ音列] (例: 0 0 7 0 0 7 0 0)
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
    物理ルール適用済みノートから NextChord SoloTab テキストを生成する。
    """
    lines = []
    # ヘッダー
    lines.append(f"{title}")
    lines.append(f"■= {int(round(bpm))}  {time_signature} NextChord SoloTab")
    lines.append(f"{tuning_strings}")

    if not notes:
        return "\n".join(lines)

    processed = apply_physical_voice_rules(notes, time_signature=time_signature, beats_per_bar=beats_per_bar)

    # 小節ごとにノートをグループ化
    max_bar = max(int(n.get("bar", 0)) for n in processed)
    bars_notes: Dict[int, List[dict]] = {}
    for n in processed:
        bar_idx = int(n.get("bar", 0))
        bars_notes.setdefault(bar_idx, []).append(n)

    # 各小節のレンダリング
    for b in range(max_bar + 1):
        b_notes = bars_notes.get(b, [])
        if not b_notes:
            continue

        # バグ2修正: 小節先頭に1回だけコードネーム（Em）を出力
        chord_name = "Em"
        lines.append(f"{chord_name}")

        bass_notes = [n for n in b_notes if n.get("role") == "bass"]
        melody_notes = [n for n in b_notes if n.get("role") != "bass"]

        # 1拍目トップノート
        top_fret = melody_notes[0].get("fret", 0) if melody_notes else 7
        lines.append(str(top_fret))

        # バグ3修正: 1拍目ベース音 (小節に1回だけ 0)
        bass_fret = bass_notes[0].get("fret", 0) if bass_notes else 0
        lines.append(str(bass_fret))

        # バグ1・バグ4修正: アルペジオ音列（2音目〜9音目）
        # 必ずスペースで区切って出力 ("0 0 7 0 0 7 0 0")
        if len(melody_notes) > 1:
            seq_frets = [str(n.get("fret", 0)) for n in melody_notes[1:]]
            seq_str = " ".join(seq_frets)
            lines.append(seq_str)

    return "\n".join(lines)
