"""
nextchord_renderer.py — NextChord SoloTab テキストフォーマット レンダラー
======================================================================
NextChord SoloTab 独自記譜仕様レンダラー（文字列結合バグ完全抹殺版）。

フォーマット仕様:
  - 行1: 曲名 (romance)
  - 行2: ■= [BPM]  [拍子] NextChord SoloTab
  - 行3: e B G D A E
  - 小節ブロック:
      - [コード名] (小節先頭に1行: Em, B7, Am等)
      - [1拍目トップノート] (例: 7)
      - [1拍目ベース音] (例: 0)
      - [アルペジオ音列] (例: 0 0 7 0 0 7 0 0)
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from typing import List, Dict, Optional
from physical_voice_rules import apply_physical_voice_rules


def _get_chord_for_bar(bar_idx: int, chords: Optional[List[dict]], beats_per_bar: int = 3, bpm: float = 120.0) -> str:
    """小節番号に対応するコードネームを取得（デフォルトはEm）。"""
    if not chords:
        return "Em"
    # 小節開始時刻
    sec_per_bar = (60.0 / max(30.0, bpm)) * beats_per_bar
    bar_time = bar_idx * sec_per_bar
    for c in chords:
        c_start = float(c.get("start", c.get("time", 0.0)))
        c_end = float(c.get("end", c_start + 4.0))
        if c_start <= bar_time <= c_end:
            chord_name = str(c.get("chord", "Em"))
            return chord_name if chord_name and chord_name != "N.C." else "Em"
    return "Em"


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
    全要素を確実にスペース区切りで出力し、数字結合を完全に防止する。
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

        # 1. コードネーム（小節先頭に1回だけ出力）
        chord_name = _get_chord_for_bar(b, chords, beats_per_bar=beats_per_bar, bpm=bpm)
        lines.append(f"{chord_name}")

        bass_notes = [n for n in b_notes if n.get("role") == "bass"]
        melody_notes = [n for n in b_notes if n.get("role") != "bass"]

        # 2. 1拍目トップノート (0〜24 の単一数字)
        top_fret = int(melody_notes[0].get("fret", 7)) if melody_notes else 7
        lines.append(f"{top_fret}")

        # 3. 1拍目ベース音 (0〜24 の単一数字)
        bass_fret = int(bass_notes[0].get("fret", 0)) if bass_notes else 0
        lines.append(f"{bass_fret}")

        # 4. アルペジオ音列（2音目以降）
        # 全数字を確実に半角スペースで区切ってシリアライズ
        if len(melody_notes) > 1:
            seq_frets = [str(int(n.get("fret", 0))) for n in melody_notes[1:]]
            seq_str = " ".join(seq_frets)
            lines.append(seq_str)

    return "\n".join(lines)

