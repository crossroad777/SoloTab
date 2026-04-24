"""
musical_filter.py — 音楽理論ベースのインテリジェントフィルタ
==============================================================
検出されたキー・コード進行を活用し、音楽的に不自然なノートを除去・補正する。

フィルタ:
  1. キースケール整合性: 検出キーのスケール外ノートにペナルティ
  2. コード整合性: 各拍のコードと矛盾するノートを除去
  3. 繰り返し構造検出: 類似フレーズの検出と一貫性向上
  4. アルペジオ整形: ベース→高音の時間順パターンを検出し、間引きすぎを防止
"""

import logging
from bisect import bisect_right
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# --- スケール定義 ---

# 各キーのスケール構成音 (半音数オフセット)
MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]           # Ionian
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]           # Natural minor
HARMONIC_MINOR = [0, 2, 3, 5, 7, 8, 11]        # Harmonic minor
MELODIC_MINOR_UP = [0, 2, 3, 5, 7, 9, 11]      # Melodic minor (ascending)
PENTATONIC_MAJOR = [0, 2, 4, 7, 9]             # Major pentatonic
PENTATONIC_MINOR = [0, 3, 5, 7, 10]            # Minor pentatonic
BLUES_SCALE = [0, 3, 5, 6, 7, 10]             # Blues scale

NOTE_NAME_TO_PC = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8,
    'A': 9, 'A#': 10, 'Bb': 10, 'B': 11,
}


def _key_to_scale_pcs(key: str) -> set:
    """
    キー名 (例: "Am", "G") からスケール構成音のピッチクラスセットを生成。
    マイナーキーはナチュラル・ハーモニック・メロディックの和集合を使用し、
    ペンタトニック/ブルースも合わせて広めに許容する。
    """
    if not key:
        return set(range(12))  # キー不明→全音許可

    is_minor = key.endswith('m')
    root_name = key[:-1] if is_minor else key
    root_pc = NOTE_NAME_TO_PC.get(root_name, 0)

    if is_minor:
        # マイナーキー: 複数スケールの和集合
        intervals = set(MINOR_SCALE) | set(HARMONIC_MINOR) | set(MELODIC_MINOR_UP) | set(PENTATONIC_MINOR) | set(BLUES_SCALE)
    else:
        # メジャーキー: メジャースケール + ペンタトニック
        intervals = set(MAJOR_SCALE) | set(PENTATONIC_MAJOR)
        # ブルーノート (♭3, ♭7) も許容
        intervals.add(3)   # ♭3
        intervals.add(10)  # ♭7

    return {(root_pc + i) % 12 for i in intervals}


def _chord_to_pcs(chord: str) -> set:
    """
    コード名 (例: "Am", "G", "E7") から構成音のピッチクラスセットを生成。
    """
    if not chord or chord == "N.C.":
        return set(range(12))

    # 7thコード
    if chord.endswith('7'):
        root_name = chord[:-1]
        root_pc = NOTE_NAME_TO_PC.get(root_name, 0)
        return {root_pc, (root_pc + 4) % 12, (root_pc + 7) % 12, (root_pc + 10) % 12}

    # マイナーコード
    if chord.endswith('m'):
        root_name = chord[:-1]
        root_pc = NOTE_NAME_TO_PC.get(root_name, 0)
        return {root_pc, (root_pc + 3) % 12, (root_pc + 7) % 12}

    # メジャーコード
    root_pc = NOTE_NAME_TO_PC.get(chord, 0)
    return {root_pc, (root_pc + 4) % 12, (root_pc + 7) % 12}


def filter_by_key(notes: List[Dict], key: str,
                  confidence: float = 0.5,
                  velocity_penalty: float = 0.5) -> List[Dict]:
    """
    キースケール整合性フィルタ。

    ★完全に除去するのではなく、スケール外ノートのvelocityにペナルティを与える。
    これにより後段のvelocityフィルタで自然に低優先度として扱われる。

    経過音（passing tone）やクロマティックアプローチは音楽的に自然なので、
    前後のノートがスケール内ならペナルティを軽減する。

    Args:
        notes: ノートリスト
        key: 検出されたキー (例: "Am")
        confidence: キー検出の確信度 (低い場合はペナルティを軽減)
        velocity_penalty: スケール外ノートに掛けるvelocity係数
    """
    if not notes or not key:
        return notes

    scale_pcs = _key_to_scale_pcs(key)

    # キー確信度が低い場合はペナルティを軽減
    effective_penalty = velocity_penalty + (1.0 - velocity_penalty) * (1.0 - confidence)

    penalized = 0
    sorted_notes = sorted(notes, key=lambda n: n["start"])

    for i, note in enumerate(sorted_notes):
        pitch_class = note.get("pitch", 60) % 12
        if pitch_class in scale_pcs:
            continue

        # 経過音チェック: 前後のノートがスケール内なら軽減
        is_passing = False
        if i > 0 and i < len(sorted_notes) - 1:
            prev_pc = sorted_notes[i - 1].get("pitch", 60) % 12
            next_pc = sorted_notes[i + 1].get("pitch", 60) % 12
            # 前後がスケール内で、時間的に近ければ経過音
            time_gap_prev = note["start"] - sorted_notes[i - 1]["start"]
            time_gap_next = sorted_notes[i + 1]["start"] - note["start"]
            if (prev_pc in scale_pcs and next_pc in scale_pcs and
                    time_gap_prev < 0.3 and time_gap_next < 0.3):
                is_passing = True

        if is_passing:
            # 経過音: 軽いペナルティのみ
            note["velocity"] = note.get("velocity", 0.5) * (effective_penalty * 1.3)
        else:
            # 非経過音のスケール外: フルペナルティ
            note["velocity"] = note.get("velocity", 0.5) * effective_penalty
            penalized += 1

    if penalized:
        logger.info(f"[key_filter] Penalized {penalized} out-of-scale notes (key={key})")

    return sorted_notes


def filter_by_chords(notes: List[Dict], chords: List[Dict],
                     strong_beat_only: bool = True,
                     min_chord_confidence: float = 0.5) -> List[Dict]:
    """
    コード整合性フィルタ。

    各ノートが発音する時点のコードの構成音と照合。
    コード構成音に含まれないノートのvelocityにペナルティを与える。

    ★強拍（ビートの頭）のノートのみ厳格に適用し、
    弱拍やメロディのノートには軽いペナルティに留める。

    Args:
        notes: ノートリスト
        chords: コード進行リスト [{start, end, chord, confidence}]
        strong_beat_only: True=強拍のみ厳格適用
        min_chord_confidence: この確信度未満のコードは無視
    """
    if not notes or not chords:
        return notes

    # コード区間のタイムラインを構築
    chord_starts = [c["start"] for c in chords]
    penalized = 0

    for note in notes:
        t = note["start"]

        # このノートが属するコード区間を探す
        idx = bisect_right(chord_starts, t) - 1
        if idx < 0 or idx >= len(chords):
            continue

        chord = chords[idx]
        if t > chord["end"]:
            continue  # コード区間外

        # 低確信度のコードはスキップ
        if chord.get("confidence", 0) < min_chord_confidence:
            continue

        chord_pcs = _chord_to_pcs(chord["chord"])
        pitch_class = note.get("pitch", 60) % 12

        if pitch_class not in chord_pcs:
            # コード外ノート → 軽いペナルティ (完全除去はしない)
            vel = note.get("velocity", 0.5)
            note["velocity"] = vel * 0.75
            penalized += 1

    if penalized:
        logger.info(f"[chord_filter] Penalized {penalized} chord-inconsistent notes")

    return notes


def detect_repeated_phrases(notes: List[Dict],
                            min_phrase_len: int = 4,
                            max_phrase_len: int = 16,
                            pitch_tolerance: int = 0,
                            time_ratio_tolerance: float = 0.3) -> List[Dict]:
    """
    繰り返しフレーズの検出。

    同じピッチシーケンスが曲中で繰り返される場合、
    最も信頼度（velocity合計）が高いインスタンスを「正解」とみなし、
    他のインスタンスの欠損ノートを補完する。

    これはギター曲の「AメロBメロサビ」等の繰り返し構造を活用して
    転写精度を向上させるアプローチ。

    Args:
        notes: 時間順ソート済みノートリスト
        min_phrase_len: 最小フレーズ長（ノート数）
        max_phrase_len: 最大フレーズ長
        pitch_tolerance: ピッチの許容誤差（半音）
        time_ratio_tolerance: 時間比率の許容誤差
    """
    if not notes or len(notes) < min_phrase_len * 2:
        return notes

    sorted_notes = sorted(notes, key=lambda n: n["start"])

    # ピッチシーケンスに変換
    pitches = [n.get("pitch", 60) for n in sorted_notes]

    # フレーズ候補の検出（最頻出パターン）
    found_phrases = []
    best_phrase_len = 0

    for phrase_len in range(min_phrase_len, min(max_phrase_len, len(pitches) // 2) + 1):
        # 全てのフレーズ開始位置でピッチパターンを抽出
        patterns = {}
        for i in range(len(pitches) - phrase_len + 1):
            pattern = tuple(pitches[i:i + phrase_len])
            if pattern not in patterns:
                patterns[pattern] = []
            patterns[pattern].append(i)

        # 2回以上出現するパターンを記録
        for pattern, positions in patterns.items():
            if len(positions) >= 2:
                # 各出現のvelocity合計
                best_pos = max(positions, key=lambda p:
                    sum(sorted_notes[p + j].get("velocity", 0.5) for j in range(phrase_len)))
                found_phrases.append({
                    "pattern": pattern,
                    "positions": positions,
                    "best_pos": best_pos,
                    "length": phrase_len,
                    "count": len(positions),
                })
                if phrase_len > best_phrase_len:
                    best_phrase_len = phrase_len

    if not found_phrases:
        return notes

    # 最も長いパターンのみ使用
    long_phrases = [p for p in found_phrases if p["length"] >= best_phrase_len * 0.8]

    # 繰り返しフレーズのvelocityを正解インスタンスに近づける
    boosted = 0
    for phrase in long_phrases:
        best_pos = phrase["best_pos"]
        for pos in phrase["positions"]:
            if pos == best_pos:
                continue
            for j in range(phrase["length"]):
                src_note = sorted_notes[best_pos + j]
                tgt_note = sorted_notes[pos + j]
                # 正解のvelocityに近づける（完全コピーではなく平均寄り）
                src_vel = src_note.get("velocity", 0.5)
                tgt_vel = tgt_note.get("velocity", 0.5)
                if src_vel > tgt_vel:
                    tgt_note["velocity"] = (src_vel + tgt_vel) / 2
                    boosted += 1

    if boosted:
        logger.info(f"[repeat_detect] Boosted {boosted} notes in "
                    f"{len(long_phrases)} repeated phrases")

    return sorted_notes


def detect_arpeggio_patterns(notes: List[Dict], beats: List[float],
                             bpm: float = 120.0) -> List[Dict]:
    """
    アルペジオパターンの検出とマーキング。

    1拍内で異なる弦を順に弾くパターンを検出し、
    "arpeggio" テクニックとしてマーク + velocity保護。

    アルペジオは個々のノートが弱くても意図的に弾かれた音なので、
    velocityフィルタで除去されないよう最低velocityを保証する。

    Args:
        notes: ノートリスト
        beats: ビート位置
        bpm: BPM
    """
    if not notes or not beats or len(beats) < 2:
        return notes

    beat_dur = 60.0 / max(bpm, 40)
    arpeggio_window = beat_dur * 1.2  # 1拍+余裕
    min_arpeggio_notes = 3  # 最低3音

    sorted_notes = sorted(notes, key=lambda n: n["start"])
    arpeggio_count = 0

    i = 0
    while i < len(sorted_notes):
        # このノートから始まるアルペジオ候補を探索
        group = [sorted_notes[i]]
        j = i + 1
        while (j < len(sorted_notes) and
               sorted_notes[j]["start"] - sorted_notes[i]["start"] < arpeggio_window):
            group.append(sorted_notes[j])
            j += 1

        if len(group) >= min_arpeggio_notes:
            # 異なる弦を使っているか確認
            strings = set(n.get("string", 0) for n in group if n.get("string", 0) > 0)
            pitches = [n.get("pitch", 0) for n in group]

            # ピッチが単調増加or単調減少（アルペジオの特徴）
            is_ascending = all(pitches[k] <= pitches[k + 1] for k in range(len(pitches) - 1))
            is_descending = all(pitches[k] >= pitches[k + 1] for k in range(len(pitches) - 1))

            if len(strings) >= 3 and (is_ascending or is_descending):
                # アルペジオ検出 → velocity保護
                for n in group:
                    if n.get("velocity", 0) < 0.35:
                        n["velocity"] = 0.35  # 最低velocity保証
                    if n.get("technique") is None:
                        n["technique"] = "arpeggio"
                arpeggio_count += 1

        i = max(i + 1, j - len(group) + 1)  # 次の探索位置

    if arpeggio_count:
        logger.info(f"[arpeggio_detect] Found {arpeggio_count} arpeggio patterns, "
                    f"protected notes from velocity filter")

    return sorted_notes


def apply_musical_filters(notes: List[Dict],
                          key: Optional[str] = None,
                          key_confidence: float = 0.5,
                          chords: Optional[List[Dict]] = None,
                          beats: Optional[List[float]] = None,
                          bpm: float = 120.0) -> Dict:
    """
    全ての音楽理論フィルタを順に適用する。

    適用順序:
    1. アルペジオ検出 (velocity保護、先に適用)
    2. キースケールフィルタ (velocityペナルティ)
    3. コード整合性フィルタ (velocityペナルティ)
    4. 繰り返し構造検出 (velocity補正)

    Returns:
        dict: {"notes": filtered_notes, "stats": {...}}
    """
    stats = {"original": len(notes)}

    # 1. アルペジオ検出 & velocity保護
    notes = detect_arpeggio_patterns(notes, beats, bpm)

    # 2. キースケール整合性
    if key and key_confidence > 0.3:
        notes = filter_by_key(notes, key, confidence=key_confidence)
        stats["key_filter"] = key

    # 3. コード整合性
    if chords:
        notes = filter_by_chords(notes, chords)
        stats["chord_filter"] = len(chords)

    # 4. 繰り返し構造検出
    notes = detect_repeated_phrases(notes)

    stats["filtered"] = len(notes)
    logger.info(f"[musical_filter] Applied musical filters: {stats}")

    return {"notes": notes, "stats": stats}
