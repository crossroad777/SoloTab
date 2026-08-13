"""
music_quantizer.py — music21ベースの音楽的量子化エンジン
======================================================
生のノートデータ(onset秒, pitch, duration秒)を音楽理論に基づいて
正確な楽譜表記に変換する。

従来のアドホックな量子化を廃止し、music21の検証済みアルゴリズムを使用。

Key concepts:
- quarterLength: 四分音符 = 1.0 の相対的な音価単位
- divisions: GP5/MusicXMLで使用する整数分割 (12 = LCM of 4 and 3)
- onset_ql: 拍頭からの quarterLength オフセット
"""

import numpy as np
from typing import List, Tuple, Optional
from music21 import stream, note, chord, meter, tempo, duration


# GP5互換のdivisions (12 per quarter note)
DIVISIONS = 12


def quantize_notes_music21(
    notes: List[dict],
    beats: List[float],
    bpm: float,
    time_signature: str = "4/4",
    beats_per_bar: int = 4,
    rhythm_subdivision: str = "straight",
) -> List[dict]:
    """
    music21を使ってノートデータを音楽的に量子化する。
    
    Parameters
    ----------
    notes : list[dict]
        生ノートデータ。keys: start, end, pitch, string, fret, velocity, technique, ...
    beats : list[float]
        ビート位置（秒）
    bpm : float
        テンポ
    time_signature : str
        拍子 ("3/4", "4/4", "6/8")
    beats_per_bar : int
        1小節あたりの拍数
    rhythm_subdivision : str
        "straight", "triplet", or "mixed"
    
    Returns
    -------
    list[dict]
        量子化済みエントリ。keys: bar, beat_pos, beat_pos_in_bar, beat_pos_absolute,
        duration_divs, pitch, string, fret, technique, velocity, start_time, ...
    """
    if not notes or not beats:
        return []
    
    sec_per_beat = 60.0 / bpm if bpm > 0 else 0.5
    beats_arr = np.array(beats)
    
    # --- Step 1: 秒 → quarterLength変換 ---
    # beats[0]を基準として、全ノートのオフセットを計算
    beat_origin = beats[0] if beats else 0.0
    
    # 量子化グリッドの選択（グローバルデフォルト）
    if rhythm_subdivision == "triplet":
        default_divisors = (3,)      # 3連符グリッドのみ
    elif rhythm_subdivision == "mixed":
        default_divisors = (4, 3)    # 16分 + 3連
    else:
        default_divisors = (4,)      # 16分音符グリッドのみ (straight)
    divisors = default_divisors
    
    # --- Step 2: music21 Streamを構築 ---
    part = stream.Part()
    part.insert(0, meter.TimeSignature(time_signature))
    part.insert(0, tempo.MetronomeMark(number=bpm))
    
    # ノートをmusic21に投入
    note_metadata = []  # 元のノート情報を保持
    
    # Hybrid IOI duration:
    # アルペジオ内(弦間10-20ms) → 同弦IOI（正しい和音持続時間）
    # 単音パッセージ → 全弦IOI（音が被らない）
    sorted_notes = sorted(notes, key=lambda x: float(x["start"]))
    MAX_DUR_BEATS = 4.0   # 全音符が最大
    MIN_DUR_BEATS = 0.25   # 16分音符が最小（32nd/64th排除）
    # BPM連動アルペジオ窓: テンポに合わせて自動調整
    CHORD_WINDOW = min(0.050, sec_per_beat / 8)  # 最大50ms, 最小=8分音符の半分
    
    for note_idx, n in enumerate(sorted_notes):
        onset_sec = float(n["start"])
        my_string = int(n.get("string", 0))
        
        # --- 直近ノートとの距離（アルペジオ検出用）---
        raw_next_ioi = None
        for k in range(note_idx + 1, len(sorted_notes)):
            next_t = float(sorted_notes[k]["start"])
            if next_t > onset_sec + 0.005:
                raw_next_ioi = next_t - onset_sec
                break
        
        is_in_arpeggio = raw_next_ioi is not None and raw_next_ioi < CHORD_WINDOW
        # 逆方向: 前ノートがアルペジオ内なら、自分もアルペジオグループの最後の音
        if not is_in_arpeggio and note_idx > 0:
            prev_onset = float(sorted_notes[note_idx - 1]["start"])
            if onset_sec - prev_onset < CHORD_WINDOW:
                is_in_arpeggio = True
        
        # --- Same-string IOI ---
        same_str_ioi = None
        if my_string > 0:
            for k in range(note_idx + 1, len(sorted_notes)):
                other = sorted_notes[k]
                t_diff = float(other["start"]) - onset_sec
                if t_diff < CHORD_WINDOW:
                    continue
                if int(other.get("string", 0)) == my_string:
                    same_str_ioi = t_diff
                    break
                if t_diff > MAX_DUR_BEATS * sec_per_beat:
                    break
        
        # --- All-string IOI ---
        all_str_ioi = None
        for k in range(note_idx + 1, len(sorted_notes)):
            next_t = float(sorted_notes[k]["start"])
            if next_t > onset_sec + CHORD_WINDOW:
                all_str_ioi = next_t - onset_sec
                break
        
        # --- Hybrid duration決定 ---
        if is_in_arpeggio and same_str_ioi is not None:
            dur_sec = same_str_ioi
        elif all_str_ioi is not None:
            dur_sec = all_str_ioi
        elif same_str_ioi is not None:
            dur_sec = same_str_ioi
        else:
            dur_sec = sec_per_beat

        # ★ 追加: 同時発音（和音）のベース音は長めに持続させる ★
        # ピッチが低く（MIDI 55以下）、かつ他の音と同時発音の場合、拍単位まで延長
        if int(n.get("pitch", 60)) <= 55 and is_in_arpeggio:
            # 最低でも1拍（sec_per_beat）は持続させる
            dur_sec = max(dur_sec, sec_per_beat)
            
        dur_sec = min(dur_sec, MAX_DUR_BEATS * sec_per_beat)
        dur_sec = max(dur_sec, MIN_DUR_BEATS * sec_per_beat)
        
        # ビート位置から quarterLength に変換
        # 最も近いビートを見つけて正確なオフセットを計算
        beat_idx = int(np.searchsorted(beats_arr, onset_sec, side='right')) - 1
        beat_idx = max(0, min(beat_idx, len(beats_arr) - 1))
        
        beat_time = float(beats_arr[beat_idx])
        if beat_idx + 1 < len(beats_arr):
            local_beat_dur = float(beats_arr[beat_idx + 1]) - beat_time
        else:
            local_beat_dur = sec_per_beat
        local_beat_dur = max(local_beat_dur, 0.1)  # ゼロ除算防止
        onset_ql = beat_idx + (onset_sec - beat_time) / local_beat_dur
        onset_ql = max(0.0, onset_ql)
        
        dur_ql = max(dur_sec / sec_per_beat, MIN_DUR_BEATS)
        
        m21_note = note.Note(int(n["pitch"]))
        m21_note.duration.quarterLength = dur_ql
        part.insert(onset_ql, m21_note)
        
        note_metadata.append({
            "original": n,
            "onset_ql": onset_ql,
            "m21_note": m21_note,
        })
    
    # --- Step 3: 小節レベル3連/ストレート自動検出 ---
    # 各小節のIOI分布から3連符比率を推定し、divisorsを動的に選択
    bar_divisors = {}  # bar_idx -> divisors
    if rhythm_subdivision in ("mixed", "straight"):
        bar_total_ql = float(beats_per_bar)
        triplet_grid = sec_per_beat / 3  # 3連8分の間隔
        straight_grid = sec_per_beat / 4  # 16分の間隔
        
        bar_notes = {}  # bar_idx -> [onset_sec, ...]
        for meta in note_metadata:
            onset_ql = meta["onset_ql"]
            bar_idx = int(onset_ql / bar_total_ql)
            bar_notes.setdefault(bar_idx, []).append(float(meta["original"]["start"]))
        
        for bar_idx, onsets in bar_notes.items():
            if len(onsets) < 3:
                continue
            onsets_sorted = sorted(onsets)
            iois = [onsets_sorted[i+1] - onsets_sorted[i] for i in range(len(onsets_sorted)-1)]
            iois = [x for x in iois if x > 0.02]  # 20ms以上のみ
            if not iois:
                continue
            # 3連グリッドへの近さ vs 16分グリッドへの近さ
            trip_err = sum(min(abs(x % triplet_grid), abs(x % triplet_grid - triplet_grid)) for x in iois) / len(iois)
            str_err = sum(min(abs(x % straight_grid), abs(x % straight_grid - straight_grid)) for x in iois) / len(iois)
            if trip_err < str_err * 0.8:  # 3連が明らかに良い
                bar_divisors[bar_idx] = (3,)
    
    # --- Step 4: music21量子化 ---
    part.quantize(
        quarterLengthDivisors=divisors,
        processOffsets=True,
        processDurations=True,
        inPlace=True,
    )
    
    # --- Step 5: 量子化結果をGP5互換フォーマットに変換 ---
    entries = []
    
    for meta in note_metadata:
        m21_n = meta["m21_note"]
        orig = meta["original"]
        
        # 量子化後のオフセット（quarterLength単位）
        quantized_ql = float(m21_n.offset) if hasattr(m21_n, 'offset') else meta["onset_ql"]
        quantized_dur_ql = float(m21_n.duration.quarterLength)
        
        # 3-tuplet duration handling:
        # music21 may produce tuplet quarterLengths like 1/3, 2/3 etc.
        # Ensure these map to correct division values on our grid (DIVISIONS=12).
        # Triplet 8th = 1/3 ql → 4 divs, Triplet quarter = 2/3 ql → 8 divs.
        # Standard rounding with DIVISIONS=12 handles this correctly since
        # 12 * (1/3) = 4.0 and 12 * (2/3) = 8.0 exactly.
        is_triplet = (m21_n.duration.tuplets is not None
                      and len(m21_n.duration.tuplets) > 0)
        
        # quarterLength → divisions変換
        # 1 quarter note = DIVISIONS (12) divs
        raw_divs = quantized_dur_ql * DIVISIONS
        if is_triplet:
            # Snap to nearest integer — triplet values 4, 8 are exact on 12-grid
            dur_divs = max(1, int(round(raw_divs)))
        else:
            dur_divs = max(1, int(round(raw_divs)))
        # 32nd/64th排除: 最小3divs (16分音符) にクランプ
        dur_divs = max(3, dur_divs)
        beat_pos_total = int(round(quantized_ql * DIVISIONS))
        
        # bar と beat_pos_in_bar を計算
        bar_total_divs = beats_per_bar * DIVISIONS
        bar = beat_pos_total // bar_total_divs
        beat_pos_in_bar = beat_pos_total % bar_total_divs
        
        entries.append({
            "bar": bar,
            "beat_pos": beat_pos_in_bar,
            "beat_pos_in_bar": beat_pos_in_bar,
            "beat_pos_absolute": beat_pos_total,
            "duration_divs": dur_divs,
            "pitch": orig["pitch"],
            "string": orig.get("string", 1) if orig.get("string", 1) >= 1 else 6,
            "fret": orig.get("fret", 0),
            "technique": orig.get("technique"),
            "velocity": orig.get("velocity", 0.5),
            "finger": orig.get("finger"),
            "left_hand_finger": orig.get("left_hand_finger"),
            "start_time": float(orig["start"]),
            "is_dotted": m21_n.duration.dots > 0,
            "is_triplet": m21_n.duration.tuplets is not None and len(m21_n.duration.tuplets) > 0,
        })
    
    # --- Step 5: 後処理 ---
    entries.sort(key=lambda x: x["beat_pos_absolute"])
    
    # 同一弦の重複防止
    _cap_durations_by_string(entries, beats_per_bar)
    
    # 拍可視性ルール (Gould "Behind Bars")
    # 弱拍から始まるノートが強拍を跨ぐ場合、タイで分割
    bar_total = beats_per_bar * DIVISIONS
    new_entries = []
    for e in entries:
        pos = e["beat_pos_in_bar"]
        dur = e["duration_divs"]
        
        # 小節内に収まるならそのまま
        if pos + dur <= bar_total:
            new_entries.append(e)
            continue
            
        # 小節をまたぐ: タイで分割（複数小節対応）
        cur_bar = e["bar"]
        cur_pos = pos
        remaining = dur
        first = True
        
        while remaining > 0:
            space = bar_total - cur_pos
            seg = min(remaining, space)
            seg = max(1, seg)
            
            part = dict(e)
            part["bar"] = cur_bar
            part["beat_pos_in_bar"] = cur_pos
            part["beat_pos"] = cur_pos
            part["beat_pos_absolute"] = cur_bar * bar_total + cur_pos
            part["duration_divs"] = seg
            
            if first and remaining > seg:
                part["_tie_start"] = True
            elif remaining > seg:
                part["_tie_start"] = True
                part["_tie_stop"] = True
            elif not first:
                part["_tie_stop"] = True
                
            new_entries.append(part)
            remaining -= seg
            cur_bar += 1
            cur_pos = 0
            first = False
            
    entries = new_entries
    entries.sort(key=lambda x: x["beat_pos_absolute"])
    
    # --- Final guards: ensure all entries have valid values ---
    for e in entries:
        # string=0 guard: cap to string=6 (lowest guitar string)
        if e.get("string", 1) < 1:
            e["string"] = 6
        # duration_divs must be at least 1
        e["duration_divs"] = max(1, e["duration_divs"])
        # beat_pos must never be negative
        e["beat_pos"] = max(0, e["beat_pos"])
        e["beat_pos_in_bar"] = max(0, e["beat_pos_in_bar"])
    
    return entries


def _cap_durations_by_string(entries: List[dict], beats_per_bar: int):
    """同一弦上の次のノートとの重複を防止（小節またぎはタイ分割に委ねる）"""
    bar_total = beats_per_bar * DIVISIONS
    for i, e in enumerate(entries):
        my_string = e.get("string", 0)
        gap_same_string = bar_total * 4  # デフォルト: 実質無制限
        
        for j in range(i + 1, len(entries)):
            other = entries[j]
            gap = other["beat_pos_absolute"] - e["beat_pos_absolute"]
            if gap <= 0:
                continue
            if other.get("string", -1) == my_string:
                gap_same_string = gap
                break
        
        # ❌ 削除: max_in_bar での小節打ち切り（タイ分割を殺していた原因）
        e["duration_divs"] = min(e["duration_divs"], gap_same_string)
        e["duration_divs"] = max(1, e["duration_divs"])


def test_quantizer():
    """簡易テスト: ロマンス風アルペジオパターン"""
    # BPM=75, 3/4, 8分音符アルペジオ (E-B-G-B-E-B)
    bpm = 75
    sec_per_beat = 60.0 / bpm  # 0.8s
    eighth = sec_per_beat / 2   # 0.4s
    
    # 1小節のパターン: ベースE + 5個の8分アルペジオ
    notes = []
    beat_start = 0.0
    
    # Beat 1: E2(bass) + B3 + E4
    notes.append({"start": beat_start, "end": beat_start + 2.4, "pitch": 40, "string": 6, "fret": 0, "velocity": 0.8})
    
    # 8分音符パターン
    arp_notes = [
        (beat_start + 0.0, 59, 2, 0),   # B3
        (beat_start + eighth, 64, 1, 0), # E4
        (beat_start + eighth*2, 59, 2, 0), # B3
        (beat_start + eighth*3, 55, 3, 0), # G3
        (beat_start + eighth*4, 59, 2, 0), # B3
        (beat_start + eighth*5, 64, 1, 0), # E4
    ]
    for t, p, s, f in arp_notes:
        notes.append({"start": t, "end": t + eighth*0.9, "pitch": p, "string": s, "fret": f, "velocity": 0.6})
    
    beats = [i * sec_per_beat for i in range(12)]
    
    result = quantize_notes_music21(
        notes, beats, bpm,
        time_signature="3/4", beats_per_bar=3,
        rhythm_subdivision="straight"
    )
    
    print(f"Input: {len(notes)} notes, BPM={bpm}, 3/4")
    print(f"Output: {len(result)} entries")
    print()
    print(f"{'bar':>3} {'pos':>4} {'dur':>4} {'pitch':>5} {'str':>3} {'fret':>4} {'dot':>4} {'trip':>4}")
    print("-" * 40)
    for e in result:
        print(f"{e['bar']:3d} {e['beat_pos_in_bar']:4d} {e['duration_divs']:4d} "
              f"{e['pitch']:5d} {e['string']:3d} {e['fret']:4d} "
              f"{'Y' if e['is_dotted'] else '.':>4} {'Y' if e['is_triplet'] else '.':>4}")


if __name__ == "__main__":
    test_quantizer()
