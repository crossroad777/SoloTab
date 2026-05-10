"""
tab_renderer.py — TAB用MusicXML生成
====================================
弦/フレットデータからAlphaTabで表示可能なMusicXMLを生成する。
五線譜パートは省略し、TABのみを出力。
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import List, Optional
import math


def notes_to_tab_musicxml(notes: List[dict], *,
                          beats: List[float],
                          bpm: float = 120.0,
                          title: str = "Guitar TAB",
                          tuning: list | None = None,
                          chords: list | None = None,
                          time_signature: str = "4/4",
                          noise_gate: float = 0.0,
                          rhythm_info: dict | None = None,
                          key_signature: str = "C") -> tuple[str, list]:
    """
    Generate a MusicXML string with TAB staff only.

    Parameters
    ----------
    notes : list[dict]
        Notes with keys: start, end, pitch, string, fret, (velocity), (technique)
    beats : list[float]
        Beat times in seconds.
    bpm : float
        Tempo in BPM.
    title : str
        Song title.
    tuning : list[int]
        Open string MIDI notes [6th→1st].

    Returns
    -------
    tuple[str, list[str]]
        (MusicXML string, technique_map) where technique_map is a list of
        technique names in the same order as notes appear in the MusicXML.
        This can be used to apply technique flags to AlphaTab's score model.
    """
    if tuning is None:
        tuning = [40, 45, 50, 55, 59, 64]

    # Parse time signature
    beats_per_bar: int = 4
    beat_type: int = 4
    if time_signature == "3/4":
        beats_per_bar = 3
        beat_type = 4
    elif time_signature == "6/8":
        beats_per_bar = 6
        beat_type = 8
    else:  # 4/4 default
        beats_per_bar = 4
        beat_type = 4
    beat_interval = 60.0 / bpm if bpm > 0 else 0.5
    divisions: int = 12  # divisions per quarter note (12 = 16分音符+3連符対応: LCM(4,3))

    # テクニックマップ: MusicXMLのノート出力順と対応
    technique_map: List[str] = []

    # --- Noise Gate Filtering (Absolute Threshold) ---
    # MoEや特定モデルの出力は速度分布が平坦だったり極端になりやすいため、
    # Track-Relative な間引きを行うと正規の音まで大幅に削られてしまう。
    # ユーザーが指定した cut の値を直接絶対閾値として用いる。
    filtered_notes = []
    if noise_gate > 0.0:
        for n in notes:
            v = float(n.get("velocity", 0.5))
            if v > 1.0: v /= 127.0
            n["_v"] = v
            # noise_gateが0.1(10%)なら、velocity 0.1未満の極小ノイズだけを弾く
            if v >= noise_gate:
                filtered_notes.append(n)
    else:
        filtered_notes = notes.copy()
        
    # Fail-safe: if gate is too aggressive...
    if not filtered_notes and notes:
        filtered_notes = [max(notes, key=lambda x: float(x.get("velocity", 0)))]

    # Assign notes to bars and beats
    note_entries = _assign_to_bars(filtered_notes, beats, beats_per_bar, rhythm_info=rhythm_info)

    # Calculate total bars
    if note_entries:
        total_bars: int = max(int(e["bar"]) for e in note_entries) + 1
    elif beats:
        total_bars: int = max(1, len(beats) // beats_per_bar)
    else:
        total_bars: int = 1

    total_bars = max(total_bars, 1)

    # --- 音楽理論統合 ---
    is_triplet_mode = (rhythm_info or {}).get("subdivision") == "triplet"

    try:
        from music_theory import quantize_note_durations
        # 音価スナップ: duration_divsを音楽的音価に補正
        note_entries = quantize_note_durations(note_entries, is_triplet_mode=is_triplet_mode, beats_per_bar=beats_per_bar)

    except Exception as e:
        import traceback; traceback.print_exc()

    # Build XML
    root = ET.Element("score-partwise", version="4.0")

    # Work / Title
    work = ET.SubElement(root, "work")
    ET.SubElement(work, "work-title").text = title

    # Part list
    part_list = ET.SubElement(root, "part-list")
    sp = ET.SubElement(part_list, "score-part", id="P1")
    ET.SubElement(sp, "part-name").text = "Guitar"
    si = ET.SubElement(sp, "score-instrument", id="P1-I1")
    ET.SubElement(si, "instrument-name").text = "Acoustic Guitar (steel)"
    mi = ET.SubElement(sp, "midi-instrument", id="P1-I1")
    ET.SubElement(mi, "midi-channel").text = "1"
    ET.SubElement(mi, "midi-program").text = "26"
    ET.SubElement(mi, "volume").text = "80"

    # Part
    part = ET.SubElement(root, "part", id="P1")



    for bar_num in range(total_bars):
        measure = ET.SubElement(part, "measure", number=str(bar_num + 1))

        # Attributes (first bar only)
        if bar_num == 0:
            attrs = ET.SubElement(measure, "attributes")
            ET.SubElement(attrs, "divisions").text = str(divisions)
            time_el = ET.SubElement(attrs, "time")
            ET.SubElement(time_el, "beats").text = str(beats_per_bar)
            ET.SubElement(time_el, "beat-type").text = str(beat_type)

            # Key signature
            key_fifths_map = {"C": 0, "Am": 0, "G": 1, "Em": 1, "D": 2, "Bm": 2,
                              "A": 3, "F#m": 3, "E": 4, "C#m": 4, "B": 5,
                              "F": -1, "Dm": -1, "Bb": -2, "Gm": -2, "Eb": -3, "Cm": -3, "Ab": -4}
            fifths = key_fifths_map.get(key_signature, 0)
            key_el = ET.SubElement(attrs, "key")
            ET.SubElement(key_el, "fifths").text = str(fifths)
            if key_signature.endswith("m"):
                ET.SubElement(key_el, "mode").text = "minor"
            else:
                ET.SubElement(key_el, "mode").text = "major"

            # 1スタッフ構造: 五線譜+TABをAlphaTab ScoreTabが統合表示
            clef1 = ET.SubElement(attrs, "clef")
            ET.SubElement(clef1, "sign").text = "G"
            ET.SubElement(clef1, "line").text = "2"
            ET.SubElement(clef1, "clef-octave-change").text = "-1"

            # TABチューニング情報
            sd = ET.SubElement(attrs, "staff-details")
            ET.SubElement(sd, "staff-lines").text = "6"
            for i in range(6):
                st = ET.SubElement(sd, "staff-tuning", line=str(i + 1))
                ET.SubElement(st, "tuning-step").text = _midi_to_step(tuning[i])
                ET.SubElement(st, "tuning-octave").text = str(_midi_to_octave(tuning[i]))

            # Direction (tempo)
            direction = ET.SubElement(measure, "direction", placement="above")
            dt = ET.SubElement(direction, "direction-type")
            metro = ET.SubElement(dt, "metronome")
            ET.SubElement(metro, "beat-unit").text = "quarter"
            ET.SubElement(metro, "per-minute").text = str(int(bpm))
            sound = ET.SubElement(direction, "sound", tempo=str(int(bpm)))



        # Add chord symbol (harmony) at start of bar
        if chords is not None and isinstance(chords, list):
            bstart_idx = int(bar_num) * int(beats_per_bar)
            bend_idx = int(min((int(bar_num) + 1) * int(beats_per_bar), len(beats) - 1))
            bar_start_time = beats[bstart_idx] if bstart_idx < len(beats) else 0.0
            bar_end_time = beats[bend_idx] if beats else 0.0
            for chord in chords:
                if chord["start"] <= bar_start_time < chord["end"]:
                    chord_name = chord["chord"]
                    if chord_name and chord_name != "N.C.":
                        harmony = ET.SubElement(measure, "harmony")
                        root_el = ET.SubElement(harmony, "root")
                        # Parse chord name: e.g. "C#m" -> root="C", alter=1, kind="minor"
                        root_step = chord_name[0]
                        alter = 0
                        kind = "major"
                        rest = chord_name[1:]
                        if rest.startswith("#"):
                            alter = 1
                            rest = rest[1:]
                        elif rest.startswith("b"):
                            alter = -1
                            rest = rest[1:]
                        if rest == "m":
                            kind = "minor"
                        elif rest == "7":
                            kind = "dominant"
                        elif rest == "m7":
                            kind = "minor-seventh"
                        ET.SubElement(root_el, "root-step").text = root_step
                        if alter != 0:
                            ET.SubElement(root_el, "root-alter").text = str(alter)
                        ET.SubElement(harmony, "kind").text = kind
                    break

        # Get notes for this bar
        bar_notes = [e for e in note_entries if e["bar"] == bar_num]

        bar_total: int = int(divisions) * int(beats_per_bar)

        if not bar_notes:
            # Empty bar: whole rest on both staves
            note_el = ET.SubElement(measure, "note")
            ET.SubElement(note_el, "rest")
            ET.SubElement(note_el, "duration").text = str(bar_total)
            ET.SubElement(note_el, "type").text = "whole"
        else:
            bar_notes.sort(key=lambda e: float(e["beat_pos"]))
            groups: List[List[dict]] = _group_by_time(bar_notes, threshold=0.1)

            # === 1スタッフ構造: 五線譜+TAB統合 ===
            current_pos: int = 0
            for group_idx, group in enumerate(groups):
                target_pos: int = int(float(group[0]["beat_pos"]))
                gap: int = target_pos - current_pos
                if gap > 0:
                    rest_el = ET.SubElement(measure, "note")
                    ET.SubElement(rest_el, "rest")
                    ET.SubElement(rest_el, "duration").text = str(gap)
                    ET.SubElement(rest_el, "type").text = _duration_to_type(gap, divisions)
                    if is_triplet_mode and gap in [4, 8]:
                        tm = ET.SubElement(rest_el, "time-modification")
                        ET.SubElement(tm, "actual-notes").text = "3"
                        ET.SubElement(tm, "normal-notes").text = "2"
                    current_pos = target_pos

                next_target: int = bar_total if group_idx + 1 >= len(groups) else int(float(groups[group_idx + 1][0]["beat_pos"]))
                gap_to_next: int = max(1, min(next_target - target_pos, bar_total - target_pos))

                for i, entry in enumerate(group):
                    dur = int(entry.get("duration_divs", gap_to_next))
                    dur = min(dur, gap_to_next, bar_total - target_pos)
                    note_el = ET.SubElement(measure, "note")
                    if i > 0: ET.SubElement(note_el, "chord")
                    pitch_el = ET.SubElement(note_el, "pitch")
                    pitch = int(entry["pitch"])
                    ET.SubElement(pitch_el, "step").text = _midi_to_step(pitch)
                    alter = _midi_to_alter(pitch)
                    if alter != 0: ET.SubElement(pitch_el, "alter").text = str(alter)
                    ET.SubElement(pitch_el, "octave").text = str(_midi_to_octave(pitch))
                    ET.SubElement(note_el, "duration").text = str(dur)
                    voice = "2" if pitch <= 52 else "1"
                    ET.SubElement(note_el, "voice").text = voice
                    ET.SubElement(note_el, "type").text = _duration_to_type(dur, divisions)
                    if entry.get("is_dotted"):
                        ET.SubElement(note_el, "dot")
                    if is_triplet_mode and dur in [4, 8]:
                        tm = ET.SubElement(note_el, "time-modification")
                        ET.SubElement(tm, "actual-notes").text = "3"
                        ET.SubElement(tm, "normal-notes").text = "2"
                    if entry.get("_tie_start"):
                        ET.SubElement(note_el, "tie", type="start")
                    ET.SubElement(note_el, "stem").text = "up" if voice == "1" else "down"

                    # notations: fret/string + テクニック（1スタッフに統合）
                    notations = ET.SubElement(note_el, "notations")
                    if entry.get("_tie_start"):
                        ET.SubElement(notations, "tied", type="start")
                    if is_triplet_mode and i == 0 and dur in [4, 8] and bar_num < 4:
                        cycle = dur * 3
                        rem = target_pos % cycle
                        if rem == 0: ET.SubElement(notations, "tuplet", type="start", bracket="yes")
                        elif rem == cycle - dur: ET.SubElement(notations, "tuplet", type="stop")

                    technical = ET.SubElement(notations, "technical")
                    ET.SubElement(technical, "string").text = str(entry.get("string", 1))
                    ET.SubElement(technical, "fret").text = str(entry.get("fret", 0))

                    tech = str(entry.get("technique") or "normal")
                    if tech == "h":
                        ho = ET.SubElement(technical, "hammer-on", type="start"); ho.text = "H"
                        technique_map.append("hammer_on")
                    elif tech == "p":
                        po = ET.SubElement(technical, "pull-off", type="start"); po.text = "P"
                        technique_map.append("pull_off")
                    elif tech == "/":
                        ET.SubElement(notations, "slide", type="start", **{"line-type": "solid"})
                        technique_map.append("slide_up")
                    elif tech == "\\":
                        ET.SubElement(notations, "slide", type="start", **{"line-type": "solid"})
                        technique_map.append("slide_down")
                    elif tech == "gliss_up":
                        gl = ET.SubElement(notations, "glissando", type="start", **{"line-type": "wavy"})
                        gl.text = "gliss."
                        technique_map.append("gliss_up")
                    elif tech == "gliss_down":
                        gl = ET.SubElement(notations, "glissando", type="start", **{"line-type": "wavy"})
                        gl.text = "gliss."
                        technique_map.append("gliss_down")
                    elif tech == "palm_mute": technique_map.append("palm_mute")
                    elif tech == "harmonic":
                        ET.SubElement(technical, "harmonic"); technique_map.append("harmonic")
                    elif tech == "b":
                        bend_el = ET.SubElement(technical, "bend")
                        ET.SubElement(bend_el, "bend-alter").text = "2"; technique_map.append("bend")
                    elif tech == "~":
                        ornaments = notations.find("ornaments")
                        if ornaments is None: ornaments = ET.SubElement(notations, "ornaments")
                        ET.SubElement(ornaments, "wavy-line", type="start"); technique_map.append("vibrato")
                    elif tech == "x":
                        ET.SubElement(note_el, "notehead").text = "x"; technique_map.append("ghost_note")
                    elif tech == "tr":
                        ornaments = notations.find("ornaments")
                        if ornaments is None: ornaments = ET.SubElement(notations, "ornaments")
                        ET.SubElement(ornaments, "trill-mark"); technique_map.append("trill")
                    elif tech == "let_ring":
                        dir_el = ET.SubElement(measure, "direction", placement="below")
                        dt = ET.SubElement(dir_el, "direction-type")
                        words = ET.SubElement(dt, "words", **{"font-style": "italic", "font-size": "7"})
                        words.text = "let ring"
                        technique_map.append("let_ring")
                    else: technique_map.append("normal")

                current_pos = current_pos + gap_to_next

            remaining = bar_total - current_pos
            if remaining > 0:
                _add_forward(measure, remaining)

    # Serialize with pretty print (DOMParser互換性のため)
    xml_str = ET.tostring(root, encoding="unicode")
    # minidomでインデント付き整形出力
    try:
        dom = minidom.parseString(xml_str)
        xml_str = dom.toprettyxml(indent="  ", encoding=None)
        # toprettyxmlは先頭に<?xml?>を付けるので、自前のheaderは不要
        # ただしDOCTYPEを挿入する必要がある
        lines = xml_str.split("\n")
        # <?xml ...?> の直後にDOCTYPEを挿入
        header_line = lines[0]  # <?xml version="1.0" ?>
        doctype = '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">'
        rest_lines = "\n".join(lines[1:])
        return header_line + "\n" + doctype + "\n" + rest_lines, technique_map
    except Exception:
        # フォールバック: 従来の1行出力
        header = '<?xml version="1.0" encoding="UTF-8"?>\n'
        header += '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
        header += '"http://www.musicxml.org/dtds/partwise.dtd">\n'
        return header + xml_str, technique_map


def _assign_to_bars(notes: List[dict], beats: List[float], beats_per_bar: int, rhythm_info: dict | None = None) -> List[dict]:
    """Assign each note to a bar and beat position (in divisions).
    
    同時発音ノート(50ms以内)は先にグルーピングし、
    代表時刻で統一的にビートスナップすることで和音を正しく出力する。
    """
    import numpy as np  # type: ignore

    if not beats or not notes:
        return []

    beats_arr = np.array(beats)
    divisions = 12  # per quarter note (12 = 三連符にも対応: 4*3)
    
    # 各ノートに個別のbeat_posを計算（グループ化しない → アルペジオの順次音を分離）
    sorted_notes = sorted(notes, key=lambda n: (float(n["start"]), int(n["pitch"])))
    
    entries: List[dict] = []

    for note in sorted_notes:
        t = float(note["start"])
        
        # Find the beat that is strictly before or exactly at t
        idx = int(np.searchsorted(beats_arr, t, side='right')) - 1
        idx = max(0, min(idx, len(beats_arr) - 1))

        bar = idx // beats_per_bar
        beat_in_bar = idx % beats_per_bar

        # Sub-beat position (fractional divisions)
        if idx < len(beats_arr):
            beat_time = float(beats_arr[idx])
            next_beat_time = float(beats_arr[idx + 1]) if idx + 1 < len(beats_arr) else beat_time + 0.5
            frac = (t - beat_time) / (next_beat_time - beat_time) if next_beat_time > beat_time else 0.0
            frac = max(0.0, min(frac, 0.99))
            
            # --- MUSICAL QUANTIZATION ---
            # Force raw fractions to snap precisely to musical grids.
            # Triplet mode: snap to 1/3 divisions (0, 4, 8)
            # Straight mode: snap to 1/4 and 1/3 (0, 3, 4, 6, 8, 9, 12)
            raw_sub = frac * divisions
            
            is_triplet = (rhythm_info or {}).get('subdivision') == 'triplet'
            if is_triplet:
                # 3連符: 0, 4, 8 のみ (= beat/3 divisions)
                grid = [0, 4, 8, 12]
            else:
                grid = [0, 3, 4, 6, 8, 9, 12]
            sub_divs = min(grid, key=lambda x: abs(x - raw_sub))
            if sub_divs == 12:
                sub_divs = 0
                beat_in_bar = (beat_in_bar + 1) % beats_per_bar
                if beat_in_bar == 0:
                    bar += 1
        else:
            sub_divs = 0

        beat_pos = bar * (beats_per_bar * divisions) + beat_in_bar * divisions + sub_divs
        beat_interval = (float(beats_arr[1]) - float(beats_arr[0])) if len(beats_arr) > 1 else 0.5

        # --- ノートの実際の持続時間を秒→divisions変換 ---
        note_end = float(note.get("end", t + beat_interval))
        actual_dur_sec = max(0.01, note_end - t)
        # 秒→divisions: 1拍 = beat_interval秒 = divisions (12) divs
        actual_dur_divs = max(1, int(round(actual_dur_sec / beat_interval * divisions)))

        entries.append({
            "bar": bar,
            "beat_pos_absolute": beat_pos,
            "beat_pos_in_bar": beat_in_bar * divisions + sub_divs,
            "duration_divs": actual_dur_divs,  # ★ノートの実際の持続時間
            "pitch": note["pitch"],
            "string": note.get("string", 1),
            "fret": note.get("fret", 0),
            "technique": note.get("technique"),
            "velocity": note.get("velocity", 0.5),
            "start_time": t,
        })

    # beat_pos: _group_by_time が参照するキーを設定
    for e in entries:
        e["beat_pos"] = e["beat_pos_in_bar"]

    # --- duration_divsの後処理: 同一弦の次のノートとの重複を防止 ---
    # ギターの物理特性: 異なる弦のノートは同時に鳴り続ける
    # 同じ弦の次のノートが来た時だけ、前のノートは切れる
    bar_total = beats_per_bar * divisions
    for i, e in enumerate(entries):
        my_string = e.get("string", 0)
        # 同じ弦の次のノートを探す
        gap_same_string = bar_total * 2  # デフォルト: 制限なし
        for j in range(i + 1, len(entries)):
            other = entries[j]
            gap = other["beat_pos_absolute"] - e["beat_pos_absolute"]
            if gap <= 0:
                continue  # 同時発音はスキップ
            if other.get("string", -1) == my_string:
                gap_same_string = gap
                break
            # 弦情報がない場合は全ノートを考慮
            if my_string == 0:
                gap_same_string = min(gap_same_string, gap)
                break
        # 実際のduration_divsを同弦の次のノートまでの距離でキャップ
        e["duration_divs"] = min(e["duration_divs"], gap_same_string)
        # 小節内に収まるようにキャップ
        max_in_bar = bar_total - e["beat_pos_in_bar"]
        e["duration_divs"] = min(e["duration_divs"], max(1, max_in_bar))
        e["duration_divs"] = max(1, e["duration_divs"])

    return entries


def _group_by_time(entries: List[dict], threshold: float = 0.1) -> list:
    """Group entries that are truly simultaneous (mapped to the exact same quantized beat_pos)."""
    if not entries:
        return []
    groups: List[List[dict]] = [[entries[0]]]
    for i in range(1, len(entries)):
        e = entries[i]
        prev = groups[-1][0]
        # Since we applied Musical Quantization, identical beat_pos means they belong in the same chord.
        # Safe float conversion to prevent "ValueError: invalid literal for int() with base 10: '4.0'"
        same_beat = abs(float(e["beat_pos"]) - float(prev["beat_pos"])) < 0.1
        if same_beat:
            groups[-1].append(e)
        else:
            groups.append([e])

    # 各グループを最大6ノート(ギターの弦数)に制限
    # 同じ弦のノートは1つだけ残す
    # 同時に「弱すぎるノイズ音(AIの誤検知オーバートーン)」を足切りするノイズゲートを導入
    limited: List[List[dict]] = []
    for group in groups:
        seen_strings: dict = {}
        for e in group:
            s = int(e.get("string", 0))
            if s not in seen_strings:
                seen_strings[s] = e
            else:
                # 同弦: velocity が高い方を優先
                if float(e.get("velocity", 0)) > float(seen_strings[s].get("velocity", 0)):
                    seen_strings[s] = e
        vals = list(seen_strings.values())
        
        cleaned = []
        for v in vals:
            vel = float(v.get("velocity", 0.5))
            if vel > 1.0: vel /= 127.0
            # Inner noise gate acting as string-level de-duplication safety
            if vel >= 0.05:
                cleaned.append(v)
                
        # If everything was filtered out but there WERE notes, keep the loudest one to avoid dropping the beat entirely
        if not cleaned and vals:
            loudest = max(vals, key=lambda x: float(x.get("velocity", 0)))
            cleaned.append(loudest)
            
        limited.append([cleaned[i] for i in range(min(6, len(cleaned)))])
        
    return limited


def _add_forward(measure: ET.Element, duration: int):
    """Add a <forward> element to advance time."""
    fwd = ET.SubElement(measure, "forward")
    ET.SubElement(fwd, "duration").text = str(int(duration))


def _midi_to_step(midi_num: int) -> str:
    steps = ["C", "C", "D", "D", "E", "F", "F", "G", "G", "A", "A", "B"]
    return steps[midi_num % 12]


def _midi_to_alter(midi_num: int) -> int:
    alters = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
    return alters[midi_num % 12]


def _midi_to_octave(midi_num: int) -> int:
    return (midi_num // 12) - 1


def _duration_to_type(dur_divs: int, divisions: int = 12) -> str:
    """Convert duration in divisions to MusicXML type name.
    
    With divisions=12:
      whole=48, half=24, quarter=12, eighth=6, 16th=3
      triplet-quarter=8, triplet-eighth=4
    """
    if dur_divs >= 48: return "whole"
    if dur_divs >= 24: return "half"
    if dur_divs >= 12: return "quarter"
    if dur_divs >= 8: return "eighth" # Triplet quarter
    if dur_divs >= 6: return "eighth"
    if dur_divs >= 4: return "eighth" # Triplet eighth
    return "16th"


def _velocity_to_dynamics(velocity: float) -> Optional[str]:
    """Convert velocity (0.0-1.0) to MusicXML dynamics marking.
    
    Returns None for moderate velocities to avoid excessive markings.
    """
    if velocity < 0.2:
        return "pp"
    elif velocity < 0.35:
        return "p"
    elif velocity < 0.5:
        return "mp"
    elif velocity < 0.65:
        return None  # mf is default, skip
    elif velocity < 0.8:
        return "f"
    else:
        return "ff"
