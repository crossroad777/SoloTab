"""
tab_renderer.py — TAB用MusicXML生成
====================================
弦/フレットデータからAlphaTabで表示可能なMusicXMLを生成する。
五線譜パートは省略し、TABのみを出力。
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import List, Optional, cast
import math


def notes_to_tab_musicxml(notes: List[dict], *,
                          beats: List[float],
                          bpm: float = 120.0,
                          title: str = "Guitar TAB",
                          tuning: list | None = None,
                          chords: list | None = None,
                          time_signature: str = "4/4",
                          noise_gate: float = 0.0) -> tuple[str, list]:
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
    note_entries = _assign_to_bars(filtered_notes, beats, beats_per_bar)

    # Calculate total bars
    if note_entries:
        total_bars: int = max(int(e["bar"]) for e in note_entries) + 1
    elif beats:
        total_bars: int = max(1, len(beats) // beats_per_bar)
    else:
        total_bars: int = 1

    total_bars = max(total_bars, 1)

    # Build XML
    root = ET.Element("score-partwise", version="4.0")

    # Work / Title
    work = ET.SubElement(root, "work")
    ET.SubElement(work, "work-title").text = title

    # Part list
    part_list = ET.SubElement(root, "part-list")
    sp = ET.SubElement(part_list, "score-part", id="P1")
    ET.SubElement(sp, "part-name").text = "Guitar TAB"

    # Part
    part = ET.SubElement(root, "part", id="P1")

    last_dyn_mark: Optional[str] = None

    for bar_num in range(total_bars):
        measure = ET.SubElement(part, "measure", number=str(bar_num + 1))

        # Attributes (first bar only)
        if bar_num == 0:
            attrs = ET.SubElement(measure, "attributes")
            ET.SubElement(attrs, "divisions").text = str(divisions)
            time_el = ET.SubElement(attrs, "time")
            ET.SubElement(time_el, "beats").text = str(beats_per_bar)
            ET.SubElement(time_el, "beat-type").text = str(beat_type)

            # TAB clef
            clef = ET.SubElement(attrs, "clef")
            ET.SubElement(clef, "sign").text = "TAB"
            ET.SubElement(clef, "line").text = "5"

            # Staff details (6-string guitar TAB)
            sd = ET.SubElement(attrs, "staff-details")
            ET.SubElement(sd, "staff-lines").text = "6"
            _tuning_names = ["E", "A", "D", "G", "B", "E"]
            _tuning_octaves = ["2", "2", "3", "3", "3", "4"]
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

        if not bar_notes:
            # Empty bar: whole rest
            note_el = ET.SubElement(measure, "note")
            rest_el = ET.SubElement(note_el, "rest")
            ET.SubElement(note_el, "duration").text = str(int(divisions) * int(beats_per_bar))
            ET.SubElement(note_el, "type").text = "whole"
        else:
            # Sort by beat position
            bar_notes.sort(key=lambda e: float(e["beat_pos"]))

            groups: List[List[dict]] = _group_by_time(bar_notes, threshold=0.1)

            current_pos: int = 0  # in divisions units
            for group_idx, group in enumerate(groups):
                target_pos: int = int(float(group[0]["beat_pos"]))

                # Insert rest if there's a gap
                gap: int = target_pos - int(current_pos)  # type: ignore
                if gap > 0:
                    # 休符として挿入（TABビューアで表示される）
                    rest_el = ET.SubElement(measure, "note")
                    ET.SubElement(rest_el, "rest")
                    rest_dur: int = int(gap)
                    ET.SubElement(rest_el, "duration").text = str(rest_dur)
                    ET.SubElement(rest_el, "type").text = _duration_to_type(rest_dur, int(divisions))
                    
                    if rest_dur in [4, 8]:
                        tm = ET.SubElement(rest_el, "time-modification")
                        ET.SubElement(tm, "actual-notes").text = "3"
                        ET.SubElement(tm, "normal-notes").text = "2"
                        
                    current_pos = target_pos

                # Determine the exact distance to the NEXT note to prevent single-voice timeline shifting
                bar_total_in: int = int(divisions) * int(beats_per_bar)
                next_target_pos: int = bar_total_in
                if group_idx + 1 < len(groups):
                    next_target_pos = int(float(groups[group_idx + 1][0]["beat_pos"]))

                dur_advance: int = next_target_pos - target_pos
                if dur_advance < 1:
                    dur_advance = max(1, int(group[0].get("duration_divs", divisions)))
                
                max_dur: int = bar_total_in - target_pos
                if dur_advance > max_dur:
                    dur_advance = max_dur

                # Dynamics marker logic removed to prevent visual clutter
                # FretNet's raw velocity variance causes chaotic f/ff switching.
                # In standard tabs (like Songsterr), velocity is implicit.
                
                for i, entry in enumerate(group):
                    # Force duration to exactly dur_advance to perfectly serialize the timeline
                    dur: int = dur_advance

                    note_el = ET.SubElement(measure, "note")

                    if i > 0:
                        ET.SubElement(note_el, "chord")

                    pitch_el = ET.SubElement(note_el, "pitch")
                    pitch: int = int(entry["pitch"])
                    ET.SubElement(pitch_el, "step").text = _midi_to_step(pitch)
                    alter = _midi_to_alter(pitch)
                    if alter != 0:
                        ET.SubElement(pitch_el, "alter").text = str(alter)
                    ET.SubElement(pitch_el, "octave").text = str(_midi_to_octave(pitch))

                    ET.SubElement(note_el, "duration").text = str(dur)
                    ET.SubElement(note_el, "type").text = _duration_to_type(dur, int(divisions))

                    # 3-tuple / Triplet modifier
                    if dur in [4, 8]:
                        tm = ET.SubElement(note_el, "time-modification")
                        ET.SubElement(tm, "actual-notes").text = "3"
                        ET.SubElement(tm, "normal-notes").text = "2"

                    ET.SubElement(note_el, "stem").text = "none"

                    # Notations with technical (string/fret)
                    notations = ET.SubElement(note_el, "notations")
                    
                    # 3-tuple brackets (bracket visualizer for AlphaTab)
                    if i == 0 and dur in [4, 8]:
                        cycle = dur * 3
                        rem = target_pos % cycle
                        if rem == 0:
                            ET.SubElement(notations, "tuplet", type="start", bracket="yes")
                        elif rem == cycle - dur:
                            ET.SubElement(notations, "tuplet", type="stop")

                    technical = ET.SubElement(notations, "technical")
                    ET.SubElement(technical, "string").text = str(entry.get("string", 1))
                    ET.SubElement(technical, "fret").text = str(entry.get("fret", 0))

                    # テクニックマップに記録 + MusicXML要素の追加
                    tech = str(entry.get("technique") or "normal")
                    
                    if tech == "h":
                        technique_map.append("hammer_on")
                    elif tech == "p":
                        technique_map.append("pull_off")
                    elif tech == "/":
                        technique_map.append("slide_up")
                    elif tech == "\\":
                        technique_map.append("slide_down")
                    elif tech == "palm_mute":
                        technique_map.append("palm_mute")
                    elif tech == "harmonic":
                        ET.SubElement(technical, "harmonic")
                        technique_map.append("harmonic")
                    elif tech == "b":
                        technique_map.append("bend")
                    elif tech == "~":
                        # ビブラート: <ornaments><wavy-line>
                        ornaments = notations.find("ornaments")
                        if ornaments is None:
                            ornaments = ET.SubElement(notations, "ornaments")
                        ET.SubElement(ornaments, "wavy-line", type="start")
                        technique_map.append("vibrato")
                    elif tech == "let_ring":
                        # Convert FretNet's let_ring to a visual indicator
                        if i == 0 and len(group) == 1:
                            direction2 = ET.SubElement(measure, "direction", placement="above")
                            dt2 = ET.SubElement(direction2, "direction-type")
                            words = ET.SubElement(dt2, "words", font_style="italic")
                            words.text = "let ring"
                        technique_map.append("normal")
                    elif tech == "x":
                        # ゴーストノート: noteheadをxに
                        notehead = ET.SubElement(note_el, "notehead")
                        notehead.text = "x"
                        technique_map.append("ghost_note")
                    elif tech == "tr":
                        # トリル: <ornaments><trill-mark>
                        ornaments = notations.find("ornaments")
                        if ornaments is None:
                            ornaments = ET.SubElement(notations, "ornaments")
                        ET.SubElement(ornaments, "trill-mark")
                        technique_map.append("trill")
                    else:
                        technique_map.append("normal")

                current_pos = int(current_pos) + int(dur_advance)  # type: ignore

            # Fill remaining bar with forward
            bar_total_out: int = int(divisions) * int(beats_per_bar)
            remaining: int = bar_total_out - int(current_pos)
            if remaining > 0:
                _add_forward(measure, remaining)

    # Serialize
    xml_str = ET.tostring(root, encoding="unicode")
    # Add XML declaration and DOCTYPE
    header = '<?xml version="1.0" encoding="UTF-8"?>\n'
    header += '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
    header += '"http://www.musicxml.org/dtds/partwise.dtd">\n'

    return header + xml_str, technique_map


def _assign_to_bars(notes: List[dict], beats: List[float], beats_per_bar: int) -> List[dict]:
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
        
        # Find closest beat
        idx = int(np.searchsorted(beats_arr, t))
        idx = max(0, min(idx, len(beats_arr) - 1))

        # Snap to nearest beat
        if idx > 0 and abs(beats_arr[idx - 1] - t) < abs(beats_arr[idx] - t):
            idx -= 1

        bar = idx // beats_per_bar
        beat_in_bar = idx % beats_per_bar

        # Sub-beat position (fractional divisions)
        if idx < len(beats_arr):
            beat_time = float(beats_arr[idx])
            next_beat_time = float(beats_arr[idx + 1]) if idx + 1 < len(beats_arr) else beat_time + 0.5
            frac = (t - beat_time) / (next_beat_time - beat_time) if next_beat_time > beat_time else 0.0
            frac = max(0.0, min(frac, 0.99))
            
            # --- MUSICAL QUANTIZATION ---
            # Force raw fractions to snap precisely to 16th (3,6,9) or 8th-triplet (4,8) grids
            # This prevents 16th/32nd-note tuplet chaos from raw mathematical timing.
            raw_sub = frac * divisions
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

        note_dur_sec = float(note["end"]) - float(note["start"])
        dur_divs = max(1, int(round(note_dur_sec / beat_interval * divisions)))
        dur_divs = min(dur_divs, divisions * beats_per_bar)

        entries.append({
            "bar": bar,
            "beat_pos_absolute": beat_pos, # Used to find exact durations without bar wrap logic bugs
            "beat_pos_in_bar": beat_in_bar * divisions + sub_divs,
            "duration_divs": dur_divs,
            "pitch": note["pitch"],
            "string": note.get("string", 1),
            "fret": note.get("fret", 0),
            "technique": note.get("technique"),
            "velocity": note.get("velocity", 0.5),
            "start_time": t,
        })

    # Fix: beat_pos in the original logic was used as relative to the bar. We use absolute for grouping across.
    # Grouping function relies on "beat_pos" so we rename back appropriately.
    for e in entries:
        e["beat_pos"] = e["beat_pos_in_bar"]

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
