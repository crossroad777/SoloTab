"""
tab_renderer.py — TAB用MusicXML生成
====================================
弦/フレットデータからAlphaTabで表示可能なMusicXMLを生成する。
1スタッフ構造でAlphaTab ScoreTabプロファイルが自動的にScore+TAB表示を生成する。
(明示的2スタッフ構造はAlphaTab 1.3.0でゴースト段を発生させるため不使用)
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import List, Optional
import math


def notes_to_tab_musicxml(notes: List[dict], *,
                          beats: List[float],
                          backing_notes: List[dict] = None,
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

    # Noise gate filter helper
    def _filter_gate(notes_list):
        filtered = []
        if noise_gate > 0.0:
            for n in notes_list:
                v = float(n.get("velocity", 0.5))
                if v > 1.0: v /= 127.0
                n["_v"] = v
                # noise_gateが0.1(10%)なら、velocity 0.1未満の極小ノイズだけを弾く
                if v >= noise_gate:
                    filtered.append(n)
        else:
            filtered = notes_list.copy()
        if not filtered and notes_list:
            filtered = [max(notes_list, key=lambda x: float(x.get("velocity", 0)))]
        return filtered

    is_2tracks = backing_notes is not None

    # Melody filter and assign
    filtered_melody = _filter_gate(notes)
    note_entries = _assign_to_bars(filtered_melody, beats, beats_per_bar, bpm=bpm, time_signature=time_signature, rhythm_info=rhythm_info)

    # Backing filter and assign
    backing_entries = []
    if is_2tracks:
        filtered_backing = _filter_gate(backing_notes)
        backing_entries = _assign_to_bars(filtered_backing, beats, beats_per_bar, bpm=bpm, time_signature=time_signature, rhythm_info=rhythm_info)

    # Calculate total bars (max of both tracks)
    combined = note_entries + backing_entries
    if combined:
        total_bars: int = max(int(e["bar"]) for e in combined) + 1
    elif beats:
        total_bars: int = max(1, len(beats) // beats_per_bar)
    else:
        total_bars: int = 1

    total_bars = max(total_bars, 1)

    # --- 音楽理論統合 ---
    is_triplet_mode = (rhythm_info or {}).get("subdivision") == "triplet"

    try:
        from music_theory import quantize_note_durations
        note_entries = quantize_note_durations(note_entries, is_triplet_mode=is_triplet_mode, beats_per_bar=beats_per_bar)
        if is_2tracks and backing_entries:
            backing_entries = quantize_note_durations(backing_entries, is_triplet_mode=is_triplet_mode, beats_per_bar=beats_per_bar)
    except Exception as e:
        import traceback; traceback.print_exc()

    # Build XML
    root = ET.Element("score-partwise", version="4.0")

    # Work / Title
    work = ET.SubElement(root, "work")
    ET.SubElement(work, "work-title").text = title

    # Part list
    part_list = ET.SubElement(root, "part-list")
    
    if not is_2tracks:
        sp = ET.SubElement(part_list, "score-part", id="P1")
        ET.SubElement(sp, "part-name").text = "Guitar"
        si = ET.SubElement(sp, "score-instrument", id="P1-I1")
        ET.SubElement(si, "instrument-name").text = "Acoustic Guitar (steel)"
        mi = ET.SubElement(sp, "midi-instrument", id="P1-I1")
        ET.SubElement(mi, "midi-channel").text = "1"
        ET.SubElement(mi, "midi-program").text = "26"
        ET.SubElement(mi, "volume").text = "80"
    else:
        # Part 1: Melody
        sp1 = ET.SubElement(part_list, "score-part", id="P1")
        ET.SubElement(sp1, "part-name").text = "Guitar (Melody)"
        si1 = ET.SubElement(sp1, "score-instrument", id="P1-I1")
        ET.SubElement(si1, "instrument-name").text = "Acoustic Guitar (steel)"
        mi1 = ET.SubElement(sp1, "midi-instrument", id="P1-I1")
        ET.SubElement(mi1, "midi-channel").text = "1"
        ET.SubElement(mi1, "midi-program").text = "26"
        ET.SubElement(mi1, "volume").text = "80"

        # Part 2: Backing
        sp2 = ET.SubElement(part_list, "score-part", id="P2")
        ET.SubElement(sp2, "part-name").text = "Guitar (Backing)"
        si2 = ET.SubElement(sp2, "score-instrument", id="P2-I1")
        ET.SubElement(si2, "instrument-name").text = "Acoustic Guitar (steel)"
        mi2 = ET.SubElement(sp2, "midi-instrument", id="P2-I1")
        ET.SubElement(mi2, "midi-channel").text = "2"
        ET.SubElement(mi2, "midi-program").text = "26"
        ET.SubElement(mi2, "volume").text = "70"

    # Part rendering loop
    parts_to_render = [("P1", note_entries)]
    if is_2tracks:
        parts_to_render.append(("P2", backing_entries))

    for part_id, entries_to_use in parts_to_render:
        part_el = ET.SubElement(root, "part", id=part_id)
        
        # Track active techniques for type="stop" tags
        active_slurs = {}       # string -> "h" or "p"
        active_slides = {}      # string -> "/" or "\\"
        active_gliss = {}       # string -> "gliss_up" or "gliss_down"
        active_start_elements = {} # string -> (parent_el, child_el)

        for bar_num in range(total_bars):
            measure = ET.SubElement(part_el, "measure", number=str(bar_num + 1))

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

                # Treble clef (standard G clef)
                clef_el = ET.SubElement(attrs, "clef")
                ET.SubElement(clef_el, "sign").text = "G"
                ET.SubElement(clef_el, "line").text = "2"

                # TABチューニング情報
                sd = ET.SubElement(attrs, "staff-details")
                ET.SubElement(sd, "staff-lines").text = "6"
                for i in range(6):
                    st = ET.SubElement(sd, "staff-tuning", line=str(i + 1))
                    ET.SubElement(st, "tuning-step").text = _midi_to_step(tuning[i])
                    ET.SubElement(st, "tuning-octave").text = str(_midi_to_octave(tuning[i]))

                # Direction (tempo) - Add to P1 only
                if part_id == "P1":
                    direction = ET.SubElement(measure, "direction", placement="above")
                    dt = ET.SubElement(direction, "direction-type")
                    metro = ET.SubElement(dt, "metronome")
                    ET.SubElement(metro, "beat-unit").text = "quarter"
                    ET.SubElement(metro, "per-minute").text = str(int(bpm))
                    sound = ET.SubElement(direction, "sound", tempo=str(int(bpm)))

            # Add chord symbol (harmony) at start of bar - Add to P1 only
            if part_id == "P1" and chords is not None and isinstance(chords, list):
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
            bar_notes = [e for e in entries_to_use if e["bar"] == bar_num]

            bar_total: int = int(divisions) * int(beats_per_bar)

            if not bar_notes:
                # Empty bar: whole rest
                note_el = ET.SubElement(measure, "note")
                ET.SubElement(note_el, "rest")
                ET.SubElement(note_el, "duration").text = str(bar_total)
                ET.SubElement(note_el, "voice").text = "1"
                ET.SubElement(note_el, "type").text = "whole"
            else:
                # 2声部（Voice 1: Melody/Arpeggio, Voice 2: Bass）の完全分離
                v2_notes = [e for e in bar_notes if (e.get("is_bass") or int(e.get("pitch", 60)) <= 52 or int(e.get("string", 1)) >= 4)]
                if not v2_notes and bar_notes:
                    # 明示的な低音弦がない場合、小節内最低音（1拍目優先）をベース声部に配置
                    sorted_by_pitch = sorted(bar_notes, key=lambda e: (float(e["beat_pos"]), int(e["pitch"])))
                    v2_notes = [sorted_by_pitch[0]]
                v1_notes = [e for e in bar_notes if e not in v2_notes]

                # --- Voice 1 (Melody / Arpeggio) ---
                if v1_notes:
                    v1_notes.sort(key=lambda e: float(e["beat_pos"]))
                    v1_groups = _group_by_time(v1_notes, threshold=0.1)
                    current_pos = 0

                    # 拍ごとのグループ集計（3連符のstart/stopペアリング用）
                    beat_groups_map = {}
                    for g_idx, group in enumerate(v1_groups):
                        t_pos = int(float(group[0]["beat_pos"]))
                        b_idx = t_pos // int(divisions)
                        beat_groups_map.setdefault(b_idx, []).append((g_idx, group, t_pos))

                    for g_idx, group in enumerate(v1_groups):
                        t_pos = int(float(group[0]["beat_pos"]))
                        b_idx = t_pos // int(divisions)
                        b_list = beat_groups_map.get(b_idx, [])
                        is_first_in_beat = (b_list and b_list[0][0] == g_idx)
                        is_last_in_beat  = (b_list and b_list[-1][0] == g_idx)

                        gap = t_pos - current_pos
                        if gap > 0:
                            rest_el = ET.SubElement(measure, "note")
                            ET.SubElement(rest_el, "rest")
                            ET.SubElement(rest_el, "duration").text = str(gap)
                            ET.SubElement(rest_el, "voice").text = "1"
                            ET.SubElement(rest_el, "type").text = _duration_to_type(gap, divisions)
                            if is_triplet_mode and gap in [4, 8]:
                                tm = ET.SubElement(rest_el, "time-modification")
                                ET.SubElement(tm, "actual-notes").text = "3"
                                ET.SubElement(tm, "normal-notes").text = "2"
                            current_pos = t_pos

                        next_t = bar_total if g_idx + 1 >= len(v1_groups) else int(float(v1_groups[g_idx + 1][0]["beat_pos"]))
                        gap_next = max(1, min(next_t - t_pos, bar_total - t_pos))

                        for i, entry in enumerate(group):
                            dur = int(entry.get("duration_divs", gap_next))
                            dur = min(dur, gap_next, bar_total - t_pos)
                            tech = str(entry.get("technique") or "normal")
                            pitch = int(entry["pitch"])
                            string_num = int(entry.get("string", 1))
                            fret_val = int(entry.get("fret", 0))

                            note_el = ET.SubElement(measure, "note")
                            if i > 0: ET.SubElement(note_el, "chord")
                            pitch_el = ET.SubElement(note_el, "pitch")
                            ET.SubElement(pitch_el, "step").text = _midi_to_step(pitch)
                            alter = _midi_to_alter(pitch)
                            if alter != 0: ET.SubElement(pitch_el, "alter").text = str(alter)
                            ET.SubElement(pitch_el, "octave").text = str(_midi_to_octave(pitch))
                            ET.SubElement(note_el, "duration").text = str(dur)
                            ET.SubElement(note_el, "voice").text = "1"
                            ET.SubElement(note_el, "type").text = _duration_to_type(dur, divisions)
                            if entry.get("is_dotted") or dur in [9, 18, 36, 54]:
                                ET.SubElement(note_el, "dot")
                            
                            is_trip = entry.get("is_triplet", False) or is_triplet_mode or (dur in [2, 4, 8])
                            if is_trip:
                                tm = ET.SubElement(note_el, "time-modification")
                                ET.SubElement(tm, "actual-notes").text = "3"
                                ET.SubElement(tm, "normal-notes").text = "2"
                            ET.SubElement(note_el, "stem").text = "up"

                            notations = ET.SubElement(note_el, "notations")
                            if is_trip and i == 0:  # 和音の場合は最初のノートのみtupletタグを付与
                                if len(b_list) > 1:
                                    if is_first_in_beat:
                                        ET.SubElement(notations, "tuplet", type="start", bracket="yes")
                                    elif is_last_in_beat:
                                        ET.SubElement(notations, "tuplet", type="stop")
                                else:
                                    ET.SubElement(notations, "tuplet", type="start", bracket="yes")
                                    ET.SubElement(notations, "tuplet", type="stop")

                            tech_el = ET.SubElement(notations, "technical")
                            ET.SubElement(tech_el, "string").text = str(string_num)
                            ET.SubElement(tech_el, "fret").text = str(fret_val)
                            
                            # 特殊奏法タグ (TASK-892)
                            n_tech = str(entry.get("technique", "")).lower()
                            if n_tech in ("x", "dead_note", "bh", "na"):
                                ET.SubElement(tech_el, "dead-note")
                                ET.SubElement(note_el, "notehead").text = "cross"
                            elif n_tech in ("t", "tap", "th"):
                                ET.SubElement(tech_el, "tap")
                            elif n_tech in ("harmonic", "nh", "ah"):
                                harm_el = ET.SubElement(tech_el, "harmonic")
                                ET.SubElement(harm_el, "natural")

                        current_pos = min(bar_total, t_pos + dur)

                    if current_pos < bar_total:
                        r_gap = bar_total - current_pos
                        rest_el = ET.SubElement(measure, "note")
                        ET.SubElement(rest_el, "rest")
                        ET.SubElement(rest_el, "duration").text = str(r_gap)
                        ET.SubElement(rest_el, "voice").text = "1"
                        ET.SubElement(rest_el, "type").text = _duration_to_type(r_gap, divisions)

                # --- Backup to Measure Start for Voice 2 ---
                if v1_notes and v2_notes:
                    backup_el = ET.SubElement(measure, "backup")
                    ET.SubElement(backup_el, "duration").text = str(bar_total)

                # --- Voice 2 (Bass) ---
                if v2_notes:
                    v2_notes.sort(key=lambda e: float(e["beat_pos"]))
                    b_entry = v2_notes[0]
                    b_pitch = int(b_entry["pitch"])
                    b_dur = bar_total  # 付点2分音符 / 小節全体

                    note_el = ET.SubElement(measure, "note")
                    pitch_el = ET.SubElement(note_el, "pitch")
                    ET.SubElement(pitch_el, "step").text = _midi_to_step(b_pitch)
                    alter = _midi_to_alter(b_pitch)
                    if alter != 0: ET.SubElement(pitch_el, "alter").text = str(alter)
                    ET.SubElement(pitch_el, "octave").text = str(_midi_to_octave(b_pitch))
                    ET.SubElement(note_el, "duration").text = str(b_dur)
                    ET.SubElement(note_el, "voice").text = "2"
                    ET.SubElement(note_el, "type").text = _duration_to_type(b_dur, divisions)
                    ET.SubElement(note_el, "dot")  # 付点
                    ET.SubElement(note_el, "stem").text = "down"

                    notations = ET.SubElement(note_el, "notations")
                    tech_el = ET.SubElement(notations, "technical")
                    ET.SubElement(tech_el, "string").text = str(b_entry.get("string", 6))
                    ET.SubElement(tech_el, "fret").text = str(b_entry.get("fret", 0))
                    
                    # 特殊奏法タグ (Voice 2)
                    b_tech = str(b_entry.get("technique", "")).lower()
                    if b_tech in ("x", "dead_note", "bh", "na"):
                        ET.SubElement(tech_el, "dead-note")
                        ET.SubElement(note_el, "notehead").text = "cross"
                    elif b_tech in ("t", "tap", "th"):
                        ET.SubElement(tech_el, "tap")
                    elif b_tech in ("harmonic", "nh", "ah"):
                        harm_el = ET.SubElement(tech_el, "harmonic")
                        ET.SubElement(harm_el, "natural")

            # 1スタッフ構造: backup/forwardは不要
            # AlphaTab ScoreTabプロファイルが自動的にTAB段を生成する

        # Remove any zombie start elements (start tags that were never stopped)
        for string_num, (parent_el, child_el) in active_start_elements.items():
            try:
                parent_el.remove(child_el)
            except Exception:
                pass

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


def _assign_to_bars(notes: List[dict], beats: List[float], beats_per_bar: int, bpm: float = 120.0, time_signature: str = "4/4", rhythm_info: dict | None = None) -> List[dict]:
    """Assign each note to a bar and beat position (in divisions) using Universal Quantizer."""
    if not beats or not notes:
        return []

    try:
        from universal_quantizer import quantize_notes_universal
        entries = quantize_notes_universal(
            notes=notes,
            beats=beats,
            bpm=bpm,
            time_signature=time_signature,
            beats_per_bar=beats_per_bar,
        )
        return entries
    except Exception as e:
        import traceback
        traceback.print_exc()
        # フォールバック処理
        divisions = 12
        beats_arr = np.array(beats)
        entries = []
        for note in notes:
            t = float(note["start"])
            idx = int(np.searchsorted(beats_arr, t, side='right')) - 1
            idx = max(0, min(idx, len(beats_arr) - 1))
            bar = idx // beats_per_bar
            beat_in_bar = idx % beats_per_bar
            entries.append({
                "bar": bar,
                "beat_pos": beat_in_bar * divisions,
                "beat_pos_in_bar": beat_in_bar * divisions,
                "beat_pos_absolute": bar * (beats_per_bar * divisions) + beat_in_bar * divisions,
                "duration_divs": 4,
                "pitch": note["pitch"],
                "string": note.get("string", 1),
                "fret": note.get("fret", 0),
                "technique": note.get("technique"),
                "velocity": note.get("velocity", 0.5),
                "start_time": t,
                "is_triplet": False,
            })
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
