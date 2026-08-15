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
                bar_notes.sort(key=lambda e: float(e["beat_pos"]))
                groups: List[List[dict]] = _group_by_time(bar_notes, threshold=0.1)

                current_pos: int = 0
                for group_idx, group in enumerate(groups):
                    target_pos: int = int(float(group[0]["beat_pos"]))
                    gap: int = target_pos - current_pos
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
                        current_pos = target_pos

                    next_target: int = bar_total if group_idx + 1 >= len(groups) else int(float(groups[group_idx + 1][0]["beat_pos"]))
                    gap_to_next: int = max(1, min(next_target - target_pos, bar_total - target_pos))

                    for i, entry in enumerate(group):
                        dur = int(entry.get("duration_divs", gap_to_next))
                        dur = min(dur, gap_to_next, bar_total - target_pos)
                        
                        tech = str(entry.get("technique") or "normal")
                        string_num = int(entry.get("string", 1))
                        fret_val = int(entry.get("fret", 0))
                        pitch = int(entry["pitch"])
                        
                        # Body Hit (bh) Handling: clamp pitch to lowest open string (usually 6th string, tuning[0])
                        is_body_hit = (tech in ("bh", "body_hit"))
                        if is_body_hit:
                            # Clamp to lowest tuning pitch
                            pitch = tuning[0] if tuning else 40
                            fret_val = 0
                        
                        note_el = ET.SubElement(measure, "note")
                        if i > 0: ET.SubElement(note_el, "chord")
                        pitch_el = ET.SubElement(note_el, "pitch")
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
                        
                        is_trip = entry.get("is_triplet", False) or (is_triplet_mode and dur in [2, 4, 8])
                        if is_trip:
                            tm = ET.SubElement(note_el, "time-modification")
                            ET.SubElement(tm, "actual-notes").text = "3"
                            ET.SubElement(tm, "normal-notes").text = "2"
                        if entry.get("_tie_start"):
                            ET.SubElement(note_el, "tie", type="start")
                        ET.SubElement(note_el, "stem").text = "up" if voice == "1" else "down"

                        # Special noteheads (Body hit / nail attack / dead note)
                        if is_body_hit:
                            ET.SubElement(note_el, "notehead").text = "circle-x"
                        elif tech in ("na", "nail_attack"):
                            ET.SubElement(note_el, "notehead").text = "x"
                        elif tech == "x":
                            ET.SubElement(note_el, "notehead").text = "x"

                        # notations: fret/string + technique
                        notations = ET.SubElement(note_el, "notations")
                        if entry.get("_tie_start"):
                            ET.SubElement(notations, "tied", type="start")

                        # Tuplet brackets across all measures
                        if is_trip and i == 0:
                            t_role = entry.get("tuplet_role", "none")
                            if t_role == "start":
                                ET.SubElement(notations, "tuplet", type="start", bracket="yes")
                            elif t_role == "stop":
                                ET.SubElement(notations, "tuplet", type="stop")
                            elif t_role == "start_stop":
                                ET.SubElement(notations, "tuplet", type="start", bracket="yes")
                                ET.SubElement(notations, "tuplet", type="stop")
                            elif t_role == "none":
                                # Fallback: beat modulo calculation
                                cycle = 12
                                rem = target_pos % cycle
                                if rem == 0:
                                    ET.SubElement(notations, "tuplet", type="start", bracket="yes")
                                elif rem >= 8:
                                    ET.SubElement(notations, "tuplet", type="stop")

                        # --- PREVIOUS TECHNIQUE STOP (type="stop") ---
                        if string_num in active_slurs:
                            prev_slur = active_slurs[string_num]
                            technical_stop = ET.SubElement(notations, "technical")
                            if prev_slur == "h":
                                ET.SubElement(technical_stop, "hammer-on", type="stop")
                            elif prev_slur == "p":
                                ET.SubElement(technical_stop, "pull-off", type="stop")
                            del active_slurs[string_num]
                            if string_num in active_start_elements:
                                del active_start_elements[string_num]

                        if string_num in active_slides:
                            ET.SubElement(notations, "slide", type="stop")
                            del active_slides[string_num]
                            if string_num in active_start_elements:
                                del active_start_elements[string_num]

                        if string_num in active_gliss:
                            ET.SubElement(notations, "glissando", type="stop")
                            del active_gliss[string_num]
                            if string_num in active_start_elements:
                                del active_start_elements[string_num]

                        technical = ET.SubElement(notations, "technical")
                        ET.SubElement(technical, "string").text = str(string_num)
                        ET.SubElement(technical, "fret").text = str(fret_val)

                        # Pluck direction (down-bow / up-bow)
                        pluck_dir = entry.get("pluck_direction")
                        if pluck_dir == "down":
                            ET.SubElement(technical, "down-bow")
                        elif pluck_dir == "up":
                            ET.SubElement(technical, "up-bow")

                        # Left-hand fingering (1=index, 2=middle, 3=ring, 4=pinky)
                        finger = entry.get("finger") or entry.get("left_hand_finger")
                        if finger is not None and finger > 0:
                            ET.SubElement(technical, "fingering").text = str(finger)

                        # Right-hand plucking finger (PIMA)
                        r_finger = entry.get("r_finger") or entry.get("pluck")
                        if r_finger is not None:
                            pima_map = {1: 'p', 2: 'i', 3: 'm', 4: 'a'}
                            r_finger_str = pima_map.get(r_finger, str(r_finger))
                            if r_finger_str in ('p', 'i', 'm', 'a'):
                                ET.SubElement(technical, "pluck").text = r_finger_str

                        # --- CURRENT TECHNIQUE START (type="start" / single notations) ---
                        if tech == "h":
                            ho = ET.SubElement(technical, "hammer-on", type="start"); ho.text = "H"
                            technique_map.append("hammer_on")
                            active_slurs[string_num] = "h"
                            active_start_elements[string_num] = (technical, ho)
                        elif tech == "p":
                            po = ET.SubElement(technical, "pull-off", type="start"); po.text = "P"
                            technique_map.append("pull_off")
                            active_slurs[string_num] = "p"
                            active_start_elements[string_num] = (technical, po)
                        elif tech == "/":
                            sl = ET.SubElement(notations, "slide", type="start", **{"line-type": "solid"})
                            technique_map.append("slide_up")
                            active_slides[string_num] = "/"
                            active_start_elements[string_num] = (notations, sl)
                        elif tech == "\\":
                            sl = ET.SubElement(notations, "slide", type="start", **{"line-type": "solid"})
                            technique_map.append("slide_down")
                            active_slides[string_num] = "\\"
                            active_start_elements[string_num] = (notations, sl)
                        elif tech == "gliss_up":
                            gl = ET.SubElement(notations, "glissando", type="start", **{"line-type": "wavy"})
                            gl.text = "gliss."
                            technique_map.append("gliss_up")
                            active_gliss[string_num] = "gliss_up"
                            active_start_elements[string_num] = (notations, gl)
                        elif tech == "gliss_down":
                            gl = ET.SubElement(notations, "glissando", type="start", **{"line-type": "wavy"})
                            gl.text = "gliss."
                            technique_map.append("gliss_down")
                            active_gliss[string_num] = "gliss_down"
                            active_start_elements[string_num] = (notations, gl)
                        elif tech in ("ah", "artificial_harmonic"):
                            harmonic_el = ET.SubElement(technical, "harmonic")
                            ET.SubElement(harmonic_el, "artificial")
                            technique_map.append("harmonic")
                        elif tech in ("th", "tapped_harmonic"):
                            harmonic_el = ET.SubElement(technical, "harmonic")
                            ET.SubElement(harmonic_el, "tapped")
                            # Add text direction above
                            dir_el = ET.SubElement(measure, "direction", placement="above")
                            dt = ET.SubElement(dir_el, "direction-type")
                            words = ET.SubElement(dt, "words", **{"font-style": "italic", "font-weight": "bold", "font-size": "8"})
                            words.text = "T.H."
                            technique_map.append("harmonic")
                        elif tech in ("na", "nail_attack"):
                            dir_el = ET.SubElement(measure, "direction", placement="above")
                            dt = ET.SubElement(dir_el, "direction-type")
                            words = ET.SubElement(dt, "words", **{"font-style": "italic", "font-size": "7"})
                            words.text = "N.A."
                            technique_map.append("ghost_note")
                        elif is_body_hit:
                            dir_el = ET.SubElement(measure, "direction", placement="above")
                            dt = ET.SubElement(dir_el, "direction-type")
                            words = ET.SubElement(dt, "words", **{"font-style": "italic", "font-weight": "bold", "font-size": "8"})
                            words.text = "Body"
                            technique_map.append("ghost_note")
                        elif tech == "palm_mute" or tech == "pm":
                            technique_map.append("palm_mute")
                        elif tech == "harmonic":
                            ET.SubElement(technical, "harmonic"); technique_map.append("harmonic")
                        elif tech == "b":
                            bend_el = ET.SubElement(technical, "bend")
                            ET.SubElement(bend_el, "bend-alter").text = "2"; technique_map.append("bend")
                        elif tech == "pre_bend":
                            bend_el = ET.SubElement(technical, "bend")
                            ET.SubElement(bend_el, "pre-bend")
                            ET.SubElement(bend_el, "bend-alter").text = "2"; technique_map.append("bend")
                        elif tech == "release_bend":
                            bend_el = ET.SubElement(technical, "bend")
                            ET.SubElement(bend_el, "release")
                            ET.SubElement(bend_el, "bend-alter").text = "2"; technique_map.append("bend")
                        elif tech == "bend_release":
                            bend_el = ET.SubElement(technical, "bend")
                            ET.SubElement(bend_el, "bend-alter").text = "2"
                            ET.SubElement(bend_el, "release"); technique_map.append("bend")
                        elif tech == "quarter_bend":
                            bend_el = ET.SubElement(technical, "bend")
                            ET.SubElement(bend_el, "bend-alter").text = "0.5"; technique_map.append("bend")
                        elif tech == "slide_in":
                            ET.SubElement(notations, "slide", type="start", **{"line-type": "solid"})
                            technique_map.append("slide_up")
                        elif tech == "slide_out":
                            ET.SubElement(notations, "slide", type="start", **{"line-type": "solid"})
                            technique_map.append("slide_down")
                        elif tech == "arpeggio":
                            ET.SubElement(notations, "arpeggiate")
                            technique_map.append("arpeggio")
                        elif tech == "tremolo":
                            ornaments = notations.find("ornaments")
                            if ornaments is None: ornaments = ET.SubElement(notations, "ornaments")
                            ET.SubElement(ornaments, "tremolo", type="single").text = "3"
                            technique_map.append("tremolo")
                        elif tech == "vibrato":
                            ornaments = notations.find("ornaments")
                            if ornaments is None: ornaments = ET.SubElement(notations, "ornaments")
                            ET.SubElement(ornaments, "wavy-line", type="start"); technique_map.append("vibrato")
                        elif tech == "x":
                            technique_map.append("ghost_note")
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
