"""
score_model.py — PowerTab互換スコアデータモデル
================================================
notes_assigned.json → score.json 変換、
score.json → MusicXML 生成の基盤。
"""
import json, copy, math
from pathlib import Path
from typing import List, Optional


# ─── データ構造定義 ───

def empty_note(string=1, fret=0, pitch=60, velocity=0.8):
    return {
        "string": string, "fret": fret, "pitch": pitch, "velocity": velocity,
        "techniques": {
            "hammer_on": False, "pull_off": False,
            "slide_up": False, "slide_down": False,
            "bend": None, "vibrato": False,
            "let_ring": False, "palm_mute": False,
            "staccato": False, "accent": False,
            "ghost_note": False, "harmonic": None,
            "trill": None, "tremolo_picking": False,
            "tap": False, "slap": False, "pop": False,
        }
    }


def empty_beat(position=0, duration="quarter"):
    return {
        "position": position, "duration": duration,
        "dotted": False, "double_dotted": False,
        "rest": False, "tie": False,
        "triplet": False, "irregular_grouping": None,
        "notes": [],
    }


def empty_bar(bar_number=1, time_signature="4/4", tempo=120):
    return {
        "bar_number": bar_number,
        "time_signature": time_signature,
        "key_signature": "C",
        "tempo": tempo,
        "barline_start": "normal",
        "barline_end": "normal",
        "repeat_count": 0,
        "rehearsal_sign": None,
        "alternate_ending": None,
        "direction": None,
        "chord_text": None,
        "text_items": [],
        "dynamic": None,
        "beats": [],
    }


def empty_score(title="", bpm=120, tuning="standard", time_signature="4/4"):
    return {
        "version": 1,
        "meta": {
            "title": title, "artist": "", "transcriber": "",
            "copyright": "", "bpm": bpm,
            "time_signature": time_signature,
            "key_signature": "C",
            "tuning": tuning, "capo": 0,
        },
        "bars": [],
    }


# ─── ピッチ ⇔ フレット/弦 変換 ───

STANDARD_TUNING = [40, 45, 50, 55, 59, 64]  # E2 A2 D3 G3 B3 E4

def fret_string_to_pitch(string, fret, tuning=None):
    """弦番号(1-6) + フレット → MIDIピッチ"""
    if tuning is None:
        tuning = STANDARD_TUNING
    if 1 <= string <= len(tuning):
        return tuning[string - 1] + fret
    return 60  # fallback

def pitch_to_best_fret_string(pitch, tuning=None, prefer_string=None):
    """MIDIピッチ → 最適な(string, fret)。prefer_stringで弦の優先指定可能。"""
    if tuning is None:
        tuning = STANDARD_TUNING
    candidates = []
    for s_idx, open_pitch in enumerate(tuning):
        fret = pitch - open_pitch
        if 0 <= fret <= 24:
            candidates.append((s_idx + 1, fret))
    if not candidates:
        # 範囲外: 最も近い弦で近似
        best = min(range(len(tuning)), key=lambda i: abs(pitch - tuning[i]))
        return (best + 1, max(0, pitch - tuning[best]))
    if prefer_string:
        for s, f in candidates:
            if s == prefer_string:
                return (s, f)
    # 中間弦を優先（2-5弦）、同じならフレットが低い方
    candidates.sort(key=lambda sf: (abs(sf[0] - 3.5), sf[1]))
    return candidates[0]

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def pitch_to_note_name(pitch):
    """MIDIピッチ → 音名文字列 (例: C4, D#3)"""
    octave = (pitch // 12) - 1
    note = NOTE_NAMES[pitch % 12]
    return f"{note}{octave}"

def note_name_to_pitch(name):
    """音名文字列 → MIDIピッチ (例: C4→60, D#3→51)"""
    name = name.strip().upper()
    if not name:
        return None
    note = name[0]
    rest = name[1:]
    sharp = 0
    if rest.startswith("#") or rest.startswith("♯"):
        sharp = 1; rest = rest[1:]
    elif rest.startswith("B") and rest[1:].lstrip("-").isdigit():
        # Bb notation — but 'B' could be the octave start
        # Only treat as flat if next char exists and is a digit
        sharp = -1; rest = rest[1:]
    elif rest.startswith("♭"):
        sharp = -1; rest = rest[1:]
    try:
        octave = int(rest)
    except ValueError:
        return None
    base_map = {"C":0,"D":2,"E":4,"F":5,"G":7,"A":9,"B":11}
    if note not in base_map:
        return None
    return (octave + 1) * 12 + base_map[note] + sharp


# ─── 旧フォーマット → 新フォーマット変換 ───

TECH_MAP_LEGACY = {
    "h": "hammer_on", "p": "pull_off",
    "/": "slide_up", "\\": "slide_down",
    "b": "bend", "~": "vibrato",
    "palm_mute": "palm_mute", "harmonic": "harmonic",
    "let_ring": "let_ring", "x": "ghost_note",
    "tr": "trill",
}

DURATION_LABELS = {
    48: "whole", 24: "half", 12: "quarter",
    8: "eighth", 6: "eighth", 4: "eighth", 3: "16th",
}


def migrate_from_notes_assigned(
    notes: List[dict],
    beats: List[float],
    bpm: float = 120.0,
    title: str = "",
    tuning: str = "standard",
    time_signature: str = "4/4",
) -> dict:
    """notes_assigned.json + beats → score.json 形式に変換"""
    import numpy as np

    score = empty_score(title=title, bpm=bpm, tuning=tuning, time_signature=time_signature)

    ts_parts = time_signature.split("/")
    beats_per_bar = int(ts_parts[0]) if len(ts_parts) == 2 else 4
    divisions = 12

    if not beats or not notes:
        score["bars"].append(empty_bar(1, time_signature, bpm))
        return score

    beats_arr = np.array(beats)
    sorted_notes = sorted(notes, key=lambda n: (float(n["start"]), int(n.get("pitch", 60))))

    # ノートをバー・ビート位置にマッピング
    bar_data = {}  # bar_num -> list of (beat_pos, note_dict)

    for note in sorted_notes:
        t = float(note["start"])
        idx = int(np.searchsorted(beats_arr, t, side='right')) - 1
        idx = max(0, min(idx, len(beats_arr) - 1))

        bar_num = idx // beats_per_bar
        beat_in_bar = idx % beats_per_bar

        if idx < len(beats_arr):
            beat_time = float(beats_arr[idx])
            next_beat = float(beats_arr[idx + 1]) if idx + 1 < len(beats_arr) else beat_time + 0.5
            frac = (t - beat_time) / (next_beat - beat_time) if next_beat > beat_time else 0.0
            frac = max(0.0, min(frac, 0.99))
            raw_sub = frac * divisions
            grid = [0, 3, 4, 6, 8, 9, 12]
            sub_divs = min(grid, key=lambda x: abs(x - raw_sub))
            if sub_divs == 12:
                sub_divs = 0
                beat_in_bar = (beat_in_bar + 1) % beats_per_bar
                if beat_in_bar == 0:
                    bar_num += 1
        else:
            sub_divs = 0

        position = beat_in_bar * divisions + sub_divs

        # テクニック変換
        tech_legacy = str(note.get("technique") or "normal")
        techniques = {k: False for k in empty_note()["techniques"]}
        techniques["bend"] = None
        techniques["harmonic"] = None
        techniques["trill"] = None
        if tech_legacy in TECH_MAP_LEGACY:
            key = TECH_MAP_LEGACY[tech_legacy]
            if key == "bend":
                techniques["bend"] = {"type": "full", "value": 1.0}
            elif key == "harmonic":
                techniques["harmonic"] = "natural"
            elif key == "trill":
                techniques["trill"] = {"fret": int(note.get("fret", 0)) + 2}
            else:
                techniques[key] = True

        n = {
            "string": int(note.get("string", 1)),
            "fret": int(note.get("fret", 0)),
            "pitch": int(note.get("pitch", 60)),
            "velocity": float(note.get("velocity", 0.8)),
            "techniques": techniques,
        }

        if bar_num not in bar_data:
            bar_data[bar_num] = []
        bar_data[bar_num].append((position, n))

    # バー構築
    total_bars = max(bar_data.keys()) + 1 if bar_data else 1
    for bar_num in range(total_bars):
        bar = empty_bar(bar_num + 1, time_signature, bpm)

        entries = bar_data.get(bar_num, [])
        # ポジションでグループ化
        pos_groups = {}
        for pos, n in entries:
            if pos not in pos_groups:
                pos_groups[pos] = []
            pos_groups[pos].append(n)

        for pos in sorted(pos_groups.keys()):
            notes_at_pos = pos_groups[pos]
            # 音価推定（次のビートとの距離）
            all_positions = sorted(pos_groups.keys())
            idx_in_bar = all_positions.index(pos)
            bar_total = beats_per_bar * divisions
            if idx_in_bar + 1 < len(all_positions):
                dur_divs = all_positions[idx_in_bar + 1] - pos
            else:
                dur_divs = bar_total - pos

            dur_divs = max(1, min(dur_divs, bar_total))
            dur_label = "quarter"
            for threshold, label in sorted(DURATION_LABELS.items(), reverse=True):
                if dur_divs >= threshold:
                    dur_label = label
                    break

            is_triplet = dur_divs in [4, 8]

            beat = empty_beat(position=pos, duration=dur_label)
            beat["triplet"] = is_triplet
            beat["notes"] = notes_at_pos
            bar["beats"].append(beat)

        score["bars"].append(bar)

    return score


# ─── score.json → MusicXML 変換 ───
# (tab_renderer.pyの拡張版。Phase 2以降で段階的に機能追加)

def score_to_musicxml(score: dict, tuning_midi: list = None, noise_gate: float = 0.0) -> str:
    """score.json → MusicXML文字列"""
    import xml.etree.ElementTree as ET

    if tuning_midi is None:
        tuning_midi = [40, 45, 50, 55, 59, 64]

    meta = score.get("meta", {})
    bars = score.get("bars", [])
    bpm = meta.get("bpm", 120)
    ts = meta.get("time_signature", "4/4")
    ts_parts = ts.split("/")
    beats_per_bar = int(ts_parts[0])
    beat_type = int(ts_parts[1]) if len(ts_parts) == 2 else 4
    divisions = 12
    title = meta.get("title", "Guitar TAB")

    root = ET.Element("score-partwise", version="4.0")
    work = ET.SubElement(root, "work")
    ET.SubElement(work, "work-title").text = title

    part_list = ET.SubElement(root, "part-list")
    sp = ET.SubElement(part_list, "score-part", id="P1")
    ET.SubElement(sp, "part-name").text = "Guitar"
    si = ET.SubElement(sp, "score-instrument", id="P1-I1")
    ET.SubElement(si, "instrument-name").text = "Acoustic Guitar (steel)"
    mi = ET.SubElement(sp, "midi-instrument", id="P1-I1")
    ET.SubElement(mi, "midi-channel").text = "1"
    ET.SubElement(mi, "midi-program").text = "26"
    ET.SubElement(mi, "volume").text = "80"

    part = ET.SubElement(root, "part", id="P1")

    for bar_idx, bar in enumerate(bars):
        measure = ET.SubElement(part, "measure", number=str(bar.get("bar_number", bar_idx + 1)))

        # 最初の小節 or 拍子変更時にattributes
        bar_ts = bar.get("time_signature", ts)
        bar_ts_parts = bar_ts.split("/")
        bar_bpb = int(bar_ts_parts[0])
        bar_bt = int(bar_ts_parts[1]) if len(bar_ts_parts) == 2 else 4

        if bar_idx == 0 or bar_ts != ts:
            attrs = ET.SubElement(measure, "attributes")
            ET.SubElement(attrs, "divisions").text = str(divisions)
            time_el = ET.SubElement(attrs, "time")
            ET.SubElement(time_el, "beats").text = str(bar_bpb)
            ET.SubElement(time_el, "beat-type").text = str(bar_bt)

            if bar_idx == 0:
                clef = ET.SubElement(attrs, "clef")
                ET.SubElement(clef, "sign").text = "TAB"
                ET.SubElement(clef, "line").text = "5"
                sd = ET.SubElement(attrs, "staff-details")
                ET.SubElement(sd, "staff-lines").text = "6"
                steps = ["C","C","D","D","E","F","F","G","G","A","A","B"]
                for i in range(6):
                    st = ET.SubElement(sd, "staff-tuning", line=str(i + 1))
                    ET.SubElement(st, "tuning-step").text = steps[tuning_midi[i] % 12]
                    ET.SubElement(st, "tuning-octave").text = str((tuning_midi[i] // 12) - 1)

        # テンポ
        bar_tempo = bar.get("tempo", bpm)
        if bar_idx == 0 or bar_tempo != bpm:
            direction = ET.SubElement(measure, "direction", placement="above")
            dt = ET.SubElement(direction, "direction-type")
            metro = ET.SubElement(dt, "metronome")
            ET.SubElement(metro, "beat-unit").text = "quarter"
            ET.SubElement(metro, "per-minute").text = str(int(bar_tempo))
            ET.SubElement(direction, "sound", tempo=str(int(bar_tempo)))

        # リハーサルマーク
        if bar.get("rehearsal_sign"):
            d = ET.SubElement(measure, "direction", placement="above")
            dt2 = ET.SubElement(d, "direction-type")
            reh = ET.SubElement(dt2, "rehearsal")
            reh.text = bar["rehearsal_sign"]

        # コードネーム
        if bar.get("chord_text"):
            _add_chord_harmony(measure, bar["chord_text"])

        # ダイナミクス
        if bar.get("dynamic"):
            d = ET.SubElement(measure, "direction", placement="below")
            dt2 = ET.SubElement(d, "direction-type")
            dyn = ET.SubElement(dt2, "dynamics")
            ET.SubElement(dyn, bar["dynamic"])

        # テキスト
        for text_item in bar.get("text_items", []):
            d = ET.SubElement(measure, "direction", placement="above")
            dt2 = ET.SubElement(d, "direction-type")
            w = ET.SubElement(dt2, "words")
            w.text = text_item

        # ビート/ノート
        beat_list = bar.get("beats", [])
        if not beat_list:
            note_el = ET.SubElement(measure, "note")
            ET.SubElement(note_el, "rest")
            ET.SubElement(note_el, "duration").text = str(divisions * bar_bpb)
            ET.SubElement(note_el, "type").text = "whole"
        else:
            current_pos = 0
            bar_total = divisions * bar_bpb

            for b_idx, beat in enumerate(beat_list):
                target_pos = int(beat.get("position", 0))

                # ギャップに休符
                gap = target_pos - current_pos
                if gap > 0:
                    r = ET.SubElement(measure, "note")
                    ET.SubElement(r, "rest")
                    ET.SubElement(r, "duration").text = str(gap)
                    ET.SubElement(r, "type").text = _dur_type(gap, divisions)
                    current_pos = target_pos

                if beat.get("rest"):
                    next_pos = beat_list[b_idx + 1]["position"] if b_idx + 1 < len(beat_list) else bar_total
                    dur = next_pos - target_pos
                    r = ET.SubElement(measure, "note")
                    ET.SubElement(r, "rest")
                    ET.SubElement(r, "duration").text = str(max(1, dur))
                    ET.SubElement(r, "type").text = beat.get("duration", "quarter")
                    current_pos = next_pos
                    continue

                # 次のビートまでの距離 = 音価
                if b_idx + 1 < len(beat_list):
                    dur = int(beat_list[b_idx + 1]["position"]) - target_pos
                else:
                    dur = bar_total - target_pos
                dur = max(1, min(dur, bar_total - target_pos))

                notes_in_beat = beat.get("notes", [])
                for n_idx, n in enumerate(notes_in_beat):
                    vel = float(n.get("velocity", 0.8))
                    if noise_gate > 0 and vel < noise_gate:
                        continue

                    note_el = ET.SubElement(measure, "note")
                    if n_idx > 0:
                        ET.SubElement(note_el, "chord")

                    pitch_el = ET.SubElement(note_el, "pitch")
                    p = int(n["pitch"])
                    steps = ["C","C","D","D","E","F","F","G","G","A","A","B"]
                    alters = [0,1,0,1,0,0,1,0,1,0,1,0]
                    ET.SubElement(pitch_el, "step").text = steps[p % 12]
                    if alters[p % 12] != 0:
                        ET.SubElement(pitch_el, "alter").text = str(alters[p % 12])
                    ET.SubElement(pitch_el, "octave").text = str((p // 12) - 1)

                    ET.SubElement(note_el, "duration").text = str(dur)
                    ET.SubElement(note_el, "type").text = _dur_type(dur, divisions)

                    if beat.get("triplet") and dur in [4, 8]:
                        tm = ET.SubElement(note_el, "time-modification")
                        ET.SubElement(tm, "actual-notes").text = "3"
                        ET.SubElement(tm, "normal-notes").text = "2"

                    ET.SubElement(note_el, "stem").text = "none"

                    notations = ET.SubElement(note_el, "notations")
                    technical = ET.SubElement(notations, "technical")
                    ET.SubElement(technical, "string").text = str(n.get("string", 1))
                    ET.SubElement(technical, "fret").text = str(n.get("fret", 0))

                    # テクニック
                    techs = n.get("techniques", {})
                    if techs.get("hammer_on"):
                        ho = ET.SubElement(technical, "hammer-on", type="start"); ho.text = "H"
                    if techs.get("pull_off"):
                        po = ET.SubElement(technical, "pull-off", type="start"); po.text = "P"
                    if techs.get("slide_up"):
                        ET.SubElement(notations, "slide", type="start")
                    if techs.get("slide_down"):
                        ET.SubElement(notations, "slide", type="start")
                    if techs.get("bend"):
                        bend_el = ET.SubElement(technical, "bend")
                        val = techs["bend"].get("value", 1.0) if isinstance(techs["bend"], dict) else 1.0
                        ET.SubElement(bend_el, "bend-alter").text = str(int(val * 2))
                    if techs.get("vibrato"):
                        orn = ET.SubElement(notations, "ornaments")
                        ET.SubElement(orn, "wavy-line", type="start")
                    if techs.get("harmonic"):
                        ET.SubElement(technical, "harmonic")
                    if techs.get("ghost_note"):
                        nh = ET.SubElement(note_el, "notehead"); nh.text = "x"
                    if techs.get("palm_mute"):
                        pass  # MusicXMLには直接的な表現がない

                current_pos = target_pos + dur

            # 残りを埋める
            remaining = bar_total - current_pos
            if remaining > 0:
                fwd = ET.SubElement(measure, "forward")
                ET.SubElement(fwd, "duration").text = str(remaining)

        # 小節線
        if bar.get("barline_end") and bar["barline_end"] != "normal":
            _add_barline(measure, bar["barline_end"], "right")

    xml_str = ET.tostring(root, encoding="unicode")
    header = '<?xml version="1.0" encoding="UTF-8"?>\n'
    header += '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
    header += '"http://www.musicxml.org/dtds/partwise.dtd">\n'
    return header + xml_str


# ─── ユーティリティ ───

def _dur_type(dur_divs, divisions=12):
    if dur_divs >= 48: return "whole"
    if dur_divs >= 24: return "half"
    if dur_divs >= 12: return "quarter"
    if dur_divs >= 6: return "eighth"
    if dur_divs >= 3: return "16th"
    return "32nd"


def _add_chord_harmony(measure, chord_text):
    import xml.etree.ElementTree as ET
    if not chord_text or chord_text == "N.C.":
        return
    harmony = ET.SubElement(measure, "harmony")
    root_el = ET.SubElement(harmony, "root")
    root_step = chord_text[0]
    alter = 0
    rest = chord_text[1:]
    if rest.startswith("#"):
        alter = 1; rest = rest[1:]
    elif rest.startswith("b"):
        alter = -1; rest = rest[1:]
    kind_map = {"": "major", "m": "minor", "7": "dominant", "m7": "minor-seventh",
                "maj7": "major-seventh", "dim": "diminished", "aug": "augmented",
                "sus4": "suspended-fourth", "sus2": "suspended-second"}
    kind = kind_map.get(rest, "major")
    ET.SubElement(root_el, "root-step").text = root_step
    if alter != 0:
        ET.SubElement(root_el, "root-alter").text = str(alter)
    ET.SubElement(harmony, "kind").text = kind


def _add_barline(measure, style, location="right"):
    import xml.etree.ElementTree as ET
    barline = ET.SubElement(measure, "barline", location=location)
    style_map = {
        "double": "light-light", "final": "light-heavy",
        "repeat_start": "heavy-light", "repeat_end": "light-heavy",
    }
    ET.SubElement(barline, "bar-style").text = style_map.get(style, "regular")
    if "repeat" in style:
        direction = "forward" if style == "repeat_start" else "backward"
        ET.SubElement(barline, "repeat", direction=direction)
