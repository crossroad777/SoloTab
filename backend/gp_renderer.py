"""
gp_renderer.py — Guitar Pro 5 (.gp5) 形式でTAB譜を生成
=========================================================
AlphaTabのネイティブ形式であるGP5を直接生成することで、
MusicXMLパース由来の表示問題を根本的に解消する。

生成された.gp5ファイルは以下で利用可能:
  - AlphaTab (Web UI内表示)
  - TuxGuitar (無料、人間による修正)
  - Guitar Pro / MuseScore 等
"""
from __future__ import annotations
from typing import List, Optional
import guitarpro as gp


# divisions per quarter note (triplet grid: 12 = LCM(4,3))
DIVISIONS = 12


def notes_to_gp5(notes: List[dict], *,
                 beats: List[float],
                 bpm: float = 120.0,
                 title: str = "Guitar TAB",
                 tuning: list | None = None,
                 time_signature: str = "4/4",
                 noise_gate: float = 0.0,
                 rhythm_info: dict | None = None,
                 key_signature: str = "C",
                 **kwargs) -> bytes:
    """
    ノートデータからGP5バイナリを生成する。

    Parameters
    ----------
    notes : list[dict]
        Keys: start, end, pitch, string, fret, velocity, technique
    beats : list[float]
        ビート時刻(秒)
    bpm : float
    title : str
    tuning : list[int]  [6th→1st] のMIDIノート番号
    time_signature : str  "3/4", "4/4", "6/8"
    rhythm_info : dict  {"subdivision": "triplet"|"straight", ...}

    Returns
    -------
    bytes : GP5バイナリデータ
    """
    if tuning is None:
        tuning = [40, 45, 50, 55, 59, 64]

    # Parse time signature
    beats_per_bar, beat_type = _parse_time_sig(time_signature)
    is_triplet = (rhythm_info or {}).get("subdivision") == "triplet"

    # Noise gate filter
    filtered = _filter_noise(notes, noise_gate)

    # Assign notes to bars (reuse tab_renderer logic)
    from tab_renderer import _assign_to_bars, _group_by_time
    note_entries = _assign_to_bars(filtered, beats, beats_per_bar, rhythm_info=rhythm_info)

    # Quantize durations
    try:
        from music_theory import quantize_note_durations
        note_entries = quantize_note_durations(
            note_entries, is_triplet_mode=is_triplet, beats_per_bar=beats_per_bar
        )
    except Exception:
        pass

    # Calculate total bars
    total_bars = 1
    if note_entries:
        total_bars = max(int(e["bar"]) for e in note_entries) + 1
    elif beats:
        total_bars = max(1, len(beats) // beats_per_bar)
    total_bars = max(total_bars, 1)

    # --- Build GP5 Song ---
    song = gp.Song()
    song.title = title
    song.artist = "SoloTab"
    song.tempo = int(bpm)

    # Track setup
    track = song.tracks[0]
    track.name = "Guitar"
    track.channel.instrument = 25  # Acoustic Guitar (steel)
    track.strings = [
        gp.GuitarString(number=i + 1, value=tuning[5 - i])
        for i in range(6)
    ]  # GP format: string 1 = highest (E4), string 6 = lowest (E2)

    # Key signature
    key_fifths = _key_to_fifths(key_signature)

    # --- Measure Headers ---
    # First measure header already exists, configure it
    mh0 = song.measureHeaders[0]
    mh0.timeSignature.numerator = beats_per_bar
    mh0.timeSignature.denominator.value = _beat_type_to_gp_dur(beat_type)
    mh0.keySignature = _fifths_to_gp_key(key_fifths)
    if is_triplet:
        mh0.tripletFeel = gp.TripletFeel.eighth

    # Add remaining measure headers
    for bar_num in range(1, total_bars):
        mh = gp.MeasureHeader()
        mh.number = bar_num + 1
        mh.start = mh0.start + bar_num * _bar_length(beats_per_bar, beat_type)
        mh.timeSignature.numerator = beats_per_bar
        mh.timeSignature.denominator.value = _beat_type_to_gp_dur(beat_type)
        mh.keySignature = _fifths_to_gp_key(key_fifths)
        if is_triplet:
            mh.tripletFeel = gp.TripletFeel.eighth
        song.measureHeaders.append(mh)

    # --- Build Measures ---
    # First measure already exists
    measures = [track.measures[0]]
    for bar_num in range(1, total_bars):
        m = gp.Measure(track, song.measureHeaders[bar_num])
        measures.append(m)
    track.measures = measures

    # --- Fill each measure with notes ---
    bar_total_divs = beats_per_bar * DIVISIONS  # e.g. 36 for 3/4

    for bar_num in range(total_bars):
        m = track.measures[bar_num]
        voice = m.voices[0]

        bar_notes = [e for e in note_entries if e["bar"] == bar_num]

        if not bar_notes:
            # Empty bar: whole rest
            rest_beat = gp.Beat(voice, status=gp.BeatStatus.rest)
            rest_beat.duration.value = gp.Duration.whole
            voice.beats = [rest_beat]
            continue

        bar_notes.sort(key=lambda e: float(e["beat_pos"]))
        groups = _group_by_time(bar_notes, threshold=0.1)

        gp_beats = []
        current_pos = 0

        for group_idx, group in enumerate(groups):
            target_pos = int(float(group[0]["beat_pos"]))

            # Rest gap before this group
            gap = target_pos - current_pos
            if gap > 0:
                rest_beats = _divs_to_gp_beats_rest(gap, voice, is_triplet)
                gp_beats.extend(rest_beats)
                current_pos = target_pos

            # Note duration
            next_target = bar_total_divs if group_idx + 1 >= len(groups) \
                else int(float(groups[group_idx + 1][0]["beat_pos"]))
            gap_to_next = max(1, min(next_target - target_pos,
                                     bar_total_divs - target_pos))

            # Get duration from quantized value
            dur_divs = int(group[0].get("duration_divs", gap_to_next))
            dur_divs = min(dur_divs, gap_to_next, bar_total_divs - target_pos)
            dur_divs = max(1, dur_divs)

            # Create beat with all notes in this chord group
            beat = gp.Beat(voice, status=gp.BeatStatus.normal)
            gp_dur, gp_dotted, gp_tuplet = _divs_to_gp_duration(dur_divs, is_triplet)
            beat.duration.value = gp_dur
            beat.duration.isDotted = gp_dotted
            if gp_tuplet:
                beat.duration.tuplet = gp.Tuplet(enters=3, times=2)

            for entry in group:
                string_num = int(entry.get("string", 1))
                fret = int(entry.get("fret", 0))
                note = gp.Note(beat)
                note.value = fret
                note.string = string_num
                note.velocity = _vel_to_gp(entry.get("velocity", 0.5))

                # Technique effects
                tech = entry.get("technique")
                if tech == "h":
                    note.effect.hammer = True
                elif tech == "p":
                    note.effect.hammer = True  # GP uses same flag for pull-off
                elif tech in ("/", "\\"):
                    note.effect.slides.append(gp.SlideType.shiftSlideTo)
                elif tech == "harmonic":
                    note.effect.harmonic = gp.NaturalHarmonic()

                beat.notes.append(note)

            gp_beats.append(beat)
            current_pos = target_pos + dur_divs

        # Trailing rest
        remaining = bar_total_divs - current_pos
        if remaining > 0:
            rest_beats = _divs_to_gp_beats_rest(remaining, voice, is_triplet)
            gp_beats.extend(rest_beats)

        voice.beats = gp_beats if gp_beats else []

    # --- Write to bytes ---
    import io
    buf = io.BytesIO()
    gp.write(song, buf)
    return buf.getvalue()


# ─── Helper Functions ───

def _parse_time_sig(ts: str) -> tuple[int, int]:
    if ts == "3/4":
        return 3, 4
    elif ts == "6/8":
        return 6, 8
    return 4, 4


def _filter_noise(notes, gate):
    if gate <= 0:
        return notes.copy()
    filtered = []
    for n in notes:
        v = float(n.get("velocity", 0.5))
        if v > 1.0:
            v /= 127.0
        if v >= gate:
            filtered.append(n)
    return filtered if filtered else [max(notes, key=lambda x: float(x.get("velocity", 0)))]


def _key_to_fifths(key: str) -> int:
    m = {"C": 0, "Am": 0, "G": 1, "Em": 1, "D": 2, "Bm": 2,
         "A": 3, "E": 4, "B": 5, "F": -1, "Dm": -1,
         "Bb": -2, "Gm": -2, "Eb": -3, "Ab": -4}
    return m.get(key, 0)


def _fifths_to_gp_key(fifths: int) -> gp.KeySignature:
    """fifths値からGPのKeySignatureを返す。"""
    mapping = {
        -4: gp.KeySignature.AMajorFlat,
        -3: gp.KeySignature.EMajorFlat,
        -2: gp.KeySignature.BMajorFlat,
        -1: gp.KeySignature.FMajor,
        0: gp.KeySignature.CMajor,
        1: gp.KeySignature.GMajor,
        2: gp.KeySignature.DMajor,
        3: gp.KeySignature.AMajor,
        4: gp.KeySignature.EMajor,
        5: gp.KeySignature.BMajor,
    }
    return mapping.get(fifths, gp.KeySignature.CMajor)


def _beat_type_to_gp_dur(beat_type: int) -> int:
    return {1: gp.Duration.whole, 2: gp.Duration.half,
            4: gp.Duration.quarter, 8: gp.Duration.eighth,
            16: gp.Duration.sixteenth}.get(beat_type, gp.Duration.quarter)


def _bar_length(beats_per_bar: int, beat_type: int) -> int:
    """GP internal tick length of one bar. Quarter note = 960 ticks."""
    quarter_ticks = 960
    beat_ticks = quarter_ticks * 4 // beat_type
    return beats_per_bar * beat_ticks


def _divs_to_gp_duration(divs: int, is_triplet: bool) -> tuple[int, bool, bool]:
    """
    divisions値 (12=quarter) をGP Duration value, isDotted, isTripletに変換。

    DIVISIONS=12 mapping:
      48 = whole, 24 = half, 12 = quarter, 6 = eighth, 3 = sixteenth
      Dotted: 9 = dotted-eighth, 18 = dotted-quarter, 36 = dotted-half
      Triplet: 4 = triplet-eighth, 8 = triplet-quarter
    """
    # Exact matches first
    exact = {
        48: (gp.Duration.whole, False, False),
        36: (gp.Duration.half, True, False),        # dotted half
        24: (gp.Duration.half, False, False),
        18: (gp.Duration.quarter, True, False),     # dotted quarter
        12: (gp.Duration.quarter, False, False),
        9:  (gp.Duration.eighth, True, False),      # dotted eighth
        8:  (gp.Duration.quarter, False, True),     # triplet quarter
        6:  (gp.Duration.eighth, False, False),
        4:  (gp.Duration.eighth, False, True),      # triplet eighth
        3:  (gp.Duration.sixteenth, False, False),
        2:  (gp.Duration.thirtySecond, False, False),
        1:  (gp.Duration.sixtyFourth, False, False),
    }
    if divs in exact:
        return exact[divs]

    # Nearest match
    best_key = min(exact.keys(), key=lambda k: abs(k - divs))
    return exact[best_key]


def _divs_to_gp_beats_rest(divs: int, voice, is_triplet: bool) -> list:
    """Rest duration expressed as one or more GP rest beats."""
    beats_out = []
    remaining = divs

    # Decompose into standard durations (largest first)
    std_durs = [48, 36, 24, 18, 12, 9, 8, 6, 4, 3, 2, 1]
    if is_triplet:
        # Prefer triplet grid
        std_durs = [48, 24, 12, 8, 4, 3, 1]

    while remaining > 0:
        best = 1
        for d in std_durs:
            if d <= remaining:
                best = d
                break
        gp_dur, gp_dot, gp_trip = _divs_to_gp_duration(best, is_triplet)
        rb = gp.Beat(voice, status=gp.BeatStatus.rest)
        rb.duration.value = gp_dur
        rb.duration.isDotted = gp_dot
        if gp_trip:
            rb.duration.tuplet = gp.Tuplet(enters=3, times=2)
        beats_out.append(rb)
        remaining -= best

    return beats_out


def _vel_to_gp(v) -> int:
    """velocity (0-1 or 0-127) to GP velocity (ppp=15 ... fff=127)."""
    v = float(v)
    if v <= 1.0:
        v = v * 127
    return max(15, min(127, int(v)))
