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
    # ギターTABではフレット番号が情報の主体であり、
    # キー検出の不正確さ（例：Am楽曲がE majorと判定される）による
    # 誤った調号表示を防ぐため、Cメジャー（調号なし）に固定する。
    key_fifths = 0  # C major = 調号なし

    # --- Measure Headers ---
    # First measure header already exists, configure it
    mh0 = song.measureHeaders[0]
    mh0.timeSignature.numerator = beats_per_bar
    mh0.timeSignature.denominator.value = _beat_type_to_gp_dur(beat_type)
    mh0.keySignature = _fifths_to_gp_key(key_fifths)
    # Note: tripletFeel (shuffle) は使わない
    # 実際の三連符は個別ノートの tuplet (3:2) で表現する

    # Add remaining measure headers
    for bar_num in range(1, total_bars):
        mh = gp.MeasureHeader()
        mh.number = bar_num + 1
        mh.start = mh0.start + bar_num * _bar_length(beats_per_bar, beat_type)
        mh.timeSignature.numerator = beats_per_bar
        mh.timeSignature.denominator.value = _beat_type_to_gp_dur(beat_type)
        mh.keySignature = _fifths_to_gp_key(key_fifths)
        # tripletFeel は個別ノートの tuplet で代替
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

    # Voice分離 (split_pitch = 52 は E3)
    split_pitch = 52

    # --- Pre-pass: ベース音(Voice 2)の後処理 ---
    # 1) 各小節のベース音をbeat_pos=0にスナップ（各小節の最初のベース音のみ保持）
    # 2) ベース音が欠落している小節に、前の小節のベース音を引き継ぎ補完
    bars_data = []
    for bar_num in range(total_bars):
        bar_notes = [e for e in note_entries if e["bar"] == bar_num]
        melody = [n for n in bar_notes if int(n.get("pitch", 60)) > split_pitch]
        bass = [n for n in bar_notes if int(n.get("pitch", 60)) <= split_pitch]
        bars_data.append({"melody": melody, "bass": bass})

    # ベース音スナップ＋補完
    last_bass_template = None
    for bar_num in range(total_bars):
        bd = bars_data[bar_num]
        if bd["bass"]:
            # 小節内の最初のベース音を1拍目にスナップ
            # 同一pitch のベース音が複数ある場合は1つに統合
            seen_pitches = set()
            snapped = []
            for b in sorted(bd["bass"], key=lambda x: float(x.get("beat_pos", 0))):
                p = int(b.get("pitch", 60))
                if p not in seen_pitches:
                    seen_pitches.add(p)
                    snap = dict(b)
                    snap["beat_pos"] = 0
                    snapped.append(snap)
            bd["bass"] = snapped
            last_bass_template = snapped
        else:
            # ベース音なし → 前の小節のベース音を引き継ぎ
            if last_bass_template and bd["melody"]:
                bd["bass"] = [dict(t) for t in last_bass_template]

    for bar_num in range(total_bars):
        m = track.measures[bar_num]
        bd = bars_data[bar_num]
        melody = bd["melody"]
        bass = bd["bass"]

        if not melody and not bass:
            # Empty bar: whole rest
            rest_beat = gp.Beat(m.voices[0], status=gp.BeatStatus.rest)
            rest_beat.duration.value = gp.Duration.whole
            m.voices[0].beats = [rest_beat]
            continue

        # Voice 1 (Melody)
        if melody:
            groups1 = _group_by_time(melody, threshold=0.1)
            m.voices[0].beats = _build_voice_beats(
                groups1, m.voices[0], bar_total_divs, is_triplet=is_triplet
            )
        else:
            rest_beat = gp.Beat(m.voices[0], status=gp.BeatStatus.rest)
            rest_beat.duration.value = gp.Duration.whole
            m.voices[0].beats = [rest_beat]

        # Voice 2 (Bass)
        if bass and len(m.voices) > 1:
            groups2 = _group_by_time(bass, threshold=0.1)
            m.voices[1].beats = _build_voice_beats(
                groups2, m.voices[1], bar_total_divs, is_triplet=is_triplet, force_legato=True
            )

    # --- Voice integrity check ---
    # AlphaTabはVoiceのbeatsが空だとサイレントハングするため、
    # 全小節の全Voiceに最低1つのbeat(全休符)を保証する
    for m in track.measures:
        for v in m.voices:
            if not v.beats:
                rest_beat = gp.Beat(v, status=gp.BeatStatus.rest)
                rest_beat.duration.value = gp.Duration.whole
                v.beats = [rest_beat]

    # --- Write to bytes ---
    import io
    buf = io.BytesIO()
    gp.write(song, buf)
    return buf.getvalue()


# ─── Helper Functions ───

def _build_voice_beats(groups, voice, bar_total_divs, is_triplet=False, force_legato=False):
    """グループ化されたノートからGP Beatリストを構築する。"""
    gp_beats = []
    current_pos = 0

    # 2声部書きの場合のスナップグリッド
    # straight: 0,3,6,9,12... (16分音符グリッド)
    # triplet:  0,4,8,12,16... (3連符8分音符グリッド)
    if is_triplet:
        snap_grid = list(range(0, bar_total_divs + 1, DIVISIONS // 3))  # 0,4,8,12,...
    else:
        snap_grid = list(range(0, bar_total_divs + 1, 3))  # 0,3,6,9,12,...

    for group_idx, group in enumerate(groups):
        raw_pos = int(float(group[0]["beat_pos"]))
        # グリッドにスナップ
        target_pos = min(snap_grid, key=lambda x: abs(x - raw_pos))

        # Rest gap before this group
        gap = target_pos - current_pos
        if gap > 0:
            rest_beats = _divs_to_gp_beats_rest(gap, voice, is_triplet)
            gp_beats.extend(rest_beats)
            current_pos = target_pos
        elif gap < 0:
            # スナップで前のノートと重なった場合はスキップしない
            target_pos = current_pos

        # Note duration
        min_dur = DIVISIONS // 3 if is_triplet else 3  # triplet eighth or sixteenth
        if group_idx + 1 < len(groups):
            next_raw = int(float(groups[group_idx + 1][0]["beat_pos"]))
            next_target = min(snap_grid, key=lambda x: abs(x - next_raw))
            next_target = max(next_target, target_pos + min_dur)
        else:
            next_target = bar_total_divs
        gap_to_next = max(1, min(next_target - target_pos,
                                 bar_total_divs - target_pos))

        # Duration: CRNNが検出した実際の音の長さ(duration_divs)を使用
        if force_legato:
            # ベース音などは次のノートまで音価を伸ばす
            dur_divs = gap_to_next
        else:
            dur_divs = int(group[0].get("duration_divs", gap_to_next))
            dur_divs = min(dur_divs, gap_to_next, bar_total_divs - target_pos)
            dur_divs = max(1, dur_divs)
            
        if not is_triplet:
            normal_durs = [48, 36, 24, 18, 12, 9, 6, 3, 2, 1]
            dur_divs = min(normal_durs, key=lambda x: abs(x - dur_divs))
        else:
            if force_legato:
                # 3連符グリッドでのレガート用音価
                triplet_durs = [48, 36, 24, 18, 12, 8, 4]
                dur_divs = min(triplet_durs, key=lambda x: abs(x - dur_divs))

        # Create beat with all notes in this chord group
        beat = gp.Beat(voice, status=gp.BeatStatus.normal)
        gp_dur, gp_dotted, gp_tuplet = _divs_to_gp_duration(dur_divs, is_triplet)
        beat.duration.value = gp_dur
        beat.duration.isDotted = gp_dotted
        if gp_tuplet and is_triplet:
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
                note.effect.hammer = True
            elif tech in ("/", "\\"):
                note.effect.slides.append(gp.SlideType.shiftSlideTo)
            elif tech == "harmonic":
                note.effect.harmonic = gp.NaturalHarmonic()

            # 左手指番号 (finger_assigner.py で割り当て済み)
            # GP5: 0=thumb, 1=index, 2=middle, 3=annular, 4=little
            # SoloTab: 0=開放弦(指なし), 1-4=人差〜小指
            finger = entry.get("finger")
            if finger is not None and finger >= 1:
                note.effect.leftHandFinger = gp.Fingering(finger)

            beat.notes.append(note)

        gp_beats.append(beat)
        current_pos = target_pos + dur_divs

    # Trailing rest / extension
    remaining = bar_total_divs - current_pos
    if remaining > 0:
        if is_triplet and gp_beats:
            # 3連符アルペジオ: 末尾の隙間はRESTではなく最後のノートを延長
            # ギターのlet ring（音が自然に伸びる）に合致
            last_beat = gp_beats[-1]
            old_dur = last_beat.duration.value
            new_divs = dur_divs + remaining  # 直前のdur_divsに余りを加算
            new_gp_dur, new_dotted, new_tuplet = _divs_to_gp_duration(new_divs, is_triplet)
            last_beat.duration.value = new_gp_dur
            last_beat.duration.isDotted = new_dotted
            if new_tuplet and is_triplet:
                last_beat.duration.tuplet = gp.Tuplet(enters=3, times=2)
        else:
            rest_beats = _divs_to_gp_beats_rest(remaining, voice, is_triplet)
            gp_beats.extend(rest_beats)

    return gp_beats if gp_beats else []


def _parse_time_sig(ts: str) -> tuple[int, int]:
    if ts == "3/4":
        return 3, 4
    elif ts == "6/8":
        return 6, 8
    return 4, 4


def _filter_noise(notes, gate):
    if gate <= 0:
        return notes.copy()
    if not notes:
        return []
    # ノート数ベース: gate=0.5 → velocity下位50%のノートをカット
    # 重要: 同時発音ノート（和音・アルペジオ開始）は分離不可のため保護する
    import random
    cut_count = int(len(notes) * gate)
    if cut_count >= len(notes):
        cut_count = len(notes) - 1  # 最低1ノート残す
    if cut_count <= 0:
        return notes.copy()

    # 同時発音ノート（50ms以内）をグループ化し、保護対象を特定
    SIMUL_THRESHOLD = 0.05  # 50ms
    sorted_by_time = sorted(enumerate(notes), key=lambda x: float(x[1].get("start", 0)))
    protected_indices = set()
    i = 0
    while i < len(sorted_by_time):
        group = [sorted_by_time[i]]
        j = i + 1
        while j < len(sorted_by_time):
            t_diff = abs(float(sorted_by_time[j][1].get("start", 0)) - float(group[0][1].get("start", 0)))
            if t_diff <= SIMUL_THRESHOLD:
                group.append(sorted_by_time[j])
                j += 1
            else:
                break
        # 2ノート以上同時発音 → 全て保護
        if len(group) >= 2:
            for idx, _ in group:
                protected_indices.add(idx)
        i = j

    # velocityでグループ化し、同一velocity内はシャッフルして偏りを防止
    # (deterministic seed for reproducibility)
    rng = random.Random(42)
    indexed = list(enumerate(notes))
    # velocity + ランダムキーでソート（同一velocity内を均等分散）
    indexed.sort(key=lambda x: (float(x[1].get("velocity", 0.5)), rng.random()))
    # 保護対象を除いてカット
    cut_indices = set()
    for idx, _ in indexed:
        if len(cut_indices) >= cut_count:
            break
        if idx not in protected_indices:
            cut_indices.add(idx)
    filtered = [n for i, n in enumerate(notes) if i not in cut_indices]
    return filtered if filtered else [notes[0]]


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
