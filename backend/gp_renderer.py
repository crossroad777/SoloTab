from __future__ import annotations
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
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
from typing import List, Optional
import guitarpro as gp


# divisions per quarter note (triplet grid: 12 = LCM(4,3))
DIVISIONS = 12


def _validate_beat_playability(note_entries: List[dict],
                                tuning: List[int]) -> List[dict]:
    """
    量子化後の物理制約チェック。
    同一ビート(bar + beat_pos)に配置されたノートが
    物理的に弾けるフレットスパンかを検証し、違反があれば
    _assign_chord_notes で弾ける組み合わせに再割り当てする。
    """
    from collections import defaultdict
    from string_assigner import _get_max_span, _assign_chord_notes

    # bar + beat_pos でグループ化
    beat_groups = defaultdict(list)
    for i, entry in enumerate(note_entries):
        key = (int(entry.get("bar", 0)), int(entry.get("beat_pos", 0)))
        beat_groups[key].append(i)

    fixes = 0
    for key, indices in beat_groups.items():
        if len(indices) < 2:
            continue

        entries = [note_entries[i] for i in indices]
        frets_all = [(int(e.get("string", 1)), int(e.get("fret", 0))) for e in entries]
        fretted = [f for _, f in frets_all if f > 0]

        if len(fretted) < 2:
            continue

        span = max(fretted) - min(fretted)
        max_span = _get_max_span(min(fretted)) + 1  # +1: 正解PDFは少し広めのスパンを許容

        if span <= max_span:
            continue

        # 違反検出 → _assign_chord_notes で弾ける組み合わせを探す
        chord_notes = []
        for e in entries:
            chord_notes.append({
                "pitch": int(e.get("pitch", 60)),
                "start": float(e.get("start", 0)),
                "end": float(e.get("end", 0)),
                "velocity": float(e.get("velocity", 0.8)),
            })

        reassigned = _assign_chord_notes(chord_notes, tuning, 9, None)  # f10+への再配置を抑制

        # 再割り当て結果を検証
        new_fretted = [n.get("fret", 0) for n in reassigned if n.get("fret", 0) > 0]
        still_bad = False
        if len(new_fretted) >= 2:
            new_span = max(new_fretted) - min(new_fretted)
            new_max = _get_max_span(min(new_fretted))
            if new_span > new_max:
                still_bad = True

        if still_bad:
            # まだ弾けない → ローフレット基準でハイフレットを下げる
            # 人間の発想: f3が正しいなら、f10をf3近くに再配置
            used_strings = set()
            # ローフレット順にソート（低い方を基準に残す）
            sorted_by_fret = sorted(reassigned, key=lambda n: n.get("fret", 0))
            anchor_fret = sorted_by_fret[0].get("fret", 0)
            max_span = _get_max_span(anchor_fret)

            # まずanchorに近いノートを確定
            for n in sorted_by_fret:
                f = n.get("fret", 0)
                if f == 0 or abs(f - anchor_fret) <= max_span:
                    used_strings.add(n.get("string"))

            # anchorから遠いノートを再配置
            for n in sorted_by_fret:
                f = n.get("fret", 0)
                if f == 0 or abs(f - anchor_fret) <= max_span:
                    continue
                # このノートはスパン違反 → 弾ける位置に移動
                pitch = n.get("pitch", 60)
                best_s, best_f, best_dist = None, None, 999
                for si, op in enumerate(tuning):
                    sn = 6 - si
                    nf = pitch - op
                    if 0 <= nf <= 14 and sn not in used_strings:
                        dist = abs(nf - anchor_fret)
                        if dist <= max_span and dist < best_dist:
                            best_s, best_f, best_dist = sn, nf, dist
                if best_s is not None:
                    n["string"] = best_s
                    n["fret"] = best_f
                    used_strings.add(best_s)
                else:
                    # 削除せず、元の弦・フレットをそのまま保護
                    pass

        # 結果を元のnote_entriesに反映
        for j, idx in enumerate(indices):
            if j < len(reassigned):
                old_s = note_entries[idx].get("string")
                old_f = note_entries[idx].get("fret")
                new_s = reassigned[j].get("string", old_s)
                new_f = reassigned[j].get("fret", old_f)
                if old_s != new_s or old_f != new_f:
                    note_entries[idx]["string"] = new_s
                    note_entries[idx]["fret"] = new_f
                    fixes += 1

    # 全ノートを100%保持（音符削除は完全廃止）

    if fixes > 0:
        print(f"[gp_renderer] 物理制約チェック: {fixes}ノートを修正/除外")

    return note_entries


def notes_to_gp5(notes: List[dict], *,
                 beats: List[float],
                 backing_notes: List[dict] = None,
                 bpm: float = 120.0,
                 title: str = "Guitar TAB",
                 tuning: list | None = None,
                 time_signature: str = "4/4",
                 noise_gate: float = 0.0,
                 rhythm_info: dict | None = None,
                 key_signature: str = "C",
                 include_techniques: bool = True,
                 **kwargs) -> bytes | tuple[bytes, List[dict]]:
    """
    ノートデータからGP5バイナリを生成する。

    Parameters
    ----------
    notes : list[dict]
        Keys: start, end, pitch, string, fret, velocity, technique
    beats : list[float]
        ビート時刻(秒)
    backing_notes : list[dict], optional
        バッキング用のノートリスト。渡された場合は2トラックで出力する。
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

    # Normalize start/end keys
    for n in notes:
        if "start" not in n and "start_time" in n:
            n["start"] = n["start_time"]
        if "end" not in n and "end_time" in n:
            n["end"] = n["end_time"]
        if "start_time" not in n and "start" in n:
            n["start_time"] = n["start"]
        if "end_time" not in n and "end" in n:
            n["end_time"] = n["end"]

    # Noise gate filter
    filtered_melody = _filter_noise(notes, noise_gate)
    is_2tracks = backing_notes is not None
    if is_2tracks:
        filtered_backing = _filter_noise(backing_notes, noise_gate)
    else:
        filtered_backing = []

    # Quantization helper using Universal Quantizer
    def _quantize_track(filtered_notes):
        try:
            from universal_quantizer import quantize_notes_universal
            entries = quantize_notes_universal(
                filtered_notes, beats, bpm,
                time_signature=time_signature,
                beats_per_bar=beats_per_bar,
            )
        except Exception as e:
            print(f"[gp_renderer] Universal Quantizer failed, falling back: {e}")
            from tab_renderer import _assign_to_bars
            entries = _assign_to_bars(filtered_notes, beats, beats_per_bar, bpm=bpm, time_signature=time_signature, rhythm_info=rhythm_info)

        # Validate playability
        entries = _validate_beat_playability(entries, tuning)
        
        # === [TASK-900-E: ピッチ整合性不変条件の強制] ===
        gp_pitch_violations = 0
        for entry in entries:
            s = int(entry.get("string", 1))
            f = int(entry.get("fret", 0))
            target_p = int(entry.get("pitch", 60))
            computed_p = tuning[6 - s] + f
            if computed_p != target_p:
                gp_pitch_violations += 1
                from string_assigner import get_possible_positions
                valid_positions = get_possible_positions(target_p, tuning, 14)
                if valid_positions:
                    entry["string"] = valid_positions[0][0]
                    entry["fret"] = valid_positions[0][1]
                else:
                    entry["string"] = 1
                    entry["fret"] = min(max(0, target_p - tuning[5]), 14)
        if gp_pitch_violations > 0:
            print(f"[gp_renderer] [INVARIANT] ピッチ不変条件違反: {gp_pitch_violations}ノートを修復")

        return entries

    melody_entries = _quantize_track(filtered_melody)
    if is_2tracks:
        backing_entries = _quantize_track(filtered_backing)
    else:
        backing_entries = []

    all_note_entries = melody_entries + backing_entries

    # Calculate total bars (ensure entire audio beats span is covered)
    notes_bars = (max(int(e["bar"]) for e in all_note_entries) + 1) if all_note_entries else 1
    beats_bars = ((len(beats) + beats_per_bar - 1) // beats_per_bar) if beats else 1
    total_bars = max(notes_bars, beats_bars, 1)

    # --- Build GP5 Song ---
    song = gp.Song()
    song.title = title
    song.artist = "SoloTab"
    song.tempo = int(bpm)

    # Track 1 setup (Melody)
    track1 = song.tracks[0]
    track1.name = "Guitar (Melody)" if is_2tracks else "Guitar"
    track1.channel.instrument = 25  # Acoustic Guitar (steel)
    track1.strings = [
        gp.GuitarString(number=i + 1, value=tuning[5 - i])
        for i in range(6)
    ]  # GP format: string 1 = highest (E4), string 6 = lowest (E2)

    tracks_to_process = [(track1, melody_entries)]

    # Track 2 setup (Backing)
    if is_2tracks:
        track2 = gp.Track(song)
        track2.name = "Guitar (Backing)"
        track2.channel.instrument = 25  # Acoustic Guitar (steel)
        track2.strings = [
            gp.GuitarString(number=i + 1, value=tuning[5 - i])
            for i in range(6)
        ]
        song.tracks.append(track2)
        tracks_to_process.append((track2, backing_entries))

    # Key signature
    key_fifths = 0  # C major = 調号なし

    # --- Measure Headers ---
    # First measure header already exists, configure it
    mh0 = song.measureHeaders[0]
    mh0.timeSignature.numerator = beats_per_bar
    mh0.timeSignature.denominator.value = _beat_type_to_gp_dur(beat_type)
    mh0.keySignature = _fifths_to_gp_key(key_fifths)

    # Add remaining measure headers
    for bar_num in range(1, total_bars):
        mh = gp.MeasureHeader()
        mh.number = bar_num + 1
        mh.start = mh0.start + bar_num * _bar_length(beats_per_bar, beat_type)
        mh.timeSignature.numerator = beats_per_bar
        mh.timeSignature.denominator.value = _beat_type_to_gp_dur(beat_type)
        mh.keySignature = _fifths_to_gp_key(key_fifths)
        song.measureHeaders.append(mh)

    # _group_by_time helper from tab_renderer
    from tab_renderer import _group_by_time

    for track, entries_to_use in tracks_to_process:
        # Build Measures
        measures = [track.measures[0]]
        for bar_num in range(1, total_bars):
            m = gp.Measure(track, song.measureHeaders[bar_num])
            measures.append(m)
        track.measures = measures

        # Fill each measure with notes
        divs_per_beat = DIVISIONS if beat_type == 4 else DIVISIONS // 2
        bar_total_divs = beats_per_bar * divs_per_beat

        split_pitch = 52

        def _is_bass(n):
            s = int(n.get("string", 0))
            if s >= 4:  # 弦4,5,6 = ベース
                return True
            if s >= 1:  # 弦1,2,3 = メロディ
                return False
            return int(n.get("pitch", 60)) <= split_pitch

        # --- 論文§6準拠: 低音弦ノートの全音符完全保護 (間引き・ワープの完全撤廃) ---
        bars_data = []
        for bar_num in range(total_bars):
            bar_notes = [e for e in entries_to_use if e["bar"] == bar_num]
            # 全ノートをそのまま時系列順で保持
            bars_data.append({"melody": bar_notes, "bass": []})


        for bar_num in range(total_bars):
            m = track.measures[bar_num]
            bd = bars_data[bar_num]
            melody = bd["melody"]
            bass = bd["bass"]

            if not melody and not bass:
                m.voices[0].beats = _divs_to_gp_beats_rest(bar_total_divs, m.voices[0], is_triplet)
                continue

            # Voice 1 (Melody)
            if melody:
                groups1 = _group_by_time(melody, threshold=0.1)
                m.voices[0].beats = _build_voice_beats(
                    groups1, m.voices[0], bar_total_divs, is_triplet=is_triplet,
                    include_techniques=include_techniques
                )
            else:
                m.voices[0].beats = _divs_to_gp_beats_rest(bar_total_divs, m.voices[0], is_triplet)

            # Voice 2 (Bass)
            if bass and len(m.voices) > 1:
                groups2 = _group_by_time(bass, threshold=0.1)
                m.voices[1].beats = _build_voice_beats(
                    groups2, m.voices[1], bar_total_divs, is_triplet=is_triplet, force_legato=True,
                    include_techniques=include_techniques
                )

        # Voice integrity check
        for m in track.measures:
            for v in m.voices:
                if not v.beats:
                    v.beats = _divs_to_gp_beats_rest(bar_total_divs, v, is_triplet)

    # --- Song Tempo Header (曲頭に1回のみ標準設定。毎小節の重複出力を防止) ---
    song.tempo = int(round(bpm))

    # --- Write to bytes ---
    import io
    buf = io.BytesIO()
    gp.write(song, buf)

    if kwargs.get("return_entries", False):
        for e in melody_entries:
            e["track"] = 0
        for e in backing_entries:
            e["track"] = 1
        all_quantized_entries = melody_entries + backing_entries
        all_quantized_entries.sort(key=lambda x: (int(x.get("bar", 0)), float(x.get("beat_pos", 0)), int(x.get("pitch", 0))))
        return buf.getvalue(), all_quantized_entries
    return buf.getvalue()


# ─── Helper Functions ───

def _build_voice_beats(groups, voice, bar_total_divs, is_triplet=False, force_legato=False, include_techniques=True):
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
        # グリッドにスナップ（密集時はより細かなスロットへ適応）
        target_pos = min(snap_grid, key=lambda x: abs(x - raw_pos))
        target_pos = max(current_pos, min(target_pos, bar_total_divs - 1))

        # 小節末でもブレークせず、最後のスロットに必ず収容
        if current_pos >= bar_total_divs:
            target_pos = bar_total_divs - 1
            current_pos = bar_total_divs - 1

        # Rest gap before this group
        gap = target_pos - current_pos
        if gap > 0:
            rest_beats = _divs_to_gp_beats_rest(gap, voice, is_triplet)
            gp_beats.extend(rest_beats)
            current_pos = target_pos

        # Note duration (密集時は最小1 divまで適応して小節内に100%収容)
        remaining_slots = len(groups) - group_idx
        available_space = bar_total_divs - target_pos
        min_dur = max(1, min(3 if not is_triplet else 4, available_space // remaining_slots)) if remaining_slots > 0 else 1

        if group_idx + 1 < len(groups):
            next_raw = int(float(groups[group_idx + 1][0].get("beat_pos_in_bar", groups[group_idx + 1][0].get("beat_pos", 0))))
            next_target = min(snap_grid, key=lambda x: abs(x - next_raw))
            next_target = max(target_pos + min_dur, min(next_target, bar_total_divs))
        else:
            next_target = bar_total_divs

        gap_to_next = max(1, min(next_target - target_pos, bar_total_divs - target_pos))

        # Duration
        if force_legato:
            dur_divs = max(1, min(gap_to_next, bar_total_divs - target_pos))
        else:
            quantized_dur = int(group[0].get("duration_divs", gap_to_next))
            if quantized_dur < gap_to_next * 0.7:
                dur_divs = gap_to_next
            else:
                dur_divs = quantized_dur
            dur_divs = max(1, min(dur_divs, gap_to_next, bar_total_divs - target_pos))
            
        if not is_triplet:
            normal_durs = [48, 36, 24, 18, 12, 9, 6, 3, 2, 1]
            dur_divs = min(normal_durs, key=lambda x: abs(x - dur_divs))
        else:
            triplet_durs = [48, 36, 24, 18, 12, 8, 4, 3, 2, 1]
            dur_divs = min(triplet_durs, key=lambda x: abs(x - dur_divs))

        # Post-snap cap: 小節からはみ出さないよう切り下げ
        remaining_in_bar = bar_total_divs - target_pos
        if dur_divs > remaining_in_bar:
            valid = normal_durs if not is_triplet else triplet_durs
            candidates = [d for d in valid if d <= remaining_in_bar]
            dur_divs = max(candidates) if candidates else 1


        # Create beat with all notes in this chord group
        beat = gp.Beat(voice, status=gp.BeatStatus.normal)
        group_is_triplet = is_triplet or group[0].get("is_triplet", False)
        gp_dur, gp_dotted, gp_tuplet = _divs_to_gp_duration(dur_divs, group_is_triplet)
        beat.duration.value = gp_dur
        beat.duration.isDotted = gp_dotted
        if (gp_tuplet and group_is_triplet) or group_is_triplet and dur_divs in [2, 4, 8]:
            beat.duration.tuplet = gp.Tuplet(enters=3, times=2)

        for entry in group:
            string_num = int(entry.get("string", 1))
            fret       = int(entry.get("fret", 0))
            note       = gp.Note(beat)
            note.value  = fret
            note.string = string_num
            note.velocity = _vel_to_gp(entry.get("velocity", 0.5))

            # ═══════════════════════════════════════════════════════════
            # YG全37パターン完全準拠 GP5テクニックエンコード
            # include_techniques=False の場合はスキップ
            # ═══════════════════════════════════════════════════════════
            tech = entry.get("technique") if include_techniques else None

            # ── 1. レガート系 ──────────────────────────────────────────
            if tech in ("h", "hammer-on", "hammer_on"):
                # ハンマリング・オン (H): slur arc 上向き
                note.effect.hammer = True

            elif tech in ("p", "pull-off", "pull_off"):
                # プリング・オフ (P): slur arc 下向き
                # GP5フォーマット: hammer=True で AlphaTab が方向を自動判定
                note.effect.hammer = True

            elif tech in ("tr", "trill"):
                # トリル (tr): 指定フレットと交互に繰り返す
                trill_fret = min(note.value + 2, 24)
                note.effect.trill = gp.TrillEffect(
                    fret=trill_fret,
                    duration=gp.Duration(value=16),
                )

            # ── 2. スライド系 ────────────────────────────────────────
            elif tech in ("/", "slide_up", "slide"):
                # スライドアップ (S): 目標音へシフト
                note.effect.slides.append(gp.SlideType.shiftSlideTo)

            elif tech in ("\\", "slide_down"):
                # スライドダウン (S): 目標音へシフト
                note.effect.slides.append(gp.SlideType.shiftSlideTo)

            elif tech in ("legato_slide",):
                # レガートスライド: タイで繋がれたスライド
                note.effect.slides.append(gp.SlideType.legatoSlideTo)

            elif tech in ("gliss_up", "gliss"):
                # グリス上 (g): 不定位置へのスライド
                note.effect.slides.append(gp.SlideType.intoFromBelow)

            elif tech in ("gliss_down",):
                # グリス下 (g)
                note.effect.slides.append(gp.SlideType.intoFromAbove)

            elif tech in ("gliss_out_up",):
                # グリスアウト上向き
                note.effect.slides.append(gp.SlideType.outUpwards)

            elif tech in ("gliss_out_down",):
                # グリスアウト下向き
                note.effect.slides.append(gp.SlideType.outDownwards)

            # ── 3. チョーキング系 ─────────────────────────────────────
            elif tech in ("b", "bend"):
                # 1音チョーキング (C): GP5スケール 4=全音, position 0-60
                note.effect.bend = gp.BendEffect(
                    type=gp.BendType.bend, value=4,
                    points=[gp.BendPoint(0,0), gp.BendPoint(30,4), gp.BendPoint(60,4)]
                )
            elif tech in ("b_half", "bend_half"):
                # 半音チョーキング (H.C): 2=半音
                note.effect.bend = gp.BendEffect(
                    type=gp.BendType.bend, value=2,
                    points=[gp.BendPoint(0,0), gp.BendPoint(30,2), gp.BendPoint(60,2)]
                )
            elif tech in ("b_1half", "bend_1half"):
                # 1音半チョーキング (1H.C): 6=1.5音
                note.effect.bend = gp.BendEffect(
                    type=gp.BendType.bend, value=6,
                    points=[gp.BendPoint(0,0), gp.BendPoint(30,6), gp.BendPoint(60,6)]
                )
            elif tech in ("b_2", "bend_2"):
                # 2音チョーキング (2C): 8=2音
                note.effect.bend = gp.BendEffect(
                    type=gp.BendType.bend, value=8,
                    points=[gp.BendPoint(0,0), gp.BendPoint(30,8), gp.BendPoint(60,8)]
                )
            elif tech in ("b_quarter", "bend_quarter"):
                # クォーターチョーキング (Q.C): 1=クォーター音
                note.effect.bend = gp.BendEffect(
                    type=gp.BendType.bend, value=1,
                    points=[gp.BendPoint(0,0), gp.BendPoint(30,1), gp.BendPoint(60,1)]
                )
            elif tech in ("bend_release",):
                # ベンド＆リリース: 全音上げてから戻す
                note.effect.bend = gp.BendEffect(
                    type=gp.BendType.bendRelease, value=4,
                    points=[gp.BendPoint(0,0), gp.BendPoint(20,4),
                            gp.BendPoint(40,4), gp.BendPoint(60,0)]
                )
            elif tech in ("pre_bend", "prebend", "U"):
                # チョークアップ/プリベンド (U): 上げた状態で発音
                note.effect.bend = gp.BendEffect(
                    type=gp.BendType.prebend, value=4,
                    points=[gp.BendPoint(0,4), gp.BendPoint(60,4)]
                )
            elif tech in ("release_bend", "D"):
                # チョークダウン/リリースベンド (D): プリベンドから解放
                note.effect.bend = gp.BendEffect(
                    type=gp.BendType.prebendRelease, value=4,
                    points=[gp.BendPoint(0,4), gp.BendPoint(30,4), gp.BendPoint(60,0)]
                )

            # ── 4. ビブラート系 ───────────────────────────────────────
            elif tech in ("~", "vibrato", "wide_vibrato"):
                # アコギ・ソロギター出版譜に合わせ、視覚ノイズとなる浮遊波線記号は出力しない
                pass

            # ── 5. ハーモニクス系 ─────────────────────────────────────
            elif tech in ("harmonic", "n.h", "nh", "natural_harmonic", "ah", "artificial_harmonic"):
                # ナチュラル/人工ハーモニクス (N.H): ◇ ヘッド表示
                note.effect.harmonic = gp.NaturalHarmonic()

            elif tech in ("p_harmonic", "p.h", "ph", "pinch_harmonic"):
                # ピッキングハーモニクス (P.H): 人工倍音
                note.effect.harmonic = gp.PinchHarmonic()

            elif tech in ("semi_harmonic", "tapped_harmonic", "th"):
                note.effect.harmonic = gp.TappedHarmonic()

            # ── 6. ミュート系 ─────────────────────────────────────────
            elif tech in ("pm", "palm_mute", "M"):
                # パームミュート (M): P.M.---ライン表示
                note.effect.palmMute = True

            elif tech in ("x", "dead_note", "mute", "brushing", "bh", "na"):
                # TAB譜上に確実にアタックミュート「x」を表示 ＆ 音符の上にもアタック記号を付与
                note.type = gp.NoteType.dead
                note.effect.accentuatedNote = True

            # ── 7. その他 ─────────────────────────────────────────────
            elif tech in ("let_ring", "let ring"):
                # レットリング: 点線ライン表示
                note.effect.letRing = True

            elif tech in ("staccato",):
                note.effect.staccato = True

            elif tech in ("accent",):
                note.effect.accentuatedNote = True

            elif tech in ("heavy_accent",):
                note.effect.heavyAccentuatedNote = True

            # ── Beat-level テクニック (note-levelループ外で設定必要) ──
            # tap/tapping は beat.effect.slapEffect で設定 → 後処理
            elif tech in ("tap", "tapping", "T"):
                # タッピング: beat.effect.slapEffect = SlapEffect.tapping
                # note-levelでは設定不可。beat_effectを直接設定する。
                beat.effect.slapEffect = gp.SlapEffect.tapping
                note.effect.hammer = True  # 叩くのでhammer

            beat.notes.append(note)


        gp_beats.append(beat)
        current_pos = target_pos + dur_divs
        current_pos = min(current_pos, bar_total_divs)



    # Trailing rest / extension
    remaining = bar_total_divs - current_pos
    if remaining > 0:
        if is_triplet and gp_beats:
            # 3連符アルペジオ: 末尾の隙間はRESTではなく最後のノートを延長
            # ギターのlet ring（音が自然に伸びる）に合致
            last_beat = gp_beats[-1]
            old_dur = last_beat.duration.value
            # Compute actual last beat duration from its GP duration instead of
            # using the loop's dur_divs which may be stale if the last group was
            # skipped by the bar-end guard.
            last_beat_divs = _gp_duration_to_divs(old_dur, last_beat.duration.isDotted,
                                                   hasattr(last_beat.duration, 'tuplet') and last_beat.duration.tuplet is not None and last_beat.duration.tuplet.enters == 3)
            new_divs = last_beat_divs + remaining
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

    # 同時発音ノート（20ms以内）をグループ化し、保護対象を特定
    SIMUL_THRESHOLD = 0.02  # 20ms
    def get_start(n):
        return float(n.get("start") if n.get("start") is not None else n.get("start_time", 0.0))

    sorted_by_time = sorted(enumerate(notes), key=lambda x: get_start(x[1]))
    protected_indices = set()
    i = 0
    while i < len(sorted_by_time):
        group = [sorted_by_time[i]]
        j = i + 1
        while j < len(sorted_by_time):
            t_diff = abs(get_start(sorted_by_time[j][1]) - get_start(group[0][1]))
            if t_diff <= SIMUL_THRESHOLD:
                group.append(sorted_by_time[j])
                j += 1
            else:
                break
        # 2ノート以上同時発音（コード/和音）→ 音楽の骨格のため無条件保護
        if len(group) >= 2:
            for idx, _ in group:
                protected_indices.add(idx)
        i = j

    # 単音（ノイズ候補）のインデックス
    single_indices = [idx for idx in range(len(notes)) if idx not in protected_indices]
    if not single_indices:
        return notes.copy()

    single_vels = [float(notes[idx].get("velocity", 0.5)) for idx in single_indices]
    
    # 動的パーセンタイルカット:
    # ユーザーがスライダーで指定した gate (0.0〜0.80) に応じて、単音の velocity 下位 rank を足切り
    cut_rank = int(len(single_vels) * min(float(gate), 0.85))
    sorted_vels = sorted(single_vels)
    cutoff_vel = sorted_vels[cut_rank] if cut_rank < len(sorted_vels) else max(single_vels)

    cut_indices = set()
    for idx in single_indices:
        if float(notes[idx].get("velocity", 0.5)) < cutoff_vel:
            cut_indices.add(idx)

    # 万が一すべてのノートがカットされてしまった場合のセーフティ
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


def _gp_duration_to_divs(gp_dur: int, is_dotted: bool, is_triplet_beat: bool) -> int:
    """Reverse lookup: GP Duration value + flags -> divs count.
    Used to recover the actual duration of a beat for trailing rest extension."""
    base_map = {
        gp.Duration.whole: 48,
        gp.Duration.half: 24,
        gp.Duration.quarter: 12,
        gp.Duration.eighth: 6,
        gp.Duration.sixteenth: 3,
        gp.Duration.thirtySecond: 2,
        gp.Duration.sixtyFourth: 1,
    }
    divs = base_map.get(gp_dur, 12)
    if is_dotted:
        divs = divs * 3 // 2  # e.g. 12 -> 18
    if is_triplet_beat:
        divs = divs * 2 // 3  # e.g. 12 -> 8, 6 -> 4
    return max(1, divs)


def _divs_to_gp_beats_rest(divs: int, voice, is_triplet: bool) -> list:
    """Rest duration expressed as one or more GP rest beats."""
    if divs <= 0:
        return []
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
    """すべての音符が五線譜・TAB譜上でクッキリ濃く読めるよう、均一な濃色Velocity (95=Forte) を返す。"""
    return 95
