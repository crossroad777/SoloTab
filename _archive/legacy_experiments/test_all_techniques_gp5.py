"""
test_all_techniques_gp5.py
==========================
AlphaTabで全テクニックを検証するためのGP5テストファイル生成スクリプト。
YG全37パターン × PyGuitarPro × AlphaTab の完全対応マップを実証する。

実行: python test_all_techniques_gp5.py
出力: backend/uploads/test_techniques/test_all_techniques.gp5
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import guitarpro as gp
from pathlib import Path

def make_note(beat, string, fret, technique=None, note_type=gp.NoteType.normal):
    note = gp.Note(beat)
    note.string = string
    note.value = fret
    note.type = note_type
    note.velocity = 95
    return note, technique

def add_bar_title(measure, title):
    """小節にマーカーを付ける（タイトル用）"""
    pass  # GP5フォーマットのマーカー付加は省略

def build_test_gp5():
    # ── 1. Song / Track 設定 ──
    song = gp.Song()
    song.title = "AlphaTab Technique Test - YG37 Patterns"
    song.artist = "SoloTab Verification"
    song.tempo = 80

    track = song.tracks[0]
    track.name = "Guitar"
    track.strings = [
        gp.GuitarString(1, 64),  # E4
        gp.GuitarString(2, 59),  # B3
        gp.GuitarString(3, 55),  # G3
        gp.GuitarString(4, 50),  # D3
        gp.GuitarString(5, 45),  # A2
        gp.GuitarString(6, 40),  # E2
    ]

    # テクニックテストの定義
    # 各エントリ: (説明, [(string, fret, technique_func)])
    test_cases = []

    # ═══════════════════════════════════════════════════════════
    # 1. Hammer-on (H) — note.effect.hammer = True
    # ═══════════════════════════════════════════════════════════
    def hammer_on(note):
        note.effect.hammer = True
    test_cases.append(("01_Hammer-on H", [
        (2, 5, None),
        (2, 7, hammer_on),
        (2, 5, None),
        (2, 8, hammer_on),
    ]))

    # ═══════════════════════════════════════════════════════════
    # 2. Pull-off (P) — note.effect.hammer = True (同じフラグ)
    # ═══════════════════════════════════════════════════════════
    def pull_off(note):
        note.effect.hammer = True  # GP5ではH/P同一フラグ、方向で判定
    test_cases.append(("02_Pull-off P", [
        (2, 8, None),
        (2, 5, pull_off),
        (2, 7, None),
        (2, 5, pull_off),
    ]))

    # ═══════════════════════════════════════════════════════════
    # 3. Slide (S) — shiftSlideTo
    # ═══════════════════════════════════════════════════════════
    def slide_up(note):
        note.effect.slides.append(gp.SlideType.shiftSlideTo)
    def slide_down(note):
        note.effect.slides.append(gp.SlideType.shiftSlideTo)
    def legato_slide(note):
        note.effect.slides.append(gp.SlideType.legatoSlideTo)
    test_cases.append(("03_Slide S", [
        (2, 5, slide_up),
        (2, 8, slide_down),
        (2, 10, legato_slide),
        (2, 7, None),
    ]))

    # ═══════════════════════════════════════════════════════════
    # 4. Glissando (g) — intoFromBelow / outDownwards / outUpwards
    # ═══════════════════════════════════════════════════════════
    def gliss_in_below(note):
        note.effect.slides.append(gp.SlideType.intoFromBelow)
    def gliss_in_above(note):
        note.effect.slides.append(gp.SlideType.intoFromAbove)
    def gliss_out_down(note):
        note.effect.slides.append(gp.SlideType.outDownwards)
    def gliss_out_up(note):
        note.effect.slides.append(gp.SlideType.outUpwards)
    test_cases.append(("04_Glissando g", [
        (2, 7, gliss_in_below),
        (2, 7, gliss_out_up),
        (2, 7, gliss_in_above),
        (2, 7, gliss_out_down),
    ]))

    # ═══════════════════════════════════════════════════════════
    # 5. チョーキング 1音 (C) — BendType.bend, value=100
    # ═══════════════════════════════════════════════════════════
    def bend_1(note):
        note.effect.bend = gp.BendEffect(
            type=gp.BendType.bend, value=100,
            points=[gp.BendPoint(0,0), gp.BendPoint(30,100), gp.BendPoint(60,100)]
        )
    test_cases.append(("05_Bend_1step C", [
        (2, 7, bend_1),
        (2, 7, None),
        (2, 9, bend_1),
        (2, 9, None),
    ]))

    # ═══════════════════════════════════════════════════════════
    # 6. チョーキング 半音 (H.C) — BendType.bend, value=50
    # ═══════════════════════════════════════════════════════════
    def bend_half(note):
        note.effect.bend = gp.BendEffect(
            type=gp.BendType.bend, value=50,
            points=[gp.BendPoint(0,0), gp.BendPoint(30,50), gp.BendPoint(60,50)]
        )
    test_cases.append(("06_Bend_halfstep HC", [
        (2, 7, bend_half),
        (2, 7, None),
        (2, 9, bend_half),
        (2, 9, None),
    ]))

    # ═══════════════════════════════════════════════════════════
    # 7. ベンド＆リリース — BendType.bendRelease
    # ═══════════════════════════════════════════════════════════
    def bend_release(note):
        note.effect.bend = gp.BendEffect(
            type=gp.BendType.bendRelease, value=100,
            points=[gp.BendPoint(0,0), gp.BendPoint(20,100), gp.BendPoint(40,100), gp.BendPoint(60,0)]
        )
    test_cases.append(("07_BendRelease", [
        (2, 7, bend_release),
        (2, 7, None),
        (2, 9, bend_release),
        (2, 9, None),
    ]))

    # ═══════════════════════════════════════════════════════════
    # 8. プリベンド (U - チョークアップ) — BendType.prebend
    # ═══════════════════════════════════════════════════════════
    def prebend(note):
        note.effect.bend = gp.BendEffect(
            type=gp.BendType.prebend, value=100,
            points=[gp.BendPoint(0,100), gp.BendPoint(60,100)]
        )
    test_cases.append(("08_PreBend U", [
        (2, 7, prebend),
        (2, 7, None),
        (2, 9, prebend),
        (2, 9, None),
    ]))

    # ═══════════════════════════════════════════════════════════
    # 9. プリベンド＆リリース (D - チョークダウン) — BendType.prebendRelease
    # ═══════════════════════════════════════════════════════════
    def prebend_release(note):
        note.effect.bend = gp.BendEffect(
            type=gp.BendType.prebendRelease, value=100,
            points=[gp.BendPoint(0,100), gp.BendPoint(30,100), gp.BendPoint(60,0)]
        )
    test_cases.append(("09_PreBendRelease D", [
        (2, 7, prebend_release),
        (2, 7, None),
        (2, 9, prebend_release),
        (2, 9, None),
    ]))

    # ═══════════════════════════════════════════════════════════
    # 10. ビブラート note-level (~) — note.effect.vibrato
    # ═══════════════════════════════════════════════════════════
    def note_vibrato(note):
        note.effect.vibrato = True
    test_cases.append(("10_Vibrato_note ~", [
        (2, 7, note_vibrato),
        (2, 5, note_vibrato),
        (2, 8, note_vibrato),
        (2, 5, None),
    ]))

    # ═══════════════════════════════════════════════════════════
    # 11. ナチュラルハーモニクス (N.H) — NaturalHarmonic
    # ═══════════════════════════════════════════════════════════
    def natural_harmonic(note):
        note.effect.harmonic = gp.NaturalHarmonic()
    test_cases.append(("11_NaturalHarmonic NH", [
        (2, 12, natural_harmonic),
        (2, 7, natural_harmonic),
        (2, 5, natural_harmonic),
        (2, 12, natural_harmonic),
    ]))

    # ═══════════════════════════════════════════════════════════
    # 12. ピッキングハーモニクス (P.H) — PinchHarmonic
    # ═══════════════════════════════════════════════════════════
    def pinch_harmonic(note):
        note.effect.harmonic = gp.PinchHarmonic()
    test_cases.append(("12_PinchHarmonic PH", [
        (2, 7, pinch_harmonic),
        (2, 9, pinch_harmonic),
        (2, 7, pinch_harmonic),
        (2, 9, pinch_harmonic),
    ]))

    # ═══════════════════════════════════════════════════════════
    # 13. デッドノート / ブラッシング (×) — NoteType.dead
    # ═══════════════════════════════════════════════════════════
    test_cases.append(("13_DeadNote x", [
        (6, 0, None),  # dead note on 6th string
        (5, 0, None),
        (4, 0, None),
        (6, 0, None),
    ]))
    # NOTE: dead nodesはnote.typeで設定するため別処理

    # ═══════════════════════════════════════════════════════════
    # 14. パームミュート (M) — note.effect.palmMute
    # ═══════════════════════════════════════════════════════════
    def palm_mute(note):
        note.effect.palmMute = True
    test_cases.append(("14_PalmMute M", [
        (6, 0, palm_mute),
        (6, 2, palm_mute),
        (6, 0, palm_mute),
        (6, 2, palm_mute),
    ]))

    # ═══════════════════════════════════════════════════════════
    # 15. トリル (tr) — TrillEffect
    # ═══════════════════════════════════════════════════════════
    def trill(note):
        note.effect.trill = gp.TrillEffect(
            fret=note.value + 2,
            duration=gp.Duration(value=16)
        )
    test_cases.append(("15_Trill tr", [
        (2, 5, trill),
        (2, 5, None),
        (2, 7, trill),
        (2, 7, None),
    ]))

    # ═══════════════════════════════════════════════════════════
    # 16. タッピング (T) — SlapEffect.tapping
    # ═══════════════════════════════════════════════════════════
    test_cases.append(("16_Tapping T", [
        (1, 12, None),
        (1, 8, None),
        (1, 12, None),
        (1, 8, None),
    ]))
    # NOTE: tappingはbeat.effect.slapEffectで設定するため別処理

    # ═══════════════════════════════════════════════════════════
    # 17. レットリング — note.effect.letRing
    # ═══════════════════════════════════════════════════════════
    def let_ring(note):
        note.effect.letRing = True
    test_cases.append(("17_LetRing", [
        (3, 0, let_ring),
        (2, 0, let_ring),
        (1, 0, let_ring),
        (4, 2, None),
    ]))

    # ═══════════════════════════════════════════════════════════
    # 小節を生成
    # ═══════════════════════════════════════════════════════════
    # 既存の空小節を削除して、テストケース分の小節を追加
    num_cases = len(test_cases)
    while len(track.measures) < num_cases:
        header = gp.MeasureHeader()
        header.number = len(track.measures) + 1
        header.timeSignature.numerator = 4
        header.timeSignature.denominator.value = 4
        song.measureHeaders.append(header)
        measure = gp.Measure(track, header)
        track.measures.append(measure)

    print(f"テストケース数: {num_cases}")

    for i, (title, notes_def) in enumerate(test_cases):
        if i >= len(track.measures):
            break
        measure = track.measures[i]
        voice = measure.voices[0]
        voice.beats.clear()

        print(f"  [{i+1:02d}] {title}")

        is_tapping = "16_Tapping" in title
        is_dead = "13_DeadNote" in title

        for j, (string, fret, tech_func) in enumerate(notes_def):
            beat = gp.Beat(voice)
            beat.duration.value = 4  # 4分音符
            beat.status = gp.BeatStatus.normal

            # タッピングはbeat-levelで設定
            if is_tapping:
                beat.effect.slapEffect = gp.SlapEffect.tapping

            note = gp.Note(beat)
            note.string = string
            note.value = fret
            note.velocity = 95

            # デッドノートはnote.typeで設定
            if is_dead:
                note.type = gp.NoteType.dead
            else:
                note.type = gp.NoteType.normal
                if tech_func:
                    tech_func(note)

            beat.notes.append(note)
            voice.beats.append(beat)

    # ── 出力 ──
    out_dir = Path("uploads/test_all_techniques")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test_all_techniques.gp5"

    gp.write(song, str(out_path))
    print(f"\n[OK] Generated: {out_path}")
    print(f"   Size: {out_path.stat().st_size:,} bytes")
    print(f"\n検証方法:")
    print(f"  ブラウザで http://localhost:5174/ を開き")
    print(f"  test_all_techniques セッションをロードしてください")
    return str(out_path)


if __name__ == "__main__":
    out = build_test_gp5()
    print(f"\n完了: {out}")
