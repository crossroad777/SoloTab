"""Full verification of Romance GP5 output."""
import sys
sys.path.insert(0, '.')
import guitarpro as gp

path = r'D:\Music\禁じられた遊び　(ロマンス ) ギター Tab譜 楽譜　コードネーム付 - アコースティック 名曲 ギター タブ 楽譜ギター タブ譜 (128k).gp5'
song = gp.parse(path)
t = song.tracks[0]

print(f"Title: {song.title}")
print(f"Tempo: {song.tempo} BPM")
print(f"Measures: {len(t.measures)}")
print(f"Time Sig: {song.measureHeaders[0].timeSignature.numerator}/{song.measureHeaders[0].timeSignature.denominator.value}")
print()

# 検証項目
issues = []
total_measures = len(t.measures)

for mi in range(total_measures):
    m = t.measures[mi]
    mh = song.measureHeaders[mi]
    ts_num = mh.timeSignature.numerator
    ts_den = mh.timeSignature.denominator.value

    for vi, v in enumerate(m.voices):
        if not v.beats:
            continue

        # 各ビートの実効tick数を計算
        total_ticks = 0
        note_count = 0
        rest_count = 0
        chord_count = 0  # 2音以上同時発音
        has_tuplet = False

        for b in v.beats:
            dur_val = b.duration.value  # 1=whole, 2=half, 4=quarter, 8=eighth...
            base_ticks = 960 * 4 // dur_val  # GP: quarter=960 ticks
            if b.duration.isDotted:
                base_ticks = int(base_ticks * 1.5)
            tp = b.duration.tuplet
            if tp and tp.enters == 3 and tp.times == 2:
                base_ticks = base_ticks * 2 // 3
                has_tuplet = True
            total_ticks += base_ticks

            if 'rest' in str(b.status).lower():
                rest_count += 1
            else:
                note_count += 1
                if len(b.notes) > 1:
                    chord_count += 1

        # 期待tick数: 3/4 = 3 * 960 = 2880
        expected_ticks = ts_num * (960 * 4 // ts_den)

        # 検証
        tick_diff = abs(total_ticks - expected_ticks)
        tick_ok = tick_diff <= 10  # 丸め誤差許容

        if not tick_ok:
            issues.append(f"M{mi+1} V{vi+1}: tick不整合 {total_ticks} vs expected {expected_ticks} (diff={tick_diff})")

        if vi == 0:  # Voice 1 (melody)
            if rest_count > 0:
                issues.append(f"M{mi+1} V1: REST {rest_count}個 (理想は0)")
            if note_count < 8:
                issues.append(f"M{mi+1} V1: ノート不足 {note_count}個 (理想は9)")
            if chord_count > 0:
                issues.append(f"M{mi+1} V1: 和音化 {chord_count}箇所 (アルペジオは全て単音のはず)")

        # 簡易表示
        status = "OK" if tick_ok and (vi > 0 or (rest_count == 0 and note_count >= 8 and chord_count == 0)) else "NG"
        print(f"  M{mi+1:2d} V{vi+1}: {status} notes={note_count:2d} rests={rest_count} chords={chord_count} ticks={total_ticks}/{expected_ticks} tuplet={has_tuplet}")

print(f"\n--- 検証結果 ---")
print(f"総小節数: {total_measures}")
print(f"問題なし: {total_measures - len(set(i.split(':')[0] for i in issues))}/{total_measures} 小節")
print(f"検出された問題: {len(issues)}件")
for iss in issues:
    print(f"  ⚠ {iss}")

# 禁じられた遊び特有の検証: 最初の8小節のパターン
print(f"\n--- パターン検証 (最初の8小節) ---")
expected_pattern_m1 = "B4 B3 G3 B4 B3 G3 B4 B3 G3"  # Romance bar 1
note_names = {40:'E2', 45:'A2', 50:'D3', 55:'G3', 59:'B3', 64:'E4',
              67:'G4', 69:'A4', 71:'B4', 66:'F#4', 60:'C4', 62:'D4',
              76:'E5', 72:'C5', 74:'D5'}
for mi in range(min(4, total_measures)):
    m = t.measures[mi]
    v1_notes = []
    for b in m.voices[0].beats:
        if 'rest' not in str(b.status).lower():
            for n in b.notes:
                pitch_name = f"s{n.string}f{n.value}"
                v1_notes.append(pitch_name)
    v2_notes = []
    if len(m.voices) > 1:
        for b in m.voices[1].beats:
            if 'rest' not in str(b.status).lower():
                for n in b.notes:
                    v2_notes.append(f"s{n.string}f{n.value}")
    print(f"  M{mi+1}: melody=[{', '.join(v1_notes)}]  bass=[{', '.join(v2_notes)}]")
