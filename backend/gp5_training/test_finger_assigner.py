"""
finger_assigner テスト — v6 ポジション一貫性スムージング検証
"""
import sys
sys.path.insert(0, r'D:\Music\nextchord-solotab\backend')
sys.stdout.reconfigure(encoding='utf-8')

from finger_assigner import assign_fingers

FINGER_NAMES = {0: "Open", 1: "Index", 2: "Middle", 3: "Ring", 4: "Pinky"}
passed = 0
failed = 0
total = 0


def test(name, notes, expected=None):
    """Run a test case and optionally verify expected finger assignments.
    expected: list of (fret, expected_finger) tuples, or None for display only.
    """
    global passed, failed, total
    print(f"\n{'='*60}")
    print(f"  テスト: {name}")
    print(f"{'='*60}")
    # 事前に右手PIMAを動的アサイン（左手遷移コストの協調のため）
    try:
        from string_assigner import _assign_right_hand_fingers
        notes = _assign_right_hand_fingers(notes)
    except Exception as e:
        print(f"  (Warning: PIMA pre-assignment failed: {e})")

    result = assign_fingers(notes)
    for n in result:
        f = n.get('left_hand_finger', '?')
        fn = FINGER_NAMES.get(f, '?')
        fret = n.get('fret', 0)
        pos_str = f"(pos={fret-(f-1)})" if isinstance(f, int) and f > 0 else ""
        print(f"  弦{n.get('string','?')} F{fret:2d} → {fn}({f}) {pos_str}")

    if expected is not None:
        fretted = [n for n in result if n.get('fret', 0) > 0]
        for i, (exp_fret, exp_finger) in enumerate(expected):
            total += 1
            if i >= len(fretted):
                print(f"  ✗ ノート{i} 不足")
                failed += 1
                continue
            actual_fret = fretted[i].get('fret', 0)
            actual_finger = fretted[i].get('left_hand_finger', -1)
            if actual_fret != exp_fret:
                print(f"  ⚠ ノート{i} フレット不一致: got F{actual_fret}, expected F{exp_fret}")
            is_correct = False
            if isinstance(exp_finger, (list, tuple, set)):
                is_correct = actual_finger in exp_finger
            else:
                is_correct = actual_finger == exp_finger

            if is_correct:
                print(f"  ✓ F{exp_fret} → finger {actual_finger} OK")
                passed += 1
            else:
                print(f"  ✗ F{exp_fret} → finger {actual_finger}, expected {exp_finger}")
                failed += 1
    return result


# ================================================================
# 既存テスト（回帰確認）
# ================================================================

# 1. ロマンス Em: B(7f)とE(0f)のアルペジオ
# ポジション4: 7f → finger 4 (pinky) が理想的
test("Romance Em (7f-0f-7f-0f)", [
    {"string": 6, "fret": 0, "pitch": 40, "start": 0.0},
    {"string": 1, "fret": 7, "pitch": 71, "start": 0.1},
    {"string": 2, "fret": 0, "pitch": 59, "start": 0.2},
    {"string": 1, "fret": 7, "pitch": 71, "start": 0.3},
    {"string": 2, "fret": 0, "pitch": 59, "start": 0.4},
    {"string": 1, "fret": 7, "pitch": 71, "start": 0.5},
    {"string": 2, "fret": 0, "pitch": 59, "start": 0.6},
])

# 2. Romance 2小節目: 5f-0f-3f-0f アルペジオ
# ポジション3付近: 5f→finger3, 3f→finger1 が理想
test("Romance 2小節 (5f-0f-3f-0f)", [
    {"string": 6, "fret": 0, "pitch": 40, "start": 0.0},
    {"string": 1, "fret": 5, "pitch": 69, "start": 0.1},
    {"string": 2, "fret": 0, "pitch": 59, "start": 0.2},
    {"string": 1, "fret": 3, "pitch": 67, "start": 0.3},
    {"string": 2, "fret": 0, "pitch": 59, "start": 0.4},
    {"string": 1, "fret": 0, "pitch": 64, "start": 0.5},
    {"string": 2, "fret": 0, "pitch": 59, "start": 0.6},
])

# 3. Cメジャーコード: 5弦3f, 4弦2f, 3弦0f, 2弦1f, 1弦0f
# 期待: 5弦3f→finger3, 4弦2f→finger2, 2弦1f→finger1
test("C major chord", [
    {"string": 5, "fret": 3, "pitch": 48, "start": 0.0},
    {"string": 4, "fret": 2, "pitch": 52, "start": 0.0},
    {"string": 3, "fret": 0, "pitch": 55, "start": 0.0},
    {"string": 2, "fret": 1, "pitch": 60, "start": 0.0},
    {"string": 1, "fret": 0, "pitch": 64, "start": 0.0},
], expected=[(3, 3), (2, 2), (1, 1)])

# ================================================================
# スケールランテスト（v6新機能: 指順序強制）
# ================================================================

# 4. スケールラン: 5f→7f→8f→10f (同弦)
# ポジション5: 5f→1, 7f→3, 8f→4 (10fは範囲外 → そのまま)
test("Scale run 5-7-8-10", [
    {"string": 1, "fret": 5, "pitch": 69, "start": 0.0},
    {"string": 1, "fret": 7, "pitch": 71, "start": 0.15},
    {"string": 1, "fret": 8, "pitch": 72, "start": 0.30},
    {"string": 1, "fret": 10, "pitch": 74, "start": 0.45},
])

# 5. ポジション移動: 1f→2f→3f → 5f→7f→8f (phrase_gap=0.5超えで分割)
# 前半: pos1 → 1f=1, 2f=2, 3f=3
# 後半: CNN v2 may use pinky stretch (pos1: 5f=4) or shift to pos5 (5f=1)
# Both are valid guitar fingerings; CNN v2 (95.4%) prefers stretch
test("Position shift 1-2-3 → 5-7-8", [
    {"string": 1, "fret": 1, "pitch": 65, "start": 0.0},
    {"string": 1, "fret": 2, "pitch": 66, "start": 0.15},
    {"string": 1, "fret": 3, "pitch": 67, "start": 0.30},
    {"string": 1, "fret": 5, "pitch": 69, "start": 1.2},   # gap > 0.5
    {"string": 1, "fret": 7, "pitch": 71, "start": 1.35},
    {"string": 1, "fret": 8, "pitch": 72, "start": 1.50},
], expected=[(1, 1), (2, 2), (3, 3), (5, 1), (7, 3), (8, 4)])

# ================================================================
# v6 新機能テスト: ポジション一貫性スムージング
# ================================================================

# 6. 同一フレーズ内のポジション一貫性
# pos=5を基準: 5f→1, 6f→2, 7f→3, 8f→4, 5f→1
test("Position consistency (pos5)", [
    {"string": 1, "fret": 5, "pitch": 69, "start": 0.0},
    {"string": 2, "fret": 6, "pitch": 65, "start": 0.15},
    {"string": 1, "fret": 7, "pitch": 71, "start": 0.30},
    {"string": 2, "fret": 8, "pitch": 67, "start": 0.45},
    {"string": 1, "fret": 5, "pitch": 69, "start": 0.60},
], expected=[(5, 1), (6, 2), (7, 3), (8, 4), (5, 1)])

# 7. 開放弦混在のアルペジオ（ポジション一貫性）
# CNNの信頼度が高い場合はCNN予測を尊重する設計
# 2f→finger2(CNN), 3f→finger2(CNN), 2f→finger2(CNN), 4f→finger4(CNN)
test("Arpeggio with open strings", [
    {"string": 4, "fret": 2, "pitch": 52, "start": 0.0},
    {"string": 3, "fret": 0, "pitch": 55, "start": 0.1},
    {"string": 2, "fret": 3, "pitch": 62, "start": 0.2},
    {"string": 1, "fret": 0, "pitch": 64, "start": 0.3},
    {"string": 4, "fret": 2, "pitch": 52, "start": 0.4},
    {"string": 3, "fret": 0, "pitch": 55, "start": 0.5},
    {"string": 2, "fret": 4, "pitch": 63, "start": 0.6},
])

# 8. 下降スケールラン: 8f→7f→5f→3f (同弦)
test("Descending scale run 8-7-5-3", [
    {"string": 1, "fret": 8, "pitch": 72, "start": 0.0},
    {"string": 1, "fret": 7, "pitch": 71, "start": 0.15},
    {"string": 1, "fret": 5, "pitch": 69, "start": 0.30},
    {"string": 1, "fret": 3, "pitch": 67, "start": 0.45},
])

# 9. 第1ポジション基本: 1f→2f→3f→4f
test("First position basic 1-2-3-4", [
    {"string": 1, "fret": 1, "pitch": 65, "start": 0.0},
    {"string": 1, "fret": 2, "pitch": 66, "start": 0.15},
    {"string": 1, "fret": 3, "pitch": 67, "start": 0.30},
    {"string": 1, "fret": 4, "pitch": 68, "start": 0.45},
], expected=[(1, 1), (2, 2), (3, 3), (4, 4)])

# 10. 全開放弦（指なし）
test("All open strings", [
    {"string": 6, "fret": 0, "pitch": 40, "start": 0.0},
    {"string": 5, "fret": 0, "pitch": 45, "start": 0.1},
    {"string": 4, "fret": 0, "pitch": 50, "start": 0.2},
    {"string": 3, "fret": 0, "pitch": 55, "start": 0.3},
    {"string": 2, "fret": 0, "pitch": 59, "start": 0.4},
    {"string": 1, "fret": 0, "pitch": 64, "start": 0.5},
])

# ================================================================
# v13.0 新機能テスト: 特殊奏法制約（ビブラート、サポート競合、レガート・スライド）
# ================================================================

# 11. ビブラート時の小指回避
# 通常ポジション7では小指(4)が選ばれやすいフレーズでも、vibratoがあるため薬指(3)や中指(2)を優先
test("Vibrato pinky avoidance (fret 7)", [
    {"string": 1, "fret": 7, "pitch": 71, "start": 0.0, "vibrato": True}
], expected=[(7, (2, 3))])

# 12. チョーキング時のサポート指競合回避 (ブルースのダブルストップ)
# 1弦7fを薬指(3)でチョーキング、同時発音の2弦5fを人差し指(1)で押弦。
# 人差し指(1)は中指(2)をサポートとして動員可能なので問題なし
test("Choking support double stop (5f-7f)", [
    {"string": 1, "fret": 7, "pitch": 71, "start": 0.0, "technique": "bend"},
    {"string": 2, "fret": 5, "pitch": 65, "start": 0.0}
], expected=[(7, 3), (5, 1)])

# 13. スライド時の同一指維持 (5f -> 7f slide_up)
# 同一の指でフレットをスライドさせる
test("Slide transition (5f -> 7f)", [
    {"string": 1, "fret": 5, "pitch": 69, "start": 0.0},
    {"string": 1, "fret": 7, "pitch": 71, "start": 0.1, "technique": "slide_up"}
], expected=[(5, 1), (7, 1)])

# 14. ハンマリング時の指順正当性 (5f -> 7f hammer_on)
# ターゲット指はソース指より高くなる（1 -> 3 など）
test("Hammer-on transition (5f -> 7f)", [
    {"string": 1, "fret": 5, "pitch": 69, "start": 0.0},
    {"string": 1, "fret": 7, "pitch": 71, "start": 0.15, "technique": "hammer_on"}
], expected=[(5, 1), (7, 3)])

# ================================================================
# v14.0 新機能テスト: 音価保持ルール（ベースサステイン）
# ================================================================

# 15. ベース音サステイン中の高音メロディ運指
# 6弦3f (G音) がサステイン中のため、ポジション1を維持しつつ、他の指でメロディを弾く。
# 6弦3f -> finger 3 (薬指), 2弦1f -> finger 1 (人差し指), 1弦3f -> finger 3 (薬指) or 4 (小指), 1弦1f -> finger 1 (人差し指)
# 期待: 6弦3fは finger 3 で固定、ポジション1が維持される
test("Bass sustain prolongation (6弦3f G sustain + melody)", [
    {"string": 6, "fret": 3, "pitch": 43, "start": 0.0, "duration": 1.0},
    {"string": 2, "fret": 1, "pitch": 60, "start": 0.2, "duration": 0.2},
    {"string": 1, "fret": 3, "pitch": 67, "start": 0.4, "duration": 0.2},
    {"string": 1, "fret": 1, "pitch": 65, "start": 0.6, "duration": 0.2},
], expected=[(3, 3), (1, 1), (3, 3), (1, 1)])

# ================================================================
# v15.0 新機能テスト: フレット幅依存のポジションシフトコスト軽減
# ================================================================

# 16. ハイポジションでのポジションシフトの容易さ
# 12f -> 14f -> 15f -> 17f のように、ハイポジションではフレット幅が狭いため、
# ポジション移動コストが軽減（約0.4倍）され、無理なストレッチをせずに自然なポジション移動が選択される。
# 期待: 12f -> finger 1 (pos=12), 14f -> finger 3 (pos=12), 15f -> finger 4 (pos=12), 17f -> finger 4 (pos=14)
test("High position shift easing (12f-14f-15f-17f)", [
    {"string": 1, "fret": 12, "pitch": 76, "start": 0.0},
    {"string": 1, "fret": 14, "pitch": 78, "start": 0.15},
    {"string": 1, "fret": 15, "pitch": 79, "start": 0.30},
    {"string": 1, "fret": 17, "pitch": 81, "start": 0.45},
], expected=[(12, 1), (14, 3), (15, 4), (17, 4)])

# ================================================================
# v16.0 新機能テスト: 指の疲労度コストと手首ねじれペナルティ
# ================================================================

# 17. 低音弦（5弦・6弦）における疲労・手首ねじれペナルティによる小指回避
# 5弦10f -> 5弦12f -> 5弦8f
# 低音弦での小指押弦はペナルティが入るが、大きなポジション移動を避けるため、
# 物理的に合理的なスパン内であれば、人差し指・薬指・小指の一貫した同一ポジション主体の運指が選択される。
# 期待: 10f -> Ring(3), 12f -> Pinky(4), 8f -> Index(1) (pos=8付近を維持)
test("Low string fatigue and wrist twist avoidance (5弦10f-12f-8f)", [
    {"string": 5, "fret": 10, "pitch": 50, "start": 0.0},
    {"string": 5, "fret": 12, "pitch": 52, "start": 0.15},
    {"string": 5, "fret": 8, "pitch": 48, "start": 0.30},
], expected=[(10, 3), (12, 4), (8, 1)])

# ================================================================
# v18.0 新機能テスト: 和音解決における指の疲労度と手首ねじれ回避
# ================================================================

# 18. 低音弦（5弦）を伴う和音における疲労・手首ねじれ回避のコード運指
# 5弦10f、4弦9f、3弦8f の和音。
# 5弦10f（低音弦・ハイフレット）で小指(4)を使用するのは疲労・手首角度ペナルティが大きいため、
# 薬指(3)・中指(2)・人差し指(1)で押さえる合理的なフォームが選択される。
# 期待: 5弦10f -> finger 3, 4弦9f -> finger 2, 3弦8f -> finger 1 (pos=8)
test("Chord fatigue and wrist twist avoidance (5弦10f/4弦9f/3弦8f chord)", [
    {"string": 5, "fret": 10, "pitch": 50, "start": 0.0},
    {"string": 4, "fret": 9,  "pitch": 54, "start": 0.0},
    {"string": 3, "fret": 8,  "pitch": 55, "start": 0.0},
], expected=[(10, 3), (9, 2), (8, 1)])

# ================================================================
# v19.0 新機能テスト: 右手運指（PIMA）動的アサインテスト
# ================================================================
def test_pima():
    global passed, failed, total
    print(f"\n{'='*60}")
    print(f"  テスト: 右手PIMA動的アサイン検証")
    print(f"{'='*60}")
    
    from string_assigner import _assign_right_hand_fingers
    
    # 1. 同弦連打時の交替指テスト (1弦を0.1秒間隔で4連打)
    notes_alt = [
        {"string": 1, "fret": 0, "pitch": 64, "start": 0.0},
        {"string": 1, "fret": 0, "pitch": 64, "start": 0.1},
        {"string": 1, "fret": 0, "pitch": 64, "start": 0.2},
        {"string": 1, "fret": 0, "pitch": 64, "start": 0.3},
    ]
    res_alt = _assign_right_hand_fingers(notes_alt)
    r_fingers_alt = [n['r_finger'] for n in res_alt]
    print(f"  1弦連打 PIMA: {r_fingers_alt} (期待: 重複なし・交替 [4, 3, 2, 3] または [4, 3, 4, 3] 等)")
    
    total += 1
    if len(r_fingers_alt) == 4 and r_fingers_alt[0] != r_fingers_alt[1] and r_fingers_alt[1] != r_fingers_alt[2] and r_fingers_alt[2] != r_fingers_alt[3]:
        print("  ✓ 同弦連打時の交替運指 OK")
        passed += 1
    else:
        print("  ✗ 同弦連打時の交替運指 NG")
        failed += 1

    # 2. 和音（同時発音）時の指重複回避テスト (5弦・4弦・3弦・2弦を同時発音)
    notes_chord = [
        {"string": 5, "fret": 3, "pitch": 48, "start": 0.0},
        {"string": 4, "fret": 2, "pitch": 52, "start": 0.0},
        {"string": 3, "fret": 0, "pitch": 55, "start": 0.0},
        {"string": 2, "fret": 1, "pitch": 60, "start": 0.0},
    ]
    res_chord = _assign_right_hand_fingers(notes_chord)
    string_to_rfinger = {n['string']: n['r_finger'] for n in res_chord}
    print(f"  和音(5,4,3,2弦) PIMA: {string_to_rfinger} (期待: 重複なし)")
    
    total += 1
    ima_fingers = [string_to_rfinger[s] for s in (2, 3, 4) if s in string_to_rfinger]
    if len(set(ima_fingers)) == len(ima_fingers) and len(ima_fingers) == 3:
        print("  ✓ 和音時の右手発音指重複回避 OK")
        passed += 1
    else:
        print("  ✗ 和音時の右手発音指重複回避 NG")
        failed += 1

# ================================================================
# v24.0 新規テスト: スケールボックスマッチングテスト
# ================================================================
def test_scale_box():
    # Am pentatonic box 1フレーズ (1弦5f, 1弦8f, 2弦5f, 2弦8f, 3弦5f, 3弦7f)
    # 期待される指アサイン: 1, 4, 1, 4, 1, 3
    notes = [
        {"string": 1, "fret": 5, "pitch": 69, "start": 0.0},
        {"string": 1, "fret": 8, "pitch": 72, "start": 0.1},
        {"string": 2, "fret": 5, "pitch": 64, "start": 0.2},
        {"string": 2, "fret": 8, "pitch": 67, "start": 0.3},
        {"string": 3, "fret": 5, "pitch": 59, "start": 0.4},
        {"string": 3, "fret": 7, "pitch": 62, "start": 0.5},
    ]
    expected = [
        (5, 1), (8, 4), (5, 1), (8, 4), (5, 1), (7, 3)
    ]
    test("Scale Box Matching (Am Pentatonic Box 1)", notes, expected)

test_scale_box()

# ================================================================
# v24.0 新規テスト: ボイスリーディング（ベースライン進行）テスト
# ================================================================
def test_voice_leading():
    # ベース順次進行フレーズ (6弦3f G -> 6弦5f A -> 6弦7f B)
    # 期待される指アサイン: 1 -> 3 -> 3 または 1 -> 1 -> 3 等（ベースラインの滑らかな進行）
    notes = [
        {"string": 6, "fret": 3, "pitch": 43, "start": 0.0},
        {"string": 6, "fret": 5, "pitch": 45, "start": 0.1},
        {"string": 6, "fret": 7, "pitch": 47, "start": 0.2},
    ]
    expected = [
        (3, 1), (5, 3), (7, (3, 4))
    ]
    test("Voice Leading (Conjunct Bass Line G-A-B)", notes, expected)

test_voice_leading()

# ================================================================
# v24.0 新規テスト: GP5 運指 Prior (N-gram 遷移頻度) テスト
# ================================================================
def test_gp5_prior():
    # 6弦2f -> 6弦2f の連続（GP5の頻出パターン "6-2-2"）
    # 期待される指アサイン: 1 -> 1
    notes = [
        {"string": 6, "fret": 2, "pitch": 42, "start": 0.0},
        {"string": 6, "fret": 2, "pitch": 42, "start": 0.1},
    ]
    expected = [
        (2, 1), (2, 1)
    ]
    test("GP5 Fingering Prior (Run 6-2-2)", notes, expected)

test_gp5_prior()

# ================================================================
# v24.0 新規テスト: 弦別指アサイン Prior テスト
# ================================================================
def test_string_finger_prior():
    # 6弦3f G音（データ駆動で6弦は人差し指(1)の割合が高い 637/1200回）
    # 期待される指アサイン: 1
    notes = [
        {"string": 6, "fret": 3, "pitch": 43, "start": 0.0},
    ]
    expected = [
        (3, 1),
    ]
    test("String Finger Prior (6th String F3)", notes, expected)

test_string_finger_prior()

test_pima()

# ================================================================
# サマリー
# ================================================================
print(f"\n{'='*60}")
print(f"  テスト結果サマリー")
print(f"{'='*60}")
print(f"  合格: {passed}/{total}")
print(f"  不合格: {failed}/{total}")
if total > 0:
    print(f"  合格率: {passed/total*100:.0f}%")
if failed == 0 and total > 0:
    print(f"  ✓ 全テスト合格！")
else:
    print(f"  ✗ {failed}件の不合格あり")
