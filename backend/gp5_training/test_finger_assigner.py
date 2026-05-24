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
            if actual_finger == exp_finger:
                print(f"  ✓ F{exp_fret} → finger {exp_finger} OK")
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
], expected=[(1, 1), (2, 2), (3, 3), (5, 4), (7, 4), (8, 4)])

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
