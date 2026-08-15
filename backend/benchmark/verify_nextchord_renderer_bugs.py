"""
verify_nextchord_renderer_bugs.py
=================================
NextChord SoloTab レンダラーの4大バグ修正の自動検証スクリプト。

検証条件:
  1. コード: "Em" のみ (D7等の混在ゼロ)
  2. 数字列: 1小節目に [1弦7, 2弦0, 3弦0] の3連符パターン×3回 (計9音) + ベース "0" = 計10音
  3. 不正な結合数字（70等）や異常フレット（8等）がゼロ
"""

import sys
import shutil
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from pipeline import run_pipeline

PROJECT_ROOT = BACKEND_DIR.parent
AUDIO_PATH = PROJECT_ROOT / "uploads" / "20260816-012216-469ce5" / "romance.wav"
if not AUDIO_PATH.exists():
    candidates = list((PROJECT_ROOT / "uploads").glob("*/romance.wav"))
    if candidates:
        AUDIO_PATH = candidates[-1]

OUT_DIR = BACKEND_DIR / "benchmark" / "nextchord_bugfix_output"
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=== Running Pipeline: NextChord Bugfix Verification ===")
result = run_pipeline(
    session_id="verify_nextchord_bugfix",
    session_dir=OUT_DIR,
    wav_path=AUDIO_PATH,
    tuning_name="standard",
    title="romance",
    skip_demucs=True,
    fast_moe=True,
    guitar_type="nylon",
    transcription_profile="standard",
)

txt_path = OUT_DIR / "tab.nextchord.txt"
assert txt_path.exists(), f"tab.nextchord.txt not found at {txt_path}"

content = txt_path.read_text(encoding="utf-8")
lines = [l.strip() for l in content.splitlines() if l.strip()]

print("\n" + "="*50)
print("NEXTCHORD TEXT OUTPUT (First 25 lines):")
print("="*50)
for l in lines[:25]:
    print(l)
print("="*50)

# Check 1: Header verification
header_valid = (lines[0] == "romance" and "■= 89  3/4 NextChord SoloTab" in lines[1] and lines[2] == "e B G D A E")

# Check 2: First measure verification
# Expected lines:
# Em
# 7
# 0
# 0 0 7 0 0 7 0 0
bar1_chord = lines[3]
bar1_top = lines[4]
bar1_bass = lines[5]
bar1_seq = lines[6]

check_chord = (bar1_chord == "Em")
check_top = (bar1_top == "7")
check_bass = (bar1_bass == "0")
check_seq = (bar1_seq == "0 0 7 0 0 7 0 0")

# Check 3: Check for invalid concatenated numbers (70, etc) or out-of-key frets (8) in measure 1
invalid_tokens = ["70", "8", "D7"]
has_invalid = any(inv in "\n".join(lines[3:7]) for inv in invalid_tokens)

print("\n" + "="*50)
print("AUTOMATED VERIFICATION REPORT:")
print("="*50)
print(f"1. Header Format: {header_valid}")
print(f"2. Measure 1 Chord ('Em' only): {check_chord} ({bar1_chord})")
print(f"3. Measure 1 Top Note ('7'): {check_top} ({bar1_top})")
print(f"4. Measure 1 Bass Note ('0'): {check_bass} ({bar1_bass})")
print(f"5. Measure 1 Arpeggio Sequence ('0 0 7 0 0 7 0 0'): {check_seq} ({bar1_seq})")
print(f"6. Invalid Tokens (70, 8, D7): {'None (PASSED)' if not has_invalid else 'FOUND (FAILED)'}")

all_passed = header_valid and check_chord and check_top and check_bass and check_seq and (not has_invalid)

if all_passed:
    print("\n🎉 ALL 4 STRUCTURAL BUGS FIXED & VERIFIED SUCCESSFULLY!")
else:
    print("\n❌ VERIFICATION FAILED. Review output above.")

assert all_passed, "Verification failed!"
