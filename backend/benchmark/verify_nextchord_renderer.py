"""
verify_nextchord_renderer.py
============================
romance.wav を解析し、NextChord SoloTab 形式のテキスト出力を検証する。
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

OUT_DIR = BACKEND_DIR / "benchmark" / "nextchord_test_output"
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Running pipeline with NextChord Renderer for Romance...")
result = run_pipeline(
    session_id="verify_nextchord_romance",
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
lines = content.splitlines()

print("\n" + "="*50)
print("NEXTCHORD SOLOTAB OUTPUT (First 25 lines):")
print("="*50)
for l in lines[:25]:
    print(l)
print("="*50)

# Check elements
has_header = "■= 89  3/4 NextChord SoloTab" in lines[1]
has_tuning = "e B G D A E" in lines[2]
has_em = any(l == "Em" for l in lines)

print(f"\nHeader Valid: {has_header}")
print(f"Tuning Valid: {has_tuning}")
print(f"Chord Em Valid: {has_em}")
print(f"Total lines: {len(lines)}")
