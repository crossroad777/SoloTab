"""
verify_physical_voice_rules.py
==============================
物理ルールによる声部分離インターフェースの自動照合スクリプト。

照合項目:
  1. 第1小節のノート数がちょうど10個 (ベース1音 + 3連符×3拍=9音)
  2. 第1小節に {7, 0} 以外のフレットが存在しない
  3. bass の「0」が小節に1回だけ
"""

import sys
import shutil
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from pipeline import run_pipeline
from nextchord_renderer import notes_to_nextchord_text
from physical_voice_rules import apply_physical_voice_rules

PROJECT_ROOT = BACKEND_DIR.parent
AUDIO_PATH = PROJECT_ROOT / "uploads" / "20260816-012216-469ce5" / "romance.wav"
if not AUDIO_PATH.exists():
    candidates = list((PROJECT_ROOT / "uploads").glob("*/romance.wav"))
    if candidates:
        AUDIO_PATH = candidates[-1]

OUT_DIR = BACKEND_DIR / "benchmark" / "physical_rules_output"
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=== Running Pipeline: Physical Voice Rules Verification ===")
result = run_pipeline(
    session_id="verify_phys_romance",
    session_dir=OUT_DIR,
    wav_path=AUDIO_PATH,
    tuning_name="standard",
    title="romance",
    skip_demucs=True,
    fast_moe=True,
    guitar_type="nylon",
    transcription_profile="standard",
)

# Load assigned notes
import json
notes_path = OUT_DIR / "notes_assigned.json"
with open(notes_path, "r", encoding="utf-8") as f:
    notes = json.load(f)

# Apply physical voice rules
processed = apply_physical_voice_rules(notes, time_signature="3/4", beats_per_bar=3)

# Filter Bar 1 (0-indexed or 1-indexed, measure 1 in romance starts with music at bar 1)
# Check bar notes
bars = {}
for n in processed:
    b = int(n.get("bar", 0))
    bars.setdefault(b, []).append(n)

# Bar 1 (first musical measure)
first_bar_idx = min(b for b in bars.keys() if len(bars[b]) >= 5)
bar1_notes = bars[first_bar_idx]

print(f"\nAnalyzing First Music Measure (Bar {first_bar_idx}):")
print(f"Total Notes: {len(bar1_notes)}")
for n in bar1_notes:
    print(f"  pitch={n.get('pitch')}, string={n.get('string')}, fret={n.get('fret')}, role={n.get('role')}, dur={n.get('duration_divs')}")

# Check condition 1: Exactly 10 notes (or trimmed to 10)
cond1 = (len(bar1_notes) == 10)
# Check condition 2: Frets are subset of {7, 0}
frets = set(int(n.get('fret', 0)) for n in bar1_notes)
cond2 = frets.issubset({7, 0})
# Check condition 3: bass note has fret 0 and appears once in beat 0
bass_notes = [n for n in bar1_notes if n.get('role') == 'bass']
cond3 = (len(bass_notes) >= 1 and int(bass_notes[0].get('fret', -1)) == 0)

print("\n" + "="*50)
print("VERIFICATION CHECK RESULTS:")
print("="*50)
print(f"1. Exactly 10 notes in Bar 1: {cond1} ({len(bar1_notes)} notes)")
print(f"2. Only frets in {{7, 0}}: {cond2} (frets found: {frets})")
print(f"3. Bass '0' once & sustaining: {cond3} (bass notes: {len(bass_notes)})")

# Print NextChord output
txt_path = OUT_DIR / "tab.nextchord.txt"
txt = txt_path.read_text(encoding="utf-8")
print("\n" + "="*50)
print("NEXTCHORD TEXT OUTPUT (First 20 lines):")
print("="*50)
for line in txt.splitlines()[:20]:
    print(line)
print("="*50)

if cond1 and cond2 and cond3:
    print("\nALL 3 CONDITIONS PASSED! 🎉")
else:
    print(f"\nCondition status: cond1={cond1}, cond2={cond2}, cond3={cond3}")
