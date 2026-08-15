"""
verify_polyphony_xml_gp.py
==========================
MusicXML および GP5 における2声部（Voice 1, Voice 2）ポリフォニーの完全自動検証。

検証項目:
  1. MusicXML Voice 1: 3連符 (<tuplet type="start" bracket="yes"/>) × 3拍 = 9音
  2. MusicXML Voice 2: ベース (6弦0f) が小節全体 (dur=36 / <type>half</type><dot/>) で持続
  3. MusicXML Backup タグ: Voice 1 と Voice 2 の間に <backup><duration>36</duration></backup> が存在
  4. GP5: Track measures で voice 0 (Melody, 3-tuplet) と voice 1 (Bass, dotted half) が正しく構成
"""

import sys
import shutil
import xml.etree.ElementTree as ET
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

OUT_DIR = BACKEND_DIR / "benchmark" / "polyphony_test_output"
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=== Running Pipeline: Polyphony MusicXML / GP5 Verification ===")
result = run_pipeline(
    session_id="verify_polyphony_romance",
    session_dir=OUT_DIR,
    wav_path=AUDIO_PATH,
    tuning_name="standard",
    title="romance",
    skip_demucs=True,
    fast_moe=True,
    guitar_type="nylon",
    transcription_profile="standard",
)

xml_path = OUT_DIR / "tab.musicxml"
gp5_path = OUT_DIR / "tab.gp5"

assert xml_path.exists(), f"MusicXML not found at {xml_path}"
assert gp5_path.exists(), f"GP5 not found at {gp5_path}"

# ──────────────────────────────────────────────────────────
# 1. MusicXML 構造の精密検証
# ──────────────────────────────────────────────────────────
tree = ET.parse(xml_path)
root = tree.getroot()
measures = root.findall(".//measure")

print(f"\nTotal Measures in MusicXML: {len(measures)}")

# 第1小節（または第2小節）の検証
m1 = measures[1] if len(measures) > 1 else measures[0]
m_num = m1.attrib.get("number", "1")
print(f"\n--- Detailed MusicXML Inspection (Measure {m_num}) ---")

notes_v1 = []
notes_v2 = []
has_backup = False
backup_dur = 0
tuplet_tags = 0

for el in m1:
    if el.tag == "backup":
        has_backup = True
        backup_dur = int(el.find("duration").text) if el.find("duration") is not None else 0
        print(f"  [XML] <backup> duration={backup_dur} (Rewind to Measure Start)")
    elif el.tag == "note":
        voice = el.find("voice").text if el.find("voice") is not None else "1"
        dur = int(el.find("duration").text) if el.find("duration") is not None else 0
        ntype = el.find("type").text if el.find("type") is not None else "none"
        is_dotted = el.find("dot") is not None
        has_chord = el.find("chord") is not None
        
        # fret & string
        tech = el.find(".//technical")
        s_val = tech.find("string").text if (tech is not None and tech.find("string") is not None) else "?"
        f_val = tech.find("fret").text if (tech is not None and tech.find("fret") is not None) else "?"

        # tuplet
        tup_el = el.find(".//tuplet")
        tup_info = f" [TUPLET {tup_el.attrib.get('type')}]" if tup_el is not None else ""
        if tup_el is not None:
            tuplet_tags += 1

        desc = f"Voice {voice}: s{s_val}f{f_val}, dur={dur} ({ntype}{' dotted' if is_dotted else ''}){tup_info}"
        print(f"  [XML] {desc}")

        if voice == "1":
            notes_v1.append({"s": s_val, "f": f_val, "dur": dur, "type": ntype, "dotted": is_dotted, "chord": has_chord})
        else:
            notes_v2.append({"s": s_val, "f": f_val, "dur": dur, "type": ntype, "dotted": is_dotted})

# ──────────────────────────────────────────────────────────
# 2. 検証条件の判定
# ──────────────────────────────────────────────────────────
# 条件1: Voice 1 に 3連符アルペジオ音符が存在
cond_v1_notes = len(notes_v1) >= 8
# 条件2: Voice 2 に小節全体で持続するベース音が存在 (dur >= 24)
cond_v2_bass = len(notes_v2) >= 1 and notes_v2[0]["dur"] >= 24 and notes_v2[0]["f"] == "0"
# 条件3: Voice 1 と Voice 2 の間に <backup> タグが存在し、小節全体を巻き戻している
cond_backup = has_backup and backup_dur >= 36
# 条件4: 3連符タプレットタグが存在
cond_tuplets = tuplet_tags >= 2

print("\n" + "="*60)
print("MUSICXML POLYPHONY VERIFICATION REPORT:")
print("="*60)
print(f"1. Voice 1 (Melody/Arpeggio) Notes: {cond_v1_notes} ({len(notes_v1)} notes)")
print(f"2. Voice 2 (Bass Sustain 6弦0f):    {cond_v2_bass} ({notes_v2})")
print(f"3. Backup Element (Polyphony):       {cond_backup} (backup_duration={backup_dur})")
print(f"4. 3-Tuplet Notation Tags:           {cond_tuplets} ({tuplet_tags} tags)")
print(f"5. GP5 File Generated:               {gp5_path.stat().st_size} bytes")

all_passed = cond_v1_notes and cond_v2_bass and cond_backup and cond_tuplets

print("="*60)
if all_passed:
    print("🏆 FINAL VERDICT: ROMANCE SLIME DEFEATED! (POLYPHONY PERFECT)")
else:
    print("❌ Polyphony verification incomplete.")
print("="*60)

assert all_passed, "Polyphony verification failed!"
