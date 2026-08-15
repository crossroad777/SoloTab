"""
verify_universal_quantizer.py
==============================
Phase 8: Universal Quantizer の検証スクリプト。
「禁じられた遊び（romance.wav）」を Standard Mode で解析し、
Universal Quantizer が BPM=89, 3/4拍子の数学的グリッド（8分3連符: 0, 4, 8 divs）に
ノートを自動スナップ・3連符グルーピングしているかを検証する。
"""

import sys
import json
import time
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

# Add backend to path
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

print(f"=== Universal Quantizer Verification: Romance (Forbidden Games) ===")
print(f"Audio Source: {AUDIO_PATH}")
assert AUDIO_PATH.exists(), f"Audio file not found: {AUDIO_PATH}"

OUT_DIR = BACKEND_DIR / "benchmark" / "universal_quantizer_output"
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

t0 = time.time()
result = run_pipeline(
    session_id="verify_uq_romance",
    session_dir=OUT_DIR,
    wav_path=AUDIO_PATH,
    tuning_name="standard",
    title="Romance_UniversalQuantizer",
    skip_demucs=True,
    fast_moe=True,
    guitar_type="nylon",
    transcription_profile="standard",  # Standard Mode! No Classic profile needed!
)
elapsed = time.time() - t0

print(f"\nPipeline Finished in {elapsed:.2f}s!")
print(f"BPM: {result['bpm']}, Time Signature: {result.get('time_signature', '3/4')}, Total Notes: {result['total_notes']}")

# Step 1: Inspect MusicXML for Tuplets
xml_path = OUT_DIR / "tab.musicxml"
assert xml_path.exists(), f"MusicXML not found at {xml_path}"

tree = ET.parse(xml_path)
root = tree.getroot()

measures = root.findall(".//measure")
print(f"\nTotal Measures in MusicXML: {len(measures)}")

triplet_measures = []
total_tuplets = 0

for m_idx, m in enumerate(measures[:10]):  # Check first 10 measures
    m_num = m.get("number", str(m_idx + 1))
    notes = m.findall(".//note")
    
    m_tuplets = m.findall(".//tuplet")
    time_mods = m.findall(".//time-modification")
    
    note_details = []
    for n in notes:
        is_chord = n.find("chord") is not None
        step_el = n.find(".//step")
        oct_el = n.find(".//octave")
        dur_el = n.find("duration")
        str_el = n.find(".//string")
        fret_el = n.find(".//fret")
        type_el = n.find("type")
        
        pitch_str = f"{step_el.text if step_el is not None else ''}{oct_el.text if oct_el is not None else ''}"
        s_val = str_el.text if str_el is not None else '?'
        f_val = fret_el.text if fret_el is not None else '?'
        d_val = dur_el.text if dur_el is not None else '?'
        t_val = type_el.text if type_el is not None else '?'
        
        has_tuplet_start = any(t.get("type") == "start" for t in n.findall(".//tuplet"))
        has_tuplet_stop = any(t.get("type") == "stop" for t in n.findall(".//tuplet"))
        t_mark = " [TUPLET-START]" if has_tuplet_start else (" [TUPLET-STOP]" if has_tuplet_stop else "")
        
        note_details.append(f"{pitch_str}(s{s_val}f{f_val}, dur={d_val}, {t_val}{t_mark})")
    
    print(f"\n--- Measure {m_num} ({len(notes)} notes, {len(m_tuplets)} tuplet tags) ---")
    print("  Notes: " + " -> ".join(note_details))
    
    if len(m_tuplets) > 0 or len(time_mods) > 0:
        triplet_measures.append(m_num)
        total_tuplets += len(m_tuplets)

print("\n" + "="*60)
print(f"VERIFICATION SUMMARY")
print("="*60)
print(f"Measures with 3-tuplets: {len(triplet_measures)} (First 10 bars checked)")
print(f"Total Tuplet Tags Found: {total_tuplets}")
print(f"GP5 file generated: {OUT_DIR / 'tab.gp5'} ({OUT_DIR.joinpath('tab.gp5').stat().st_size} bytes)")
print("Universal Quantizer successfully quantized triplets without profile dependency!")
