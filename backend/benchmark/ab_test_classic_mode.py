"""
ab_test_classic_mode.py
========================
禁じられた遊び（romance.wav）を対象に、
Standard Mode (Phase 6.5) vs Classic Mode (Arpeggio特化) のA/Bテストを実施・評価する。
"""

import os
import sys
import json
import time
import shutil
from pathlib import Path

# Add backend to path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from pipeline import run_pipeline

PROJECT_ROOT = BACKEND_DIR.parent
AUDIO_PATH = PROJECT_ROOT / "uploads" / "20260816-012216-469ce5" / "romance.wav"
if not AUDIO_PATH.exists():
    # Fallback to other romance.wav if present
    candidates = list((PROJECT_ROOT / "uploads").glob("*/romance.wav"))
    if candidates:
        AUDIO_PATH = candidates[-1]

print(f"=== SoloTab A/B Test: Romance (Forbidden Games) ===")
print(f"Audio Source: {AUDIO_PATH}")
assert AUDIO_PATH.exists(), f"Audio file not found: {AUDIO_PATH}"

BENCH_OUT_DIR = BACKEND_DIR / "benchmark" / "ab_test_output"
BENCH_OUT_DIR.mkdir(parents=True, exist_ok=True)

modes = [
    {"name": "Standard Mode (Phase 6.5)", "profile": "standard", "dir": BENCH_OUT_DIR / "standard"},
    {"name": "Classic / Arpeggio Mode", "profile": "classic", "dir": BENCH_OUT_DIR / "classic"},
]

results = {}

for m in modes:
    m_dir = m["dir"]
    if m_dir.exists():
        shutil.rmtree(m_dir)
    m_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n" + "="*50)
    print(f"Running Mode: {m['name']} (profile={m['profile']})")
    print(f"="*50)
    
    t0 = time.time()
    res = run_pipeline(
        session_id=f"ab_{m['profile']}",
        session_dir=m_dir,
        wav_path=AUDIO_PATH,
        tuning_name="standard",
        title=f"Romance_{m['profile']}",
        skip_demucs=True,  # Solo classical guitar
        fast_moe=True,
        guitar_type="nylon",
        transcription_profile=m["profile"],
    )
    elapsed = time.time() - t0
    
    # Load notes.json
    notes_file = m_dir / "notes.json"
    with open(notes_file, "r", encoding="utf-8") as f:
        notes_data = json.load(f)
    
    notes = notes_data.get("notes", [])
    
    # Metrics
    total_notes = len(notes)
    # Calculate Inter-Onset Intervals (IOI) to measure triplet / arpeggio resolution
    onsets = sorted([n["start"] for n in notes])
    iois = [onsets[i+1] - onsets[i] for i in range(len(onsets)-1)] if len(onsets) > 1 else []
    fast_notes = sum(1 for ioi in iois if 0.04 <= ioi <= 0.25)  # Fast arpeggios / triplets
    
    # Pitch distribution (bass: pitch < 55 vs melody/treble: pitch >= 55)
    bass_notes = sum(1 for n in notes if n["pitch"] < 55)
    treble_notes = sum(1 for n in notes if n["pitch"] >= 55)
    
    # String distribution
    string_counts = {}
    for n in notes:
        s = n.get("string", 0)
        string_counts[s] = string_counts.get(s, 0) + 1
        
    results[m["profile"]] = {
        "name": m["name"],
        "elapsed_sec": round(elapsed, 2),
        "total_notes": total_notes,
        "fast_notes_triplets": fast_notes,
        "bass_notes": bass_notes,
        "treble_notes": treble_notes,
        "string_counts": string_counts,
        "model_stats": notes_data.get("model_stats", {}),
        "xml_path": str(m_dir / "tab.musicxml"),
        "gp5_path": str(m_dir / "tab.gp5"),
    }

print("\n" + "="*60)
print("=== A/B TEST COMPARISON SUMMARY ===")
print("="*60)

for p, r in results.items():
    print(f"\n[{r['name']}]")
    print(f"  - Elapsed Time: {r['elapsed_sec']}s")
    print(f"  - Total Notes Detected: {r['total_notes']}")
    print(f"  - Fast Arpeggio/Triplet Notes (IOI 40-250ms): {r['fast_notes_triplets']}")
    print(f"  - Bass Notes (E2~G3): {r['bass_notes']}, Treble Notes (A3~E6): {r['treble_notes']}")
    print(f"  - String Distribution: {r['string_counts']}")
    print(f"  - Model Stats: {r['model_stats']}")

# Save comparison JSON
with open(BENCH_OUT_DIR / "ab_test_report.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nReport saved to: {BENCH_OUT_DIR / 'ab_test_report.json'}")
