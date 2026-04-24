"""
augment_guitarset.py — GuitarSet データ拡張スクリプト
=====================================================
GuitarSetの音声をピッチシフトし、JAMSアノテーションのピッチ値も
対応させて拡張データを生成する。

拡張: ±1, ±2半音 = 元データの4倍追加 (合計5倍)

Usage:
    python augment_guitarset.py
"""

import os
import json
import glob
import time
import argparse
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf


GUITARSET_DIR = Path(r"c:\Users\kotan\Desktop\Datasets\GuitarSet_mirdata")
AUDIO_DIR = GUITARSET_DIR / "audio_mono-mic"
ANNOT_DIR = GUITARSET_DIR / "annotation"


def pitch_shift_audio(audio, sr, n_steps):
    """音声のピッチシフト"""
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def pitch_shift_jams(jams_path, n_steps, output_path):
    """JAMSアノテーションのピッチ値をシフト
    
    GuitarSetのJAMS構造:
      - note_midi: data=list, value=float (MIDI note number) → シフト
      - pitch_contour: data=dict (time/duration/value/confidence arrays) → 周波数シフト
      - chord, key_mode: value=string → スキップ (ピッチシフト非対応)
      - beat_position, tempo: → スキップ
    """
    with open(jams_path, "r") as f:
        jams = json.load(f)
    
    for annot in jams.get("annotations", []):
        ns = annot.get("namespace", "")
        data = annot.get("data", [])
        
        if ns == "note_midi" and isinstance(data, list):
            # note_midi: value is float MIDI note number
            for obs in data:
                val = obs.get("value")
                if isinstance(val, (int, float)):
                    obs["value"] = val + n_steps
        
        elif ns == "pitch_contour" and isinstance(data, dict):
            # pitch_contour: data is dict with arrays
            # value array contains Hz frequencies
            values = data.get("value", [])
            if isinstance(values, list):
                ratio = 2 ** (n_steps / 12)
                data["value"] = [v * ratio if isinstance(v, (int, float)) else v for v in values]
        
        # chord, key_mode, beat_position, tempo → skip (no pitch modification needed)
    
    with open(output_path, "w") as f:
        json.dump(jams, f)


def augment_all(shifts=None, max_files=None):
    """全GuitarSetファイルを拡張"""
    if shifts is None:
        shifts = [-2, -1, 1, 2]
    
    wav_files = sorted(AUDIO_DIR.glob("*.wav"))
    # Only process original files (exclude already pitch-shifted files)
    wav_files = [f for f in wav_files if '_p+' not in f.name and '_p-' not in f.name]
    if max_files:
        wav_files = wav_files[:max_files]
    
    total = len(wav_files) * len(shifts)
    print(f"GuitarSet augmentation: {len(wav_files)} files x {len(shifts)} shifts = {total} augmented files")
    print(f"Shifts: {shifts} semitones")
    print(f"Audio dir: {AUDIO_DIR}")
    print(f"Annot dir: {ANNOT_DIR}")
    print()
    
    created = 0
    skipped = 0
    errors = 0
    t0 = time.time()
    
    for i, wav_file in enumerate(wav_files):
        base = wav_file.stem  # e.g., "00_BN1-129-Eb_comp_mic"
        
        # Find corresponding JAMS (remove _mic suffix)
        jams_base = base.replace("_mic", "")
        jams_file = ANNOT_DIR / f"{jams_base}.jams"
        
        for shift in shifts:
            shift_str = f"p{shift:+d}"  # e.g., "p+1", "p-2"
            
            # Output paths - in same directories
            out_wav = AUDIO_DIR / f"{base}_{shift_str}.wav"
            out_jams = ANNOT_DIR / f"{jams_base}_{shift_str}.jams"
            
            if out_wav.exists() and out_jams.exists():
                skipped += 1
                continue
            
            try:
                # Pitch shift audio
                if not out_wav.exists():
                    audio, sr = librosa.load(str(wav_file), sr=22050, mono=True)
                    shifted = pitch_shift_audio(audio, sr, shift)
                    sf.write(str(out_wav), shifted, sr)
                
                # Shift JAMS annotations
                if not out_jams.exists() and jams_file.exists():
                    pitch_shift_jams(str(jams_file), shift, str(out_jams))
                
                created += 1
                
                if (created + skipped) % 50 == 0:
                    elapsed = time.time() - t0
                    rate = (created + skipped) / elapsed if elapsed > 0 else 0
                    remaining = (total - created - skipped) / rate if rate > 0 else 0
                    print(f"  [{created+skipped}/{total}] created={created}, skipped={skipped} "
                          f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")
                
            except Exception as e:
                errors += 1
                print(f"  Error {base} shift={shift}: {e}")
    
    elapsed = time.time() - t0
    print(f"\n{'='*50}")
    print(f"Results: created={created}, skipped={skipped}, errors={errors}")
    print(f"Time: {elapsed:.0f}s")
    
    # Count total files
    total_wavs = len(list(AUDIO_DIR.glob("*.wav")))
    total_jams = len(list(ANNOT_DIR.glob("*.jams")))
    print(f"Total WAVs: {total_wavs}, Total JAMS: {total_jams}")
    print(f"{'='*50}")
    
    return created


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment GuitarSet with pitch shifting")
    parser.add_argument("--max", type=int, default=None, help="Max files to process")
    parser.add_argument("--shifts", type=str, default="-2,-1,1,2", help="Comma-separated shifts")
    args = parser.parse_args()
    
    shifts = [int(s) for s in args.shifts.split(",")]
    augment_all(shifts=shifts, max_files=args.max)
