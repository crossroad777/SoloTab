"""
synth_from_jams.py — SynthTab JAMS→FluidSynth 高品質WAV合成
=============================================================
SynthTabのJAMSアノテーション(note_tab + tempo)をJSONとして直接パースし、
FluidSynthの高品質SoundFontでギター音声を合成する。

SynthTab JAMSの特殊事情:
- namespace "note_tab" はJAMSライブラリで未サポート → JSONパース
- time/durationはtick単位(PPQ=480前提) → BPMからsecに変換
- 弦情報はannotation.sandbox.string_index
- テンポはann[6] namespace="tempo"

Usage:
    python synth_from_jams.py --max-tracks 5000
"""

import os
import sys
import json
import time as time_mod
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pretty_midi
import soundfile as sf

# Auto-detect FluidSynth DLL path
_FLUIDSYNTH_BIN = os.path.join(
    os.path.dirname(__file__), '..', 'tools', 'fluidsynth', 'bin')
if os.path.exists(_FLUIDSYNTH_BIN) and _FLUIDSYNTH_BIN not in os.environ.get('PATH', ''):
    os.environ['PATH'] = _FLUIDSYNTH_BIN + os.pathsep + os.environ.get('PATH', '')
    print(f"[synth] FluidSynth DLL added to PATH: {_FLUIDSYNTH_BIN}")


# Standard tuning MIDI pitches
STANDARD_TUNING = [40, 45, 50, 55, 59, 64]

PPQ = 480  # SynthTab default ticks per quarter note

# GM Guitar programs
GUITAR_PROGRAMS = {
    "nylon": 24,
    "steel": 25,
    "jazz_electric": 26,
    "clean": 27,
}


def parse_synthtab_jams(jams_path: str) -> tuple:
    """
    SynthTab JAMS をJSONとして直接パースし、ノートリストとBPMを返す。
    
    Returns:
        (notes: list[dict], bpm: float)
    """
    with open(jams_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    annotations = data.get("annotations", [])
    if not annotations:
        return [], 120.0
    
    # 1. テンポを抽出 (最後のannotationがtempoの場合が多い)
    bpm = 120.0  # デフォルト
    for ann in annotations:
        if ann.get("namespace") == "tempo":
            obs = ann.get("data", [])
            if obs:
                bpm = float(obs[0].get("value", 120.0))
            break
    
    # tick→秒変換係数
    tick_to_sec = 60.0 / (bpm * PPQ)
    
    # 2. ノート情報を抽出
    notes = []
    for ann in annotations:
        if ann.get("namespace") != "note_tab":
            continue
        
        # 弦情報: annotation.sandbox.string_index (1-based in SynthTab!)
        sb = ann.get("sandbox", {})
        string_idx_raw = sb.get("string_index", -1)
        open_tuning = sb.get("open_tuning", None)
        
        # SynthTab uses 1-based string numbering (1=high E, 6=low E)
        # Convert to 0-based
        string_idx = string_idx_raw - 1 if string_idx_raw >= 1 else -1
        
        if string_idx < 0 or string_idx >= 6:
            continue
        
        # 開放弦ピッチ
        if open_tuning is not None:
            open_pitch = int(open_tuning)
        else:
            open_pitch = STANDARD_TUNING[string_idx]
        
        for obs in ann.get("data", []):
            val = obs.get("value", {})
            if isinstance(val, dict):
                fret = val.get("fret", 0)
                velocity = val.get("velocity", 80)
            else:
                fret = int(val) if val else 0
                velocity = 80
            
            tick_time = float(obs.get("time", 0))
            tick_dur = float(obs.get("duration", PPQ))
            
            # tick→秒変換
            start = tick_time * tick_to_sec
            duration = max(tick_dur * tick_to_sec, 0.05)  # 最小50ms
            end = start + duration
            
            pitch = open_pitch + fret
            
            if pitch < 28 or pitch > 96:
                continue
            
            notes.append({
                "pitch": pitch,
                "string": string_idx,
                "fret": fret,
                "start": round(start, 4),
                "end": round(end, 4),
                "velocity": velocity,
            })
    
    notes.sort(key=lambda n: (n["start"], n["pitch"]))
    return notes, bpm


def notes_to_midi(notes: List[Dict], output_path: str,
                  program: int = 24, tempo: float = 120) -> bool:
    """ノートリスト→MIDIファイル。"""
    if not notes:
        return False
    
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    guitar = pretty_midi.Instrument(program=program, is_drum=False, name="Guitar")
    
    for note in notes:
        pitch = note["pitch"]
        if pitch < 0 or pitch > 127:
            continue
        
        velocity = note.get("velocity", 80)
        if isinstance(velocity, float) and velocity <= 1.0:
            velocity = int(velocity * 127)
        velocity = max(30, min(127, int(velocity)))
        
        start = max(0, note["start"])
        end = note["end"]
        if end <= start:
            end = start + 0.1
        
        midi_note = pretty_midi.Note(
            velocity=velocity, pitch=pitch,
            start=start, end=end,
        )
        guitar.notes.append(midi_note)
    
    midi.instruments.append(guitar)
    midi.write(output_path)
    return True


def synth_midi_to_wav(midi_path: str, wav_path: str,
                      soundfont_path: str, sample_rate: int = 22050) -> bool:
    """FluidSynthでMIDI→WAV変換。"""
    try:
        from midi2audio import FluidSynth
        fs = FluidSynth(soundfont_path, sample_rate=sample_rate)
        fs.midi_to_audio(midi_path, wav_path)
        return True
    except Exception as e1:
        try:
            midi = pretty_midi.PrettyMIDI(midi_path)
            audio = midi.fluidsynth(fs=sample_rate, sf2_path=soundfont_path)
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak * 0.95
            sf.write(wav_path, audio, sample_rate)
            return True
        except Exception as e2:
            return False


def save_training_jams(notes: List[Dict], output_path: str):
    """
    SynthGuitarDataset互換のJAMSを保存。
    namespace='note_midi', data_source=弦番号。
    """
    import jams as jams_lib
    
    jam = jams_lib.JAMS()
    
    string_notes = {s: [] for s in range(6)}
    for n in notes:
        s = n.get("string", 0)
        if 0 <= s < 6:
            string_notes[s].append(n)
    
    for string_idx in range(6):
        ann = jams_lib.Annotation(namespace='note_midi')
        ann.annotation_metadata = jams_lib.AnnotationMetadata(
            data_source=str(string_idx))
        
        for n in string_notes[string_idx]:
            ann.append(
                time=n["start"],
                duration=n["end"] - n["start"],
                value=n["pitch"],
            )
        
        jam.annotations.append(ann)
    
    jam.save(output_path)


def process_track(jams_dir, track_name, output_dir, soundfont_path,
                  program, sample_rate):
    """1トラックの処理: JAMS→MIDI→WAV + JAMS保存。"""
    jams_subdir = os.path.join(jams_dir, track_name)
    
    # Find JAMS files
    jams_files = [f for f in os.listdir(jams_subdir) if f.endswith('.jams')]
    if not jams_files:
        return None
    
    # Parse all JAMS in this track dir
    all_notes = []
    bpm = 120.0
    for jf in jams_files:
        jams_path = os.path.join(jams_subdir, jf)
        notes, track_bpm = parse_synthtab_jams(jams_path)
        all_notes.extend(notes)
        bpm = track_bpm
    
    if len(all_notes) < 5:
        return None
    
    # Deduplicate
    seen = set()
    unique = []
    for n in all_notes:
        key = (n["pitch"], round(n["start"], 3))
        if key not in seen:
            seen.add(key)
            unique.append(n)
    unique.sort(key=lambda n: (n["start"], n["pitch"]))
    
    # Output paths
    safe_name = track_name.replace('/', '_').replace('\\', '_')[:80]
    track_out = os.path.join(output_dir, safe_name)
    os.makedirs(track_out, exist_ok=True)
    
    wav_path = os.path.join(track_out, "audio.wav")
    midi_path = os.path.join(track_out, "guitar.mid")
    jams_out = os.path.join(track_out, "annotation.jams")
    
    # Skip if already done
    if os.path.exists(wav_path) and os.path.getsize(wav_path) > 1000:
        return {"name": safe_name, "notes": len(unique), "status": "skipped"}
    
    # MIDI
    if not notes_to_midi(unique, midi_path, program=program, tempo=bpm):
        return None
    
    # WAV
    if not synth_midi_to_wav(midi_path, wav_path, soundfont_path, sample_rate):
        return None
    
    # Training JAMS (SynthGuitarDataset compatible)
    try:
        save_training_jams(unique, jams_out)
    except Exception:
        # Fallback: save as JSON
        with open(jams_out.replace('.jams', '.json'), 'w') as f:
            json.dump(unique, f)
    
    return {"name": safe_name, "notes": len(unique), "status": "ok",
            "bpm": bpm, "duration": round(unique[-1]["end"], 1)}


def main():
    parser = argparse.ArgumentParser(
        description="SynthTab JAMS → FluidSynth WAV Synthesis")
    parser.add_argument('--jams-dir', type=str,
        default=r"D:\Music\nextchord-solotab\datasets\SynthTab_Full\jams_midi\outall")
    parser.add_argument('--output-dir', type=str,
        default=r"D:\Music\nextchord-solotab\datasets\SynthTab_FluidSynth")
    parser.add_argument('--soundfont', type=str,
        default=r"D:\Music\nextchord-solotab\tools\FluidR3_GM.sf2")
    parser.add_argument('--guitar-type', type=str, default='nylon',
        choices=['nylon', 'steel', 'jazz_electric', 'clean'])
    parser.add_argument('--sample-rate', type=int, default=22050)
    parser.add_argument('--max-tracks', type=int, default=None)
    parser.add_argument('--acoustic-only', action='store_true', default=True)
    args = parser.parse_args()
    
    program = GUITAR_PROGRAMS[args.guitar_type]
    
    if not os.path.exists(args.jams_dir):
        print(f"ERROR: JAMS dir not found: {args.jams_dir}")
        sys.exit(1)
    
    track_dirs = sorted([d for d in os.listdir(args.jams_dir)
                         if os.path.isdir(os.path.join(args.jams_dir, d))])
    
    if args.acoustic_only:
        tracks = [d for d in track_dirs
                  if 'Acoustic' in d or 'Nylon' in d]
    else:
        tracks = track_dirs
    
    if args.max_tracks:
        tracks = tracks[:args.max_tracks]
    
    print("=" * 60)
    print("SynthTab JAMS → FluidSynth Synthesis")
    print("=" * 60)
    print(f"Input: {args.jams_dir}")
    print(f"Output: {args.output_dir}")
    print(f"SoundFont: {args.soundfont}")
    print(f"Guitar: {args.guitar_type} (GM {program})")
    print(f"Tracks: {len(tracks)}")
    print()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    t0 = time_mod.time()
    success = 0
    skipped = 0
    failed = 0
    total_notes = 0
    total_duration = 0
    
    for i, track_name in enumerate(tracks):
        try:
            result = process_track(
                args.jams_dir, track_name, args.output_dir,
                args.soundfont, program, args.sample_rate
            )
        except Exception as e:
            result = None
        
        if result is None:
            failed += 1
        elif result["status"] == "skipped":
            skipped += 1
        else:
            success += 1
            total_notes += result["notes"]
            total_duration += result.get("duration", 0)
        
        if (i + 1) % 100 == 0 or (i + 1) == len(tracks):
            elapsed = time_mod.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(tracks) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(tracks)}] "
                  f"ok={success} skip={skipped} fail={failed} "
                  f"notes={total_notes} dur={total_duration/3600:.1f}h "
                  f"({rate:.1f}/s ETA={eta/60:.0f}min)")
    
    elapsed = time_mod.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Complete! {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Success: {success}, Skipped: {skipped}, Failed: {failed}")
    print(f"Total notes: {total_notes}")
    print(f"Total duration: {total_duration/3600:.1f} hours")
    print(f"Output: {args.output_dir}")


if __name__ == '__main__':
    main()
