"""
synth_from_midi.py — V2 60K MIDI→FluidSynth 高速大規模WAV合成
================================================================
V2 60K datasetの弦別MIDIファイル(string_1.mid～string_6.mid)を
直接マージしてFluidSynthで高品質WAVを合成する。
JAMSパースを省略するため高速。

V2 60K構造:
  track_dir/
    1 - Acoustic Nylon Guitar.jams  ← アノテーション
    string_1.mid ～ string_5.mid    ← 弦別MIDI
    tempo.txt                       ← テンポ情報

Usage:
    python synth_from_midi.py --max-tracks 45120
    python synth_from_midi.py --guitar-only    # ギタートラックのみ
"""

import os
import sys
import json
import time as time_mod
import argparse
import struct
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

from solotab_utils import STANDARD_TUNING
PPQ = 480

GUITAR_PROGRAMS = {
    "nylon": 24, "steel": 25, "jazz_electric": 26,
    "clean": 27, "distortion": 29, "overdriven": 29,
}

# GM Program→SoundFont Program mapping for diverse instruments
INSTRUMENT_PROGRAM = {
    "Acoustic Nylon Guitar": 24,
    "Acoustic Steel Guitar": 25,
    "Electric Jazz Guitar": 26,
    "Electric Clean Guitar": 27,
    "Overdriven Guitar": 29,
    "Distortion Guitar": 30,
    "Guitar Harmonics": 31,
    "Fingered Electric Bass": 33,
    "Plucked Electric Bass": 34,
    "Acoustic Bass": 32,
    "Fretless Bass": 35,
}


def detect_instrument(track_name: str) -> int:
    """トラック名からGM Program番号を推定する。"""
    for key, program in INSTRUMENT_PROGRAM.items():
        if key in track_name:
            return program
    # Default to nylon guitar
    return 24


def read_tempo(track_dir: str) -> float:
    """tempo.txtからBPMを読む。なければJAMSから。"""
    tempo_file = os.path.join(track_dir, "tempo.txt")
    if os.path.exists(tempo_file):
        try:
            with open(tempo_file, "r") as f:
                return float(f.read().strip())
        except:
            pass
    
    # Fallback: JAMSから
    for f in os.listdir(track_dir):
        if f.endswith(".jams"):
            try:
                with open(os.path.join(track_dir, f), "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                for ann in data.get("annotations", []):
                    if ann.get("namespace") == "tempo":
                        obs = ann.get("data", [])
                        if obs:
                            return float(obs[0].get("value", 120))
            except:
                pass
    return 120.0


def parse_jams_for_annotations(track_dir: str, bpm: float) -> List[Dict]:
    """JAMSからアノテーション(ノートリスト)を抽出する。"""
    tick_to_sec = 60.0 / (bpm * PPQ)
    notes = []
    
    for f in os.listdir(track_dir):
        if not f.endswith(".jams"):
            continue
        fp = os.path.join(track_dir, f)
        try:
            with open(fp, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except:
            continue
        
        for ann in data.get("annotations", []):
            if ann.get("namespace") != "note_tab":
                continue
            
            sb = ann.get("sandbox", {})
            string_idx_raw = sb.get("string_index", -1)
            open_tuning = sb.get("open_tuning", None)
            
            string_idx = string_idx_raw - 1 if string_idx_raw >= 1 else -1
            if string_idx < 0 or string_idx >= 6:
                continue
            
            open_pitch = int(open_tuning) if open_tuning is not None else STANDARD_TUNING[string_idx]
            
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
                
                start = tick_time * tick_to_sec
                duration = max(tick_dur * tick_to_sec, 0.05)
                pitch = open_pitch + fret
                
                if pitch < 20 or pitch > 100:
                    continue
                
                notes.append({
                    "pitch": pitch, "string": string_idx, "fret": fret,
                    "start": round(start, 4),
                    "end": round(start + duration, 4),
                    "velocity": velocity,
                })
    
    notes.sort(key=lambda n: (n["start"], n["pitch"]))
    return notes


def merge_string_midis(track_dir: str, output_midi: str,
                       program: int = 24, bpm: float = 120) -> bool:
    """弦別MIDIをマージして1つのMIDIにする。"""
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    guitar = pretty_midi.Instrument(program=program, is_drum=False, name="Guitar")
    
    note_count = 0
    for i in range(1, 7):
        string_midi = os.path.join(track_dir, f"string_{i}.mid")
        if not os.path.exists(string_midi):
            continue
        try:
            sm = pretty_midi.PrettyMIDI(string_midi)
            for inst in sm.instruments:
                for note in inst.notes:
                    # Remap to guitar program
                    guitar.notes.append(pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=note.start,
                        end=note.end,
                    ))
                    note_count += 1
        except:
            continue
    
    if note_count < 3:
        return False
    
    midi.instruments.append(guitar)
    midi.write(output_midi)
    return True


def synth_wav(midi_path: str, wav_path: str,
              soundfont_path: str, sample_rate: int = 22050) -> bool:
    """FluidSynthでMIDI→WAV。"""
    try:
        from midi2audio import FluidSynth
        fs = FluidSynth(soundfont_path, sample_rate=sample_rate)
        fs.midi_to_audio(midi_path, wav_path)
        return True
    except Exception:
        try:
            midi = pretty_midi.PrettyMIDI(midi_path)
            audio = midi.fluidsynth(fs=sample_rate, sf2_path=soundfont_path)
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak * 0.95
            sf.write(wav_path, audio, sample_rate)
            return True
        except Exception:
            return False


def process_track(track_dir, track_name, output_dir, soundfont_path, sample_rate):
    """1トラック処理: MIDI merge → FluidSynth WAV + annotation.json"""
    
    safe_name = track_name.replace('/', '_').replace('\\', '_')[:100]
    track_out = os.path.join(output_dir, safe_name)
    
    wav_path = os.path.join(track_out, "audio.wav")
    midi_path = os.path.join(track_out, "merged.mid")
    ann_path = os.path.join(track_out, "annotation.json")
    
    # Skip if already synthesized
    if os.path.exists(wav_path) and os.path.getsize(wav_path) > 1000:
        return {"name": safe_name, "status": "skipped", "notes": 0, "duration": 0}
    
    # Detect instrument type
    program = detect_instrument(track_name)
    
    # Read tempo
    bpm = read_tempo(track_dir)
    
    os.makedirs(track_out, exist_ok=True)
    
    # Strategy 1: Merge string MIDIs (preferred, more reliable)
    has_midis = any(os.path.exists(os.path.join(track_dir, f"string_{i}.mid"))
                    for i in range(1, 7))
    
    if has_midis:
        if not merge_string_midis(track_dir, midi_path, program, bpm):
            return None
    else:
        # Strategy 2: Parse JAMS and create MIDI
        notes = parse_jams_for_annotations(track_dir, bpm)
        if len(notes) < 3:
            return None
        
        midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        guitar = pretty_midi.Instrument(program=program)
        for n in notes:
            guitar.notes.append(pretty_midi.Note(
                velocity=max(30, min(127, int(n["velocity"]))),
                pitch=n["pitch"],
                start=max(0, n["start"]),
                end=max(n["start"] + 0.05, n["end"]),
            ))
        midi.instruments.append(guitar)
        midi.write(midi_path)
    
    # Synthesize WAV
    if not synth_wav(midi_path, wav_path, soundfont_path, sample_rate):
        return None
    
    # Extract annotations from JAMS
    notes = parse_jams_for_annotations(track_dir, bpm)
    
    # Deduplicate
    seen = set()
    unique = []
    for n in notes:
        key = (n["pitch"], round(n["start"], 3))
        if key not in seen:
            seen.add(key)
            unique.append(n)
    
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(unique, f)
    
    duration = unique[-1]["end"] if unique else 0
    return {"name": safe_name, "status": "ok", "notes": len(unique),
            "duration": round(duration, 1), "bpm": bpm}


def main():
    parser = argparse.ArgumentParser(
        description="V2 60K MIDI → FluidSynth WAV Synthesis (Full Scale)")
    parser.add_argument('--input-dir', type=str,
        default=r"D:\Music\all_jams_midi_V2_60000_tracks\outall")
    parser.add_argument('--output-dir', type=str,
        default=r"D:\Music\nextchord-solotab\datasets\SynthTab_FluidSynth_V2")
    parser.add_argument('--soundfont', type=str,
        default=r"D:\Music\nextchord-solotab\tools\FluidR3_GM.sf2")
    parser.add_argument('--sample-rate', type=int, default=22050)
    parser.add_argument('--max-tracks', type=int, default=None)
    parser.add_argument('--guitar-only', action='store_true',
        help='Only guitar/bass tracks (exclude drums etc)')
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"ERROR: Input dir not found: {args.input_dir}")
        sys.exit(1)
    
    track_dirs = sorted([d for d in os.listdir(args.input_dir)
                         if os.path.isdir(os.path.join(args.input_dir, d))])
    
    if args.guitar_only:
        guitar_keywords = ['Guitar', 'Bass', 'Nylon', 'Steel', 'Acoustic']
        tracks = [d for d in track_dirs
                  if any(kw in d for kw in guitar_keywords)]
    else:
        tracks = track_dirs
    
    if args.max_tracks:
        tracks = tracks[:args.max_tracks]
    
    print("=" * 60)
    print("V2 60K → FluidSynth Full-Scale Synthesis")
    print("=" * 60)
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"SoundFont: {args.soundfont}")
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
        track_dir = os.path.join(args.input_dir, track_name)
        try:
            result = process_track(
                track_dir, track_name, args.output_dir,
                args.soundfont, args.sample_rate)
        except Exception:
            result = None
        
        if result is None:
            failed += 1
        elif result["status"] == "skipped":
            skipped += 1
        else:
            success += 1
            total_notes += result["notes"]
            total_duration += result.get("duration", 0)
        
        if (i + 1) % 500 == 0 or (i + 1) == len(tracks):
            elapsed = time_mod.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(tracks) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(tracks)}] "
                  f"ok={success} skip={skipped} fail={failed} "
                  f"notes={total_notes:,} dur={total_duration/3600:.1f}h "
                  f"({rate:.1f}/s ETA={eta/60:.0f}min)")
    
    elapsed = time_mod.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Complete! {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Success: {success}, Skipped: {skipped}, Failed: {failed}")
    print(f"Total notes: {total_notes:,}")
    print(f"Total duration: {total_duration/3600:.1f} hours")
    print(f"Output: {args.output_dir}")


if __name__ == '__main__':
    main()
