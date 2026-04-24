"""GP5 → MIDI → FluidSynth WAV 一括変換パイプライン（容量最適化版）
- 1,000ファイルのサブセット
- モノラル 22050Hz 16bit WAV
- ディスク使用量: 推定5-10GB
"""
import guitarpro
import mido
import subprocess
import json
import os
import soundfile as sf
import numpy as np
from pathlib import Path
from collections import Counter

FLUIDSYNTH = r'D:\Music\nextchord-solotab\tools\fluidsynth-v2.5.2-win10-x64-cpp11\bin\fluidsynth.exe'
SF2_PATH = r'D:\Music\nextchord-solotab\tools\FluidR3_GM.sf2'
GP5_DIR = Path(r'D:\Music\nextchord-solotab\datasets\score-set')
OUTPUT_DIR = Path(r'D:\Music\nextchord-solotab\datasets\score-set-processed')
TEMP_DIR = Path(r'C:\Users\kotan\AppData\Local\Temp\gp5_midi')


def gp5_to_midi(gp5_path, midi_path):
    """GP5 → MIDI変換"""
    gp = guitarpro.parse(str(gp5_path))
    tempo = gp.tempo
    
    mid = mido.MidiFile()
    mid.ticks_per_beat = 480
    track = mido.MidiTrack()
    mid.tracks.append(track)
    
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo), time=0))
    track.append(mido.Message('program_change', program=25, time=0))  # Steel Guitar
    
    gt = gp.tracks[0]
    tuning = [s.value for s in gt.strings]
    
    notes = []
    current_tick = 0
    
    for measure in gt.measures:
        ts_num = measure.header.timeSignature.numerator
        ts_den = measure.header.timeSignature.denominator.value
        
        header_tempo = getattr(measure.header, 'tempo', None)
        if header_tempo and hasattr(header_tempo, 'value') and header_tempo.value > 0:
            tempo = header_tempo.value
        
        for voice in measure.voices:
            vt = current_tick
            for beat in voice.beats:
                dur_value = beat.duration.value
                dur_ticks = int(480 * 4 / dur_value)
                if beat.duration.isDotted:
                    dur_ticks = int(dur_ticks * 1.5)
                
                for note in beat.notes:
                    fret = note.value
                    string = note.string
                    midi_pitch = tuning[string - 1] + fret if string <= len(tuning) else 60 + fret
                    midi_pitch = max(0, min(127, midi_pitch))
                    notes.append({
                        'pitch': midi_pitch,
                        'start_tick': vt,
                        'dur_ticks': dur_ticks,
                        'velocity': 80,
                    })
                vt += dur_ticks
        
        bar_ticks = int(480 * 4 * ts_num / ts_den)
        current_tick += bar_ticks
    
    # Sort and write MIDI events
    events = []
    for n in notes:
        events.append(('on', n['start_tick'], n['pitch'], n['velocity']))
        events.append(('off', n['start_tick'] + n['dur_ticks'], n['pitch'], 0))
    events.sort(key=lambda e: (e[1], 0 if e[0] == 'off' else 1))
    
    last_tick = 0
    for ev in events:
        delta = ev[1] - last_tick
        if ev[0] == 'on':
            track.append(mido.Message('note_on', note=ev[2], velocity=ev[3], time=delta))
        else:
            track.append(mido.Message('note_off', note=ev[2], velocity=0, time=delta))
        last_tick = ev[1]
    
    mid.save(str(midi_path))
    return len(notes)


def midi_to_wav(midi_path, wav_path):
    """FluidSynth MIDI → WAV (ステレオ)"""
    tmp_wav = str(wav_path) + '.tmp.wav'
    cmd = [
        FLUIDSYNTH, '-ni',
        '-F', tmp_wav, '-T', 'wav',
        '-r', '22050', '-g', '1.0', '-R', '0', '-C', '0',
        SF2_PATH, str(midi_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f'FluidSynth error: {result.stderr[:200]}')
    
    # ステレオ → モノラル 16bit変換
    audio, sr = sf.read(tmp_wav)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)  # stereo → mono
    audio = (audio * 32767).astype(np.int16)
    sf.write(str(wav_path), audio, sr, subtype='PCM_16')
    os.remove(tmp_wav)
    
    return sf.info(str(wav_path)).duration


def process_batch(max_files=1000):
    """一括変換"""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    gp5_files = sorted(GP5_DIR.rglob('*.gp5'))
    # ノート数が50以上のファイルを優先的に選択
    selected = []
    for f in gp5_files:
        if len(selected) >= max_files:
            break
        # notes.jsonが既に存在するファイルを優先
        name = f.stem
        notes_path = OUTPUT_DIR / name / 'notes.json'
        if notes_path.exists():
            with open(notes_path) as jf:
                data = json.load(jf)
            if len(data['notes']) > 30:
                selected.append(f)
    
    # 足りない場合はランダムに追加
    if len(selected) < max_files:
        remaining = [f for f in gp5_files if f not in set(selected)]
        for f in remaining[:max_files - len(selected)]:
            selected.append(f)
    
    print(f'Processing {len(selected)} GP5 files...')
    
    success = 0
    errors = 0
    total_size_mb = 0
    
    for i, gp5_path in enumerate(selected):
        name = gp5_path.stem
        out_dir = OUTPUT_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        wav_out = out_dir / 'audio.wav'
        
        if wav_out.exists():
            # Already done
            total_size_mb += wav_out.stat().st_size / (1024*1024)
            success += 1
            continue
        
        try:
            midi_path = TEMP_DIR / f'{name}.mid'
            n_notes = gp5_to_midi(str(gp5_path), str(midi_path))
            
            if n_notes == 0:
                errors += 1
                continue
            
            duration = midi_to_wav(str(midi_path), str(wav_out))
            
            file_size = wav_out.stat().st_size / (1024*1024)
            total_size_mb += file_size
            success += 1
            
            # Cleanup MIDI
            midi_path.unlink(missing_ok=True)
            
            if (i+1) % 50 == 0 or i == 0:
                print(f'  [{i+1}/{len(selected)}] {name}: {n_notes} notes, {duration:.0f}s, {file_size:.1f}MB (total: {total_size_mb/1024:.1f}GB)')
        
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f'  ERROR [{i+1}]: {name}: {e}')
    
    print(f'\nDone! Success: {success}, Errors: {errors}')
    print(f'Total WAV size: {total_size_mb/1024:.1f} GB')
    return success


if __name__ == '__main__':
    import sys
    max_files = int(sys.argv[1]) if len(sys.argv) > 1 else 5  # default: test 5 files
    process_batch(max_files=max_files)
