"""GP5 → notes.json + WAV 変換パイプライン (FluidSynth不要版)
PyGuitarProでGP5解析 → pretty_midi経由でMIDI作成 → FluidSynthかscipy.signal正弦波合成でWAV生成
"""
import guitarpro
import json
import numpy as np
import soundfile as sf
from pathlib import Path
from collections import defaultdict

def gp5_to_notes(gp5_path: str) -> dict:
    """GP5ファイルからノート情報を抽出"""
    gp = guitarpro.parse(gp5_path)
    tempo = gp.tempo  # BPM
    
    notes = []
    for track in gp.tracks:
        tuning = [s.value for s in track.strings]  # MIDI values per string
        
        current_time = 0.0  # seconds
        for measure in track.measures:
            # Time signature from header
            ts_num = measure.header.timeSignature.numerator
            ts_den = measure.header.timeSignature.denominator.value
            
            # Tempo: use header's tempo if available, else global
            header_tempo = getattr(measure.header, 'tempo', None)
            if header_tempo and hasattr(header_tempo, 'value') and header_tempo.value > 0:
                tempo = header_tempo.value
            
            beat_duration_s = 60.0 / tempo  # seconds per quarter note
            
            for voice in measure.voices:
                voice_time = current_time
                for beat in voice.beats:
                    # Beat duration in quarter notes
                    dur_value = beat.duration.value  # 1=whole, 2=half, 4=quarter, 8=eighth...
                    dur_quarter = 4.0 / dur_value
                    if beat.duration.isDotted:
                        dur_quarter *= 1.5
                    dur_s = dur_quarter * beat_duration_s
                    
                    for note in beat.notes:
                        fret = note.value
                        string = note.string  # 1-indexed from top
                        
                        # Calculate MIDI pitch from string tuning + fret
                        if string <= len(tuning):
                            midi_pitch = tuning[string - 1] + fret
                        else:
                            midi_pitch = 60 + fret  # fallback
                        
                        # Techniques
                        techs = []
                        eff = note.effect
                        if eff.palmMute: techs.append('palm_mute')
                        if eff.hammer: techs.append('hammer')
                        if eff.harmonic: techs.append('harmonic')
                        if eff.bend: techs.append('bend')
                        if eff.slides: techs.append('slide')
                        if eff.vibrato: techs.append('vibrato')
                        if eff.staccato: techs.append('staccato')
                        if eff.letRing: techs.append('let_ring')
                        
                        notes.append({
                            'start': round(voice_time, 4),
                            'duration': round(dur_s, 4),
                            'pitch': midi_pitch,
                            'string': string,
                            'fret': fret,
                            'technique': techs[0] if techs else 'normal',
                            'techniques': techs
                        })
                    
                    voice_time += dur_s
            
            # Advance to next measure
            bar_duration = ts_num * beat_duration_s * (4.0 / ts_den)
            current_time += bar_duration
    
    return {
        'tempo': tempo,
        'notes': notes,
        'tuning': tuning if gp.tracks else [],
    }


def notes_to_wav(notes_data: dict, wav_path: str, sr=22050):
    """ノートデータから簡易ギター音声を合成"""
    notes = notes_data['notes']
    if not notes:
        return
    
    # Total duration
    max_end = max(n['start'] + n['duration'] for n in notes)
    total_samples = int((max_end + 1.0) * sr)
    audio = np.zeros(total_samples, dtype=np.float32)
    
    for note in notes:
        freq = 440.0 * (2.0 ** ((note['pitch'] - 69) / 12.0))
        start_sample = int(note['start'] * sr)
        dur_samples = int(note['duration'] * sr)
        end_sample = min(start_sample + dur_samples, total_samples)
        
        if start_sample >= total_samples or dur_samples <= 0:
            continue
        
        t = np.arange(end_sample - start_sample) / sr
        
        # Guitar-like envelope: fast attack, exponential decay
        envelope = np.exp(-3.0 * t / max(note['duration'], 0.01))
        
        # Fundamental + harmonics (guitar-like timbre)
        wave = (
            0.6 * np.sin(2 * np.pi * freq * t) +
            0.25 * np.sin(2 * np.pi * 2 * freq * t) +
            0.1 * np.sin(2 * np.pi * 3 * freq * t) +
            0.05 * np.sin(2 * np.pi * 4 * freq * t)
        )
        
        # Palm mute: add more damping
        if note.get('technique') == 'palm_mute':
            envelope *= np.exp(-8.0 * t / max(note['duration'], 0.01))
            wave += 0.15 * np.sin(2 * np.pi * 5 * freq * t)
        
        # Harmonic: pure sine
        if note.get('technique') == 'harmonic':
            wave = np.sin(2 * np.pi * freq * 2 * t)
        
        signal = wave * envelope * 0.3
        audio[start_sample:end_sample] += signal[:end_sample - start_sample]
    
    # Normalize
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.8
    
    sf.write(wav_path, audio, sr)


def process_gp5_batch(gp5_dir: str, output_dir: str, max_files=None):
    """一括変換"""
    gp5_dir = Path(gp5_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    gp5_files = sorted(gp5_dir.rglob('*.gp5'))
    if max_files:
        gp5_files = gp5_files[:max_files]
    
    print(f'Processing {len(gp5_files)} GP5 files...')
    
    success = 0
    errors = 0
    total_notes = 0
    
    for i, gp5_path in enumerate(gp5_files):
        try:
            name = gp5_path.stem
            out_subdir = output_dir / name
            out_subdir.mkdir(exist_ok=True)
            
            # GP5 → notes
            data = gp5_to_notes(str(gp5_path))
            notes_path = out_subdir / 'notes.json'
            with open(notes_path, 'w') as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
            
            # Notes → WAV
            wav_path = out_subdir / 'audio.wav'
            notes_to_wav(data, str(wav_path))
            
            total_notes += len(data['notes'])
            success += 1
            
            if (i+1) % 100 == 0 or i == 0:
                print(f'  [{i+1}/{len(gp5_files)}] {name}: {len(data["notes"])} notes')
        
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f'  ERROR: {gp5_path.name}: {e}')
    
    print(f'\nDone! Success: {success}, Errors: {errors}')
    print(f'Total notes: {total_notes:,}')
    return success


if __name__ == '__main__':
    import sys
    max_files = int(sys.argv[1]) if len(sys.argv) > 1 else 10  # default: test 10 files
    
    gp5_dir = r'D:\Music\nextchord-solotab\datasets\score-set'
    output_dir = r'D:\Music\nextchord-solotab\datasets\score-set-processed'
    
    process_gp5_batch(gp5_dir, output_dir, max_files=max_files)
    
    # Show sample
    sample_dir = Path(output_dir)
    for d in sorted(sample_dir.iterdir())[:1]:
        notes_path = d / 'notes.json'
        if notes_path.exists():
            with open(notes_path) as f:
                data = json.load(f)
            print(f'\nSample {d.name}:')
            print(f'  Tempo: {data["tempo"]}, Notes: {len(data["notes"])}')
            for n in data['notes'][:5]:
                print(f'  t={n["start"]:.2f}s S{n["string"]} F{n["fret"]} P{n["pitch"]} {n["technique"]}')
