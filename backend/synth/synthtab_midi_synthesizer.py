"""
synthtab_midi_synthesizer.py — SynthTabのMIDIからオーディオを合成
================================================================
SynthTabの弦別MIDI (string_1.mid~string_6.mid) と JAMS を読み込み、
学習用の (audio, tablature) ペアを生成する。

FluidSynth不要: pretty_midi.synthesize() で正弦波合成。
学習には音色の正確さより「正しいピッチ・タイミング」が重要。
"""

import os
import sys
import json
import glob
import numpy as np
import pretty_midi
import jams
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# ギター標準チューニング (MIDI note numbers: E2=40, A2=45, D3=50, G3=55, B3=59, E4=64)
from solotab_utils import STANDARD_TUNING
SR = 22050  # FretNet学習と同じサンプルレート


def load_string_midis(track_dir: str) -> Dict[int, pretty_midi.PrettyMIDI]:
    """弦別MIDIファイルを読み込む (string_1.mid ~ string_6.mid)"""
    midis = {}
    for s in range(1, 7):
        path = os.path.join(track_dir, f'string_{s}.mid')
        if os.path.exists(path):
            try:
                midis[s] = pretty_midi.PrettyMIDI(path)
            except Exception:
                pass
    return midis


def load_tempo(track_dir: str) -> float:
    """tempo.txt からBPMを読む"""
    path = os.path.join(track_dir, 'tempo.txt')
    if os.path.exists(path):
        try:
            with open(path) as f:
                return float(f.read().strip())
        except Exception:
            pass
    return 120.0


def load_jams_notes(track_dir: str) -> Optional[jams.JAMS]:
    """JAMSアノテーションを読み込む"""
    jams_files = glob.glob(os.path.join(track_dir, '*.jams'))
    if not jams_files:
        return None
    try:
        return jams.load(jams_files[0])
    except Exception:
        return None


def synthesize_track(track_dir: str, output_wav: str = None, sr: int = SR) -> Dict:
    """
    1トラックのMIDIを合成してWAVを生成し、TABアノテーションも返す。
    
    Returns:
        dict with keys: 'audio_path', 'notes', 'duration', 'bpm'
    """
    string_midis = load_string_midis(track_dir)
    if not string_midis:
        return None
    
    bpm = load_tempo(track_dir)
    
    # 全弦のMIDIをマージして合成
    combined = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    notes_list = []
    
    for string_num, midi in sorted(string_midis.items()):
        inst = pretty_midi.Instrument(program=25)  # Acoustic Guitar (steel)
        for orig_inst in midi.instruments:
            for note in orig_inst.notes:
                inst.notes.append(note)
                # TABアノテーション生成
                fret = note.pitch - STANDARD_TUNING[string_num - 1]
                if 0 <= fret <= 24:
                    notes_list.append({
                        'onset': float(note.start),
                        'offset': float(note.end),
                        'pitch': int(note.pitch),
                        'string': int(string_num),
                        'fret': int(fret),
                        'velocity': int(note.velocity),
                    })
        combined.instruments.append(inst)
    
    if not combined.instruments:
        return None
    
    # オーディオ合成 (正弦波)
    audio = combined.synthesize(fs=sr)
    
    # ピーク正規化
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9
    
    duration = len(audio) / sr
    
    # WAV保存
    if output_wav is None:
        output_wav = os.path.join(track_dir, 'synthesized.wav')
    os.makedirs(os.path.dirname(output_wav), exist_ok=True)
    sf.write(output_wav, audio, sr)
    
    # ノートをソート
    notes_list.sort(key=lambda n: (n['onset'], n['string']))
    
    return {
        'audio_path': output_wav,
        'notes': notes_list,
        'duration': duration,
        'bpm': bpm,
        'n_notes': len(notes_list),
    }


def scan_synthtab_tracks(base_dir: str) -> List[str]:
    """SynthTab JaMS+MIDIの全トラックディレクトリを列挙"""
    outall = os.path.join(base_dir, 'outall')
    if not os.path.exists(outall):
        outall = base_dir
    
    tracks = []
    for entry in sorted(os.listdir(outall)):
        track_dir = os.path.join(outall, entry)
        if os.path.isdir(track_dir):
            # string_1.mid が存在するかチェック
            if os.path.exists(os.path.join(track_dir, 'string_1.mid')):
                tracks.append(track_dir)
    return tracks


def batch_synthesize(base_dir: str, output_dir: str, max_tracks: int = None,
                     sr: int = SR) -> List[Dict]:
    """複数トラックを一括合成"""
    tracks = scan_synthtab_tracks(base_dir)
    if max_tracks:
        tracks = tracks[:max_tracks]
    
    print(f'Found {len(tracks)} tracks')
    results = []
    errors = 0
    
    for i, track_dir in enumerate(tracks):
        track_name = os.path.basename(track_dir)
        out_wav = os.path.join(output_dir, track_name, 'audio.wav')
        
        try:
            result = synthesize_track(track_dir, out_wav, sr)
            if result:
                results.append(result)
                if (i + 1) % 100 == 0:
                    print(f'  [{i+1}/{len(tracks)}] {result["n_notes"]} notes, '
                          f'{result["duration"]:.1f}s')
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f'  Error in {track_name}: {e}')
    
    print(f'\nDone: {len(results)} tracks synthesized, {errors} errors')
    return results


if __name__ == '__main__':
    # テスト: SynthTabの最初の5トラックを合成
    base_dir = r'd:\Music\all_jams_midi_V2_60000_tracks'
    output_dir = r'd:\Music\nextchord-solotab\datasets\SynthTab_Full\synthesized'
    
    print('=== SynthTab MIDI Synthesizer Test ===')
    tracks = scan_synthtab_tracks(base_dir)
    print(f'Total tracks found: {len(tracks)}')
    
    if tracks:
        # 最初の3トラックで動作テスト
        results = batch_synthesize(base_dir, output_dir, max_tracks=3)
        for r in results:
            print(f'\n  {os.path.basename(os.path.dirname(r["audio_path"]))}:')
            print(f'    Notes: {r["n_notes"]}, Duration: {r["duration"]:.1f}s, BPM: {r["bpm"]}')
            if r['notes']:
                n = r['notes'][0]
                print(f'    First note: t={n["onset"]:.2f}s str={n["string"]} fret={n["fret"]}')
