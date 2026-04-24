"""
precache_synthtab.py — SynthTab_Full のキャッシュ事前生成 (CPUのみ)
====================================================================
Stage 1 学習のボトルネックであるオンライン合成+HCQT計算を
事前に全トラック分実行してキャッシュに保存する。

並列で実行可能: 学習プロセスとは独立して動作。

Usage:
    python precache_synthtab.py
    python precache_synthtab.py --max-items 5000 --workers 4
"""

import os
import sys
import time
import argparse
import types
import glob
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

if 'muda' not in sys.modules:
    _muda = types.ModuleType('muda')
    _muda.deform = lambda *a, **k: None
    sys.modules['muda'] = _muda

import numpy as np
import mido

STANDARD_TUNING = [40, 45, 50, 55, 59, 64]
TARGET_SR = 22050
HOP_LENGTH = 512


def parse_string_midis(track_dir, num_frets=19):
    """弦別MIDIからノートイベントを抽出"""
    tempo_path = os.path.join(track_dir, 'tempo.txt')
    bpm = 120.0
    if os.path.exists(tempo_path):
        try:
            bpm = float(open(tempo_path).read().strip())
        except Exception:
            pass
    
    microseconds_per_beat = int(60_000_000 / bpm)
    events = []
    
    for string_idx in range(6):
        midi_path = os.path.join(track_dir, f'string_{string_idx + 1}.mid')
        if not os.path.exists(midi_path):
            continue
        try:
            mid = mido.MidiFile(midi_path)
            tpb = mid.ticks_per_beat
            for track in mid.tracks:
                tick = 0
                active = {}
                for msg in track:
                    tick += msg.time
                    if msg.type == 'note_on' and msg.velocity > 0:
                        active[msg.note] = (tick, msg.velocity)
                    elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                        if msg.note in active:
                            st, vel = active.pop(msg.note)
                            start_sec = mido.tick2second(st, tpb, microseconds_per_beat)
                            end_sec = mido.tick2second(tick, tpb, microseconds_per_beat)
                            dur = end_sec - start_sec
                            if dur > 0.01:
                                fret = msg.note - STANDARD_TUNING[string_idx]
                                if 0 <= fret <= num_frets:
                                    events.append({
                                        'string': string_idx, 'fret': fret,
                                        'pitch': msg.note, 'start': start_sec,
                                        'duration': dur,
                                    })
        except Exception:
            pass
    return events


def synthesize_and_cache(args_tuple):
    """1トラックを合成+HCQT計算してキャッシュに保存"""
    track_dir, cache_path, idx = args_tuple
    
    if os.path.exists(cache_path):
        return 'cached'
    
    try:
        import librosa
        from amt_tools.features import HCQT
        
        data_proc = HCQT(sample_rate=TARGET_SR, hop_length=HOP_LENGTH,
                         fmin=librosa.note_to_hz('E2'),
                         harmonics=[0.5, 1, 2, 3, 4, 5],
                         n_bins=144, bins_per_octave=36)
        
        events = parse_string_midis(track_dir)
        if not events or len(events) < 3:
            return 'skip'
        
        # 簡易合成（サイン波 + 倍音）
        max_time = max(e['start'] + e['duration'] for e in events)
        total_samples = min(int((max_time + 1.0) * TARGET_SR), TARGET_SR * 60)
        audio = np.zeros(total_samples)
        
        for event in events:
            freq = 440.0 * (2.0 ** ((event['pitch'] - 69) / 12.0))
            dur = min(event['duration'], 3.0)
            n_samples = int(dur * TARGET_SR)
            if n_samples <= 0:
                continue
            t = np.arange(n_samples) / TARGET_SR
            envelope = np.exp(-3.0 * t / max(dur, 0.01))
            note_audio = (
                0.5 * np.sin(2 * np.pi * freq * t) +
                0.25 * np.sin(2 * np.pi * 2 * freq * t) +
                0.125 * np.sin(2 * np.pi * 3 * freq * t)
            ) * envelope * 0.5
            
            start_sample = int(event['start'] * TARGET_SR)
            end_sample = min(start_sample + n_samples, total_samples)
            if start_sample < total_samples:
                audio[start_sample:end_sample] += note_audio[:end_sample - start_sample]
        
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.8
        
        # HCQT特徴量
        features = data_proc.process_audio(audio)
        C, F_dim, T = features.shape
        if T < 10:
            return 'short'
        
        features_t = np.transpose(features, (2, 0, 1))[:, :, :, np.newaxis]
        
        # Tablature
        num_frames = T
        tablature = np.full((6, num_frames), -1, dtype=np.int64)
        for event in events:
            onset_f = max(0, min(int(event['start'] * TARGET_SR / HOP_LENGTH), num_frames - 1))
            offset_f = max(onset_f + 1, min(int((event['start'] + event['duration']) * TARGET_SR / HOP_LENGTH), num_frames))
            tablature[event['string'], onset_f:offset_f] = event['fret']
        
        # 200フレームにクロップ
        num_target = 200
        if T > num_target:
            start = T // 4  # 先頭25%からスタート
            features_t = features_t[start:start + num_target]
            tablature = tablature[:, start:start + num_target]
        elif T < num_target:
            pad = num_target - T
            features_t = np.pad(features_t, ((0, pad), (0, 0), (0, 0), (0, 0)))
            tablature = np.pad(tablature, ((0, 0), (0, pad)), constant_values=-1)
        
        np.savez_compressed(cache_path, features=features_t, tablature=tablature)
        return 'ok'
    except Exception as e:
        return f'error: {e}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-items', type=int, default=5000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--cache-dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 
                                            'generated', 'fretnet_3stage', 'cache', 'synthtab'))
    parser.add_argument('--synthtab-dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 
                                            'datasets', 'SynthTab_Full'))
    args = parser.parse_args()
    
    os.makedirs(args.cache_dir, exist_ok=True)
    
    outall = os.path.join(args.synthtab_dir, 'jams_midi', 'outall')
    all_dirs = [
        os.path.join(outall, d) for d in os.listdir(outall)
        if os.path.isdir(os.path.join(outall, d)) and 'Acoustic' in d
    ]
    
    np.random.seed(42)
    np.random.shuffle(all_dirs)
    all_dirs = all_dirs[:args.max_items]
    
    print(f"Pre-caching {len(all_dirs)} SynthTab tracks with {args.workers} workers...")
    print(f"Cache dir: {args.cache_dir}")
    
    # 既にキャッシュ済みをスキップ
    tasks = []
    for idx, td in enumerate(all_dirs):
        cache_name = os.path.basename(td).replace(' ', '_')[:80]
        cache_path = os.path.join(args.cache_dir, f'{cache_name}_{idx}.npz')
        tasks.append((td, cache_path, idx))
    
    existing = sum(1 for _, cp, _ in tasks if os.path.exists(cp))
    print(f"Already cached: {existing}/{len(tasks)}")
    
    ok = existing
    errors = 0
    skipped = 0
    t0 = time.time()
    
    # シーケンシャル実行（ProcessPoolは import の問題を避けるため）
    for i, task in enumerate(tasks):
        if os.path.exists(task[1]):
            continue
        
        result = synthesize_and_cache(task)
        if result == 'ok':
            ok += 1
        elif result.startswith('error'):
            errors += 1
        else:
            skipped += 1
        
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1 - existing) / elapsed * 60 if elapsed > 0 else 0
            remaining = len(tasks) - i - 1
            eta = remaining / max(rate, 0.01)
            print(f"  [{i+1}/{len(tasks)}] ok={ok} err={errors} skip={skipped} "
                  f"rate={rate:.0f}/min ETA={eta:.0f}min")
    
    elapsed = time.time() - t0
    print(f"\nDone! ok={ok} errors={errors} skipped={skipped} in {elapsed/60:.1f}min")


if __name__ == '__main__':
    main()
