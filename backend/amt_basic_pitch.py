"""
amt_basic_pitch.py — Spotify Basic-Pitch による軽量・高速・完全ローカルAMTモジュール
========================================================================================
- ライセンス: Apache 2.0 (Spotify OSS, 完全無料・商用利用可)
- 処理フロー:
    1. 実録音音声 (WAV/MP3) → Basic-Pitch ONNX/Local Engine → MIDI/ノートイベント抽出
    2. MIDIバイパス (ユーザー持ち込み .mid/.midi ファイル直接パース)
- コスト: ローカルCPU/GPU完結・追加クラウド課金ゼロ ($0.00)
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import numpy as np

# 警告抑制
logging.getLogger("basic_pitch").setLevel(logging.ERROR)
logging.getLogger("root").setLevel(logging.ERROR)

try:
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
    BASIC_PITCH_AVAILABLE = True
except Exception as e:
    BASIC_PITCH_AVAILABLE = False
    _import_err = str(e)


def transcribe_audio_to_notes(
    audio_path: Union[str, Path],
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length: float = 58.0,  # ms
    min_freq: Optional[float] = None,    # ギター低音域 (E2 = 82.4Hz, Drop D = 73.4Hz, Drop C = 65.4Hz)
    max_freq: Optional[float] = None,    # ギター高音域 (E6 = 1318.5Hz)
) -> List[Dict[str, Any]]:
    """
    Spotify Basic-Pitch を用いて音声からノートイベントリストを抽出。
    
    Returns:
        List of dicts: [
            {
                "start": float (seconds),
                "end": float (seconds),
                "pitch": int (MIDI pitch 0-127),
                "velocity": float (0.0-1.0),
                "confidence": float (0.0-1.0),
            }, ...
        ]
    """
    if not BASIC_PITCH_AVAILABLE:
        raise RuntimeError(f"Basic-Pitch is not available: {_import_err}")

    audio_path = str(audio_path)
    t0 = time.time()
    
    # ギター向けの周波数範囲デフォルト設定 (Dropチューニング対応: 60Hz〜1500Hz)
    if min_freq is None:
        min_freq = 60.0
    if max_freq is None:
        max_freq = 1500.0

    # 推論実行
    model_output, midi_data, note_events = predict(
        audio_path,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=minimum_note_length,
        minimum_frequency=min_freq,
        maximum_frequency=max_freq,
    )

    notes = []
    # note_events: List of (start_time_s, end_time_s, pitch_midi, amplitude, list_of_pitch_bends)
    for ev in note_events:
        start_time, end_time, pitch, amplitude = ev[0], ev[1], int(round(ev[2])), float(ev[3])
        if end_time <= start_time:
            continue
        notes.append({
            "start": float(start_time),
            "end": float(end_time),
            "pitch": int(pitch),
            "velocity": float(min(1.0, max(0.1, amplitude))),
            "confidence": float(min(1.0, max(0.1, amplitude))),
            "technique": "normal",
        })

    # 時間順にソート
    notes.sort(key=lambda n: (n["start"], n["pitch"]))
    elapsed = time.time() - t0
    print(f"[Basic-Pitch AMT] Transcribed {len(notes)} notes in {elapsed:.2f}s (Cost: $0.00, Engine: Local Apache-2.0)")
    
    return notes


def parse_midi_to_notes(midi_file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    ユーザーが提供した外部MIDIファイル (.mid / .midi) を直接パースしてノート配列を生成 (MIDIバイパス)。
    """
    import mido  # type: ignore

    midi_path = str(midi_file_path)
    mid = mido.MidiFile(midi_path)
    
    notes = []
    # 各トラックのアクティブノートトラッキング: {(channel, pitch): (start_time, velocity)}
    current_time = 0.0
    active_notes = {}

    for msg in mid:
        current_time += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            active_notes[(msg.channel, msg.note)] = (current_time, msg.velocity / 127.0)
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            key = (msg.channel, msg.note)
            if key in active_notes:
                start_t, vel = active_notes.pop(key)
                dur = max(0.05, current_time - start_t)
                notes.append({
                    "start": float(start_t),
                    "end": float(start_t + dur),
                    "pitch": int(msg.note),
                    "velocity": float(vel),
                    "confidence": 1.0,
                    "technique": "normal",
                })

    # 閉じられていないノートの回収
    for (channel, pitch), (start_t, vel) in active_notes.items():
        dur = max(0.1, current_time - start_t)
        notes.append({
            "start": float(start_t),
            "end": float(start_t + dur),
            "pitch": int(pitch),
            "velocity": float(vel),
            "confidence": 1.0,
            "technique": "normal",
        })

    notes.sort(key=lambda n: (n["start"], n["pitch"]))
    print(f"[MIDI Bypass] Parsed {len(notes)} notes directly from MIDI file: {Path(midi_path).name}")
    
    return notes
