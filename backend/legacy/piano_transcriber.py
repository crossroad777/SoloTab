"""
piano_transcriber.py — ピアノ転写モデルによるギターノート検出
==============================================================
ByteDance の High-Resolution Piano Transcription (Kong et al.)
モデルを利用して、ギター音声からノートを検出する。

このモデルは MAESTRO データセット (200+時間) で学習されており、
ギター専用モデルの GuitarSet (30分) と比較して圧倒的なデータ量で
学習されているため、より多くのノートを検出できる。

ただしピアノモデルのため以下の制約がある:
  - 弦/フレット情報は出力しない → 後段の assign で補完
  - ギターの倍音構造に完全対応していない → フィルタで補正
"""

import numpy as np
import librosa
from typing import Dict, Optional, List
from pathlib import Path

SAMPLE_RATE = 16000  # piano_transcription_inference のデフォルト
GUITAR_MIDI_MIN = 40   # E2 (6弦開放)
GUITAR_MIDI_MAX = 88   # E6 (1弦24フレット相当)

_transcriptor = None


def is_model_available() -> bool:
    """ピアノ転写モデルが利用可能かチェック"""
    try:
        import piano_transcription_inference
        return True
    except ImportError:
        return False


def _load_transcriptor():
    """Transcriptoをロード (初回はモデルをダウンロード)"""
    global _transcriptor
    if _transcriptor is not None:
        return _transcriptor
    
    import torch
    from piano_transcription_inference import PianoTranscription
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _transcriptor = PianoTranscription(device=device)
    print(f"[piano_transcriber] Model loaded on {device}")
    return _transcriptor


def transcribe_guitar(
    wav_path: str,
    tuning_pitches: Optional[Dict[int, int]] = None,
    onset_threshold: float = 0.3,
    min_duration: float = 0.04,
) -> Dict:
    """
    ピアノ転写モデルを使ってギター音域のノートを検出する。
    
    Parameters
    ----------
    wav_path : str
        WAVファイルパス
    tuning_pitches : dict, optional
        弦番号→MIDI (string_assigner用)
    onset_threshold : float
        オンセット閾値 (低くするほどノートが増える)
    min_duration : float
        最小ノート長 (秒)
    
    Returns
    -------
    dict
        {"notes": [...], "method": "piano_da"}
    """
    from string_assigner import assign_strings_dp, STANDARD_TUNING, TUNINGS
    
    transcriptor = _load_transcriptor()
    
    # Load audio at 16kHz (piano_transcription_inference の要求)
    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    total_duration = len(audio) / SAMPLE_RATE
    print(f"[piano_transcriber] Audio: {total_duration:.1f}s")
    
    # 推論実行
    result = transcriptor.transcribe(audio, None)
    
    # est_note_events: list of dict with onset_time, offset_time, midi_note, velocity
    raw_notes = result.get('est_note_events', [])
    print(f"[piano_transcriber] Raw detection: {len(raw_notes)} notes")
    
    # ギター音域フィルタ + 最小長フィルタ
    guitar_notes = []
    for note in raw_notes:
        midi = int(note['midi_note'])
        onset = float(note['onset_time'])
        offset = float(note['offset_time'])
        velocity = float(note.get('velocity', 80)) / 127.0
        
        # ギター音域外を除外
        if midi < GUITAR_MIDI_MIN or midi > GUITAR_MIDI_MAX:
            continue
        
        # 最小ノート長
        if (offset - onset) < min_duration:
            continue
        
        guitar_notes.append({
            "start": round(onset, 4),
            "end": round(offset, 4),
            "pitch": midi,
            "string": 0,  # 後で assign_strings_dp で割り当て
            "fret": 0,
            "velocity": round(velocity, 2),
        })
    
    print(f"[piano_transcriber] Guitar-range filtered: {len(guitar_notes)} notes")
    
    # 弦/フレット割り当て (DP)
    if tuning_pitches:
        tuning = [tuning_pitches[i] for i in range(6)]
    else:
        tuning = STANDARD_TUNING
    
    if guitar_notes:
        guitar_notes = assign_strings_dp(guitar_notes, tuning=tuning)
        print(f"[piano_transcriber] After string assignment: {len(guitar_notes)} notes")
    
    return {
        "notes": guitar_notes,
        "total_count": len(guitar_notes),
        "method": "piano_da",
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python piano_transcriber.py <wav_path>")
        sys.exit(1)
    
    wav = sys.argv[1]
    result = transcribe_guitar(wav)
    print(f"\nTotal notes: {result['total_count']}")
    for n in result['notes'][:20]:
        print(f"  t={n['start']:.2f}-{n['end']:.2f} s{n['string']} f{n['fret']} MIDI={n['pitch']}")
