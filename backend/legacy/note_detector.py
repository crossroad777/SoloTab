"""
note_detector.py — Spotify Basic Pitch によるノート検出
======================================================
ポリフォニック対応の高精度ピッチ検出モデルで
アコギの個々の音符を抽出する。
"""

import numpy as np
from pathlib import Path


def detect_notes(wav_path: str, *, onset_threshold: float = 0.5,
                 frame_threshold: float = 0.3,
                 min_note_length_ms: float = 50,
                 min_freq: float = 75.0,
                 max_freq: float = 1500.0) -> dict:
    """
    Detect individual notes from an audio file using Basic Pitch.

    Parameters
    ----------
    wav_path : str
        Path to WAV file.
    onset_threshold : float
        Onset detection sensitivity (0-1). Lower = more onsets.
    frame_threshold : float
        Frame activation threshold (0-1). Lower = more notes.
    min_note_length_ms : float
        Minimum note length in milliseconds.
    min_freq : float
        Minimum frequency (Hz). Standard guitar low E = ~82 Hz.
    max_freq : float
        Maximum frequency (Hz). Guitar 24th fret high E ~ 1320 Hz.

    Returns
    -------
    dict with keys:
        notes : list[dict] - each with:
            start : float  — onset time (seconds)
            end : float    — offset time (seconds)
            pitch : int    — MIDI note number
            velocity : float — estimated velocity (0-1)
            pitch_bend : list[float] — pitch bend values per frame (optional)
        total_notes : int
    """
    from basic_pitch.inference import predict, Model
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from pathlib import Path as _Path

    # Pre-load model: prefer TFLite (avoids TF SavedModel compatibility issues)
    # ICASSP_2022_MODEL_PATH points to icassp_2022/nmp/ (TF SavedModel dir)
    # TFLite file is at icassp_2022/nmp.tflite (sibling)
    model_dir = _Path(ICASSP_2022_MODEL_PATH).parent
    tflite_path = model_dir / "nmp.tflite"
    if tflite_path.exists():
        model = Model(str(tflite_path))
        print(f"[note_detector] Loaded TFLite model: {tflite_path}")
    else:
        # Fallback to default (TF SavedModel)
        model = Model(str(ICASSP_2022_MODEL_PATH))
        print(f"[note_detector] Loaded model: {ICASSP_2022_MODEL_PATH}")

    model_output, midi_data, note_events = predict(
        wav_path,
        model_or_model_path=model,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=min_note_length_ms,
        minimum_frequency=min_freq,
        maximum_frequency=max_freq,
        multiple_pitch_bends=True,   # ポリフォニック対応pitch bend
    )

    # note_events: list of (start_time, end_time, pitch, velocity, pitch_bends)
    notes = []
    for event in note_events:
        start_time, end_time, pitch, velocity = event[:4]
        pitch_bends = event[4] if len(event) > 4 else None

        note = {
            "start": round(float(start_time), 4),
            "end": round(float(end_time), 4),
            "pitch": int(pitch),
            "velocity": round(float(velocity), 4),  # Basic Pitch outputs 0-1 range directly
        }
        if pitch_bends is not None and len(pitch_bends) > 0:
            note["pitch_bend"] = [round(float(pb), 3) for pb in pitch_bends]

        notes.append(note)

    # Sort by start time, then by pitch (low to high)
    notes.sort(key=lambda n: (n["start"], n["pitch"]))

    # Merge very short duplicate notes (same pitch within 30ms)
    notes = _merge_close_notes(notes, time_threshold=0.03)

    return {
        "notes": notes,
        "total_notes": len(notes),
    }


def _merge_close_notes(notes: list, time_threshold: float = 0.03) -> list:
    """Merge notes with same pitch that are very close together."""
    if len(notes) <= 1:
        return notes

    merged = [notes[0]]
    for note in notes[1:]:
        prev = merged[-1]
        if (note["pitch"] == prev["pitch"]
                and note["start"] - prev["end"] < time_threshold):
            # Extend previous note
            prev["end"] = max(prev["end"], note["end"])
            prev["velocity"] = max(prev["velocity"], note["velocity"])
        else:
            merged.append(note)
    return merged


def filter_guitar_range(notes: list, min_velocity: float = 0.08) -> list:
    """Filter notes to standard guitar range and quality."""
    filtered = []
    for n in notes:
        # Guitar pitch range: E2(40) to C6(84)
        if not (40 <= n["pitch"] <= 84):
            continue
        # Velocity threshold: remove ghost notes
        if n.get("velocity", 1.0) < min_velocity:
            continue
        filtered.append(n)

    # Limit simultaneous notes to 6 (max guitar strings)
    filtered.sort(key=lambda n: (n["start"], -n["velocity"]))
    result = []
    i = 0
    while i < len(filtered):
        group = [filtered[i]]
        j = i + 1
        while j < len(filtered) and filtered[j]["start"] - filtered[i]["start"] < 0.03:
            group.append(filtered[j])
            j += 1
        # Keep up to 6 strongest notes per time slot
        group.sort(key=lambda n: -n.get("velocity", 1.0))
        result.extend(group[:6])
        i = j

    result.sort(key=lambda n: (n["start"], n["pitch"]))
    return result
