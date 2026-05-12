"""
tab_data_api.py — TAB描画用の構造化JSONデータを生成
===================================================
notes_assigned.json + beats.json を読み込み、
小節単位に構造化したTABレンダリング用データを返す。
"""
import json
from pathlib import Path


def build_tab_render_data(session_dir: str | Path, session_meta: dict) -> dict:
    """
    Returns:
        {
            "title": str,
            "bpm": int,
            "timeSignature": "3/4",
            "tuning": [40,45,50,55,59,64],
            "measures": [
                {
                    "index": 0,
                    "startTime": 1.89,   # seconds
                    "endTime": 3.26,
                    "notes": [
                        {"beat": 0, "string": 1, "fret": 7, "duration": "eighth"},
                        ...
                    ],
                    "bassNotes": [
                        {"beat": 0, "string": 6, "fret": 0, "duration": "half."},
                    ]
                },
                ...
            ]
        }
    """
    session_dir = Path(session_dir)

    # Load notes
    assigned_path = session_dir / "notes_assigned.json"
    if not assigned_path.exists():
        return {"error": "notes not found", "measures": []}
    with open(assigned_path, "r", encoding="utf-8") as f:
        notes_data = json.load(f)
    notes = notes_data if isinstance(notes_data, list) else notes_data.get("notes", [])

    # Load beats
    beats_path = session_dir / "beats.json"
    beats, bpm, time_sig = [], 120, "4/4"
    if beats_path.exists():
        with open(beats_path, "r", encoding="utf-8") as f:
            bd = json.load(f)
        beats = bd.get("beats", [])
        bpm = bd.get("bpm", 120)
        time_sig = bd.get("time_signature", "4/4")

    # Parse time signature
    parts = time_sig.split("/")
    beats_per_bar = int(parts[0]) if len(parts) == 2 else 4

    # Calculate expected beat interval from BPM
    expected_interval = 60.0 / bpm
    if len(beats) > 2:
        avg_interval = (beats[-1] - beats[0]) / (len(beats) - 1)
        ratio = avg_interval / expected_interval
        if ratio < 0.8 or ratio > 1.2:
            # beats.json is at sub-beat level (e.g. triplet pulses)
            # Use BPM-based bar timing instead
            first_beat = beats[0]
            bar_dur = beats_per_bar * expected_interval
            measures_data = []
            bar_idx = 0
            while True:
                bar_start = first_beat + bar_idx * bar_dur
                bar_end = bar_start + bar_dur
                if bar_start > (beats[-1] + bar_dur):
                    break
                bar_notes = [n for n in notes
                             if float(n["start"]) >= bar_start and float(n["start"]) < bar_end]
                if not bar_notes and bar_idx > 5:
                    break
                measures_data.append(_build_measure(
                    bar_idx, bar_start, bar_end, bar_notes,
                    beats_per_bar, expected_interval
                ))
                bar_idx += 1
            return _finalize(session_meta, bpm, time_sig, measures_data)

    # Normal mode: use beats directly
    measures_data = []
    beat_idx = 0
    bar_idx = 0
    while beat_idx < len(beats):
        bar_start = beats[beat_idx]
        end_beat_idx = beat_idx + beats_per_bar
        bar_end = beats[end_beat_idx] if end_beat_idx < len(beats) else bar_start + beats_per_bar * expected_interval
        bar_notes = [n for n in notes
                     if float(n["start"]) >= bar_start and float(n["start"]) < bar_end]
        measures_data.append(_build_measure(
            bar_idx, bar_start, bar_end, bar_notes,
            beats_per_bar, expected_interval
        ))
        beat_idx += beats_per_bar
        bar_idx += 1

    return _finalize(session_meta, bpm, time_sig, measures_data)


def _build_measure(idx, start, end, notes, beats_per_bar, beat_interval):
    """Build a single measure's data."""
    bar_dur = end - start
    melody = []
    bass = []
    for n in sorted(notes, key=lambda x: float(x["start"])):
        s = int(n.get("string", 1))
        f = int(n.get("fret", 0))
        t = float(n["start"])
        # Beat position within bar (0-based float)
        beat_pos = (t - start) / beat_interval if beat_interval > 0 else 0
        entry = {
            "beatPos": round(beat_pos, 3),
            "string": s,
            "fret": f,
            "technique": n.get("technique"),
        }
        if int(n.get("pitch", 60)) <= 52:  # bass notes
            bass.append(entry)
        else:
            melody.append(entry)
    return {
        "index": idx,
        "startTime": round(start, 4),
        "endTime": round(end, 4),
        "notes": melody,
        "bassNotes": bass,
    }


def _finalize(meta, bpm, time_sig, measures):
    title = meta.get("filename", "Guitar TAB")
    # Remove audio extension from title
    import os
    name, ext = os.path.splitext(title)
    if ext.lower() in {".mp3", ".wav", ".m4a", ".flac", ".ogg"}:
        title = name
    return {
        "title": title,
        "bpm": round(bpm),
        "timeSignature": time_sig,
        "totalMeasures": len(measures),
        "measures": measures,
    }
