"""
task_901_inventory_and_inspection.py
====================================
TASK-901: ローカルMIDIアセットの棚卸しとブラックボックスの透明化
"""

import os
import sys
import pathlib
import json
import mido
import guitarpro
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.abspath("backend"))

from solotab_utils import STANDARD_TUNING
from pipeline import run_pipeline
from string_assigner import assign_strings_dp

# ノート番号から音名（C4, E2など）への変換テーブル
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
def midi_to_pitch_name(pitch: int) -> str:
    octave = (pitch // 12) - 1
    name = NOTE_NAMES[pitch % 12]
    return f"{name}{octave}"


def step_1_inventory_midi_files():
    search_dirs = [
        pathlib.Path("."),
        pathlib.Path("datasets"),
        pathlib.Path("outputs"),
        pathlib.Path("backend"),
    ]
    
    found_files = []
    seen = set()
    
    for sd in search_dirs:
        if sd.exists():
            for p in sd.rglob("*.mid*"):
                if p.is_file() and p.resolve() not in seen:
                    seen.add(p.resolve())
                    try:
                        size_bytes = p.stat().st_size
                        mid = mido.MidiFile(str(p))
                        n_tracks = len(mid.tracks)
                        total_notes = sum(1 for t in mid.tracks for msg in t if msg.type == 'note_on' and msg.velocity > 0)
                        found_files.append({
                            "path": str(p).replace("\\", "/"),
                            "filename": p.name,
                            "size_bytes": size_bytes,
                            "tracks_count": n_tracks,
                            "total_notes": total_notes,
                            "inferred_usage": "Guitar / SoloTab Score" if "romance" in p.name.lower() or "solo" in p.name.lower() else "General MIDI / Test Asset"
                        })
                    except Exception as e:
                        found_files.append({
                            "path": str(p).replace("\\", "/"),
                            "filename": p.name,
                            "size_bytes": p.stat().st_size,
                            "tracks_count": "error",
                            "total_notes": "error",
                            "inferred_usage": f"Unreadable ({e})"
                        })

    # romance_clean.mid を生成して使用可能なテストアセットを確定
    romance_clean_path = pathlib.Path("outputs/romance_clean.mid")
    romance_clean_path.parent.mkdir(parents=True, exist_ok=True)
    
    gt_json_path = pathlib.Path("backend/ground_truth/romance_forbidden_games.json")
    with open(gt_json_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
        
    gt_notes = []
    current_t = 0.0
    for m in gt_data["measures_detailed"]:
        for n in m["notes"]:
            gt_notes.append({
                "start": round(current_t, 4),
                "end": round(current_t + 0.22, 4),
                "duration": 0.22,
                "string": n["string"],
                "fret": n["fret"],
                "pitch": n["pitch"],
                "velocity": 80
            })
            current_t += 0.25

    mid_clean = mido.MidiFile()
    track = mido.MidiTrack()
    mid_clean.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(80), time=0))
    track.append(mido.MetaMessage('time_signature', numerator=3, denominator=4, time=0))
    
    last_time = 0.0
    for n in gt_notes:
        dt_on = int(max(0, (n["start"] - last_time) * 480))
        track.append(mido.Message('note_on', note=n["pitch"], velocity=n["velocity"], time=dt_on))
        dt_off = int(max(10, n["duration"] * 480))
        track.append(mido.Message('note_off', note=n["pitch"], velocity=0, time=dt_off))
        last_time = n["start"] + n["duration"]
        
    mid_clean.save(str(romance_clean_path))
    
    found_files.append({
        "path": str(romance_clean_path).replace("\\", "/"),
        "filename": romance_clean_path.name,
        "size_bytes": romance_clean_path.stat().st_size,
        "tracks_count": 1,
        "total_notes": len(gt_notes),
        "inferred_usage": "Clean Classical Solo Guitar Benchmark (Romance)"
    })
    
    return found_files, romance_clean_path, gt_notes


def step_2_dissect_midi(midi_path: pathlib.Path, limit: int = 10):
    mid = mido.MidiFile(str(midi_path))
    tempo = 500000
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                break

    ticks_per_beat = mid.ticks_per_beat
    
    notes = []
    current_time_sec = 0.0
    active_notes = {}
    
    for track in mid.tracks:
        track_time_ticks = 0
        for msg in track:
            track_time_ticks += msg.time
            time_sec = mido.tick2second(track_time_ticks, ticks_per_beat, tempo)
            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes[msg.note] = (time_sec, msg.velocity)
            elif (msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0)) and msg.note in active_notes:
                start_sec, vel = active_notes.pop(msg.note)
                dur_sec = round(time_sec - start_sec, 4)
                notes.append({
                    "time_sec": round(start_sec, 4),
                    "pitch_midi": msg.note,
                    "pitch_name": midi_to_pitch_name(msg.note),
                    "duration_sec": dur_sec,
                    "velocity": vel
                })
                
    notes.sort(key=lambda x: x["time_sec"])
    
    formatted_dump = []
    for idx, n in enumerate(notes[:limit], 1):
        formatted_dump.append({
            "index": idx,
            "time_sec": n["time_sec"],
            "pitch_midi": n["pitch_midi"],
            "pitch_name": n["pitch_name"],
            "duration_sec": n["duration_sec"],
            "velocity": n["velocity"]
        })
    return formatted_dump, notes


def step_3_visualize_solotab_translation(raw_notes, limit: int = 10):
    tuning = STANDARD_TUNING # [40, 45, 50, 55, 59, 64]
    
    pipeline_notes = []
    for n in raw_notes:
        pipeline_notes.append({
            "start": n["time_sec"],
            "end": n["time_sec"] + n["duration_sec"],
            "duration": n["duration_sec"],
            "pitch": n["pitch_midi"],
            "velocity": n["velocity"] / 127.0
        })
        
    assigned = assign_strings_dp(
        pipeline_notes,
        tuning=tuning,
        audio_path=None
    )
    
    translation_table = []
    for idx, a in enumerate(assigned[:limit], 1):
        p_in = a["pitch"]
        s = a["string"]
        f = a["fret"]
        p_comp = tuning[6 - s] + f
        translation_table.append({
            "index": idx,
            "pitch_name": midi_to_pitch_name(p_in),
            "input_midi": p_in,
            "assigned_string": s,
            "assigned_fret": f,
            "computed_pitch": p_comp,
            "match": (p_in == p_comp)
        })
    return translation_table, assigned


def step_4_render_gp5_and_tab_text(assigned_notes, romance_clean_path):
    session_dir = pathlib.Path("outputs/task_901_inspection")
    session_dir.mkdir(parents=True, exist_ok=True)
    dummy_wav = session_dir / "converted.wav"
    sr = 22050
    t_arr = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)
    dummy_sig = 0.1 * np.sin(2 * np.pi * 220.0 * t_arr)
    sf.write(str(dummy_wav), dummy_sig, sr)
    
    res = run_pipeline(
        "task_901_inspection", session_dir, dummy_wav,
        tuning_name="standard",
        transcription_profile="classic",
        midi_path=romance_clean_path
    )
    
    gp5_path = session_dir / "tab.gp5"
    out_gp = guitarpro.parse(str(gp5_path))
    
    # 1小節目（Measure 1）のTAB譜テキスト表現を構築
    measure1 = out_gp.tracks[0].measures[0]
    
    # 6弦 (e=1, B=2, G=3, D=4, A=5, E=6)
    tab_lines = {1: "e|", 2: "B|", 3: "G|", 4: "D|", 5: "A|", 6: "E|"}
    
    # Measure 1 の全ビート・ノートを抽出
    beats_m1 = []
    for v in measure1.voices:
        for b in v.beats:
            if b.notes:
                beats_m1.append(b)
    beats_m1.sort(key=lambda b: b.start)
    
    # ビートごとにフレットを配置
    for b in beats_m1:
        fret_by_string = {n.string: str(n.value) for n in b.notes}
        max_w = max(len(f) for f in fret_by_string.values()) if fret_by_string else 1
        
        for s in range(1, 7):
            if s in fret_by_string:
                val = fret_by_string[s]
                tab_lines[s] += f"-{val.ljust(max_w, '-')}-"
            else:
                tab_lines[s] += f"-{'-' * max_w}-"
                
    for s in range(1, 7):
        tab_lines[s] += "|"
        
    tab_text = "\n".join([tab_lines[s] for s in [1, 2, 3, 4, 5, 6]])
    return str(gp5_path).replace("\\", "/"), tab_text


def main():
    found_files, romance_clean_path, gt_notes = step_1_inventory_midi_files()
    step2_dump, raw_notes = step_2_dissect_midi(romance_clean_path, 10)
    step3_table, assigned_notes = step_3_visualize_solotab_translation(raw_notes, 10)
    gp5_file_path, tab_text = step_4_render_gp5_and_tab_text(assigned_notes, romance_clean_path)
    
    output = {
        "step_1_local_midi_inventory": found_files,
        "step_2_midi_dissection_first_10_notes": step2_dump,
        "step_3_solotab_transformer_v3_translation": step3_table,
        "step_4_final_gp5_artifact": {
            "gp5_path": gp5_file_path,
            "measure_1_tablature": tab_text
        }
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
