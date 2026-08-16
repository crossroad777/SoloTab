"""
test_romance_bypass_e2e.py
"""
import sys, os, pathlib, json
import numpy as np
import soundfile as sf
import mido
import guitarpro

sys.path.insert(0, os.path.abspath("backend"))

from pipeline import run_pipeline

# 参照 GT データの読み込み
gt_json_path = pathlib.Path("backend/ground_truth/romance_forbidden_games.json")
with open(gt_json_path, "r", encoding="utf-8") as f:
    gt_data = json.load(f)
    
gt_notes = []
current_t = 0.0
for m in gt_data["measures_detailed"]:
    for n in m["notes"]:
        gt_notes.append({
            "start": current_t,
            "end": current_t + 0.22,
            "string": n["string"],
            "fret": n["fret"],
            "pitch": n["pitch"],
            "dur": 0.22
        })
        current_t += 0.25
        
test_midi = pathlib.Path("backend/benchmark/romance_bypass_test.mid")
mid = mido.MidiFile()
track = mido.MidiTrack()
mid.tracks.append(track)

last_time = 0.0
for n in gt_notes:
    dt_on = int(max(0, (n["start"] - last_time) * 480))
    track.append(mido.Message('note_on', note=n["pitch"], velocity=80, time=dt_on))
    dt_off = int(max(10, n["dur"] * 480))
    track.append(mido.Message('note_off', note=n["pitch"], velocity=0, time=dt_off))
    last_time = n["start"] + n["dur"]
mid.save(str(test_midi))

session_dir = pathlib.Path("backend/benchmark/romance_bypass_test_session")
session_dir.mkdir(parents=True, exist_ok=True)
dummy_wav = session_dir / "converted.wav"
sr = 22050
t_arr = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)
dummy_sig = 0.1 * np.sin(2 * np.pi * 220.0 * t_arr)
for c in range(0, len(dummy_sig), int(sr * 0.5)):
    dummy_sig[c:c+100] += 0.8
sf.write(str(dummy_wav), dummy_sig, sr)

pipeline_res = run_pipeline(
    "romance_bypass_test_session", session_dir, dummy_wav,
    tuning_name="standard",
    transcription_profile="classic",
    midi_path=test_midi
)

# 出力GP5と突き合わせ
out_gp5_path = session_dir / "tab.gp5"
out_gp = guitarpro.parse(str(out_gp5_path))
out_notes = []
tuning_arr = [64, 59, 55, 50, 45, 40]
for m in out_gp.tracks[0].measures:
    for v in m.voices:
        for b in v.beats:
            for n in b.notes:
                out_notes.append({
                    "string": n.string,
                    "fret": n.value,
                    "pitch": tuning_arr[n.string - 1] + n.value
                })
                
comp_len = min(len(gt_notes), len(out_notes))
exact_matches = 0
for i in range(comp_len):
    if gt_notes[i]["string"] == out_notes[i]["string"] and gt_notes[i]["fret"] == out_notes[i]["fret"]:
        exact_matches += 1
        
match_rate = round(exact_matches / comp_len, 4) if comp_len > 0 else 0.0

print("\n" + "=" * 50)
print(f"Source GT Notes Count   : {len(gt_notes)}")
print(f"Output GP5 Notes Count  : {len(out_notes)}")
print(f"Exact Matches (String&Fret): {exact_matches} / {comp_len}")
print(f"Match Rate              : {match_rate:.2%}")
print("=" * 50)

test_midi.unlink(missing_ok=True)
import shutil
shutil.rmtree(str(session_dir), ignore_errors=True)
