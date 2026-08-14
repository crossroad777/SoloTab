import json
import backend.music_quantizer as mq

data = json.load(open(r"uploads/20260814-040430/notes_assigned_original.json", encoding="utf-8"))
notes = data.get("notes", data) if isinstance(data, dict) else data
beats_data = json.load(open(r"uploads/20260814-040430/beats.json", encoding="utf-8"))

orig = mq.quantize_notes_music21.__code__
with open("backend/music_quantizer.py", "r", encoding="utf-8") as f:
    code = f.read()
code = code.replace("dur_ql = max(dur_sec / sec_per_beat, MIN_DUR_BEATS)", "dur_ql = max(dur_sec / sec_per_beat, MIN_DUR_BEATS)\n        if n['pitch'] == 40: print(f'BEFORE M21: sec={dur_sec} ql={dur_ql} is_chord={is_chord}')")
with open("backend/music_quantizer_debug.py", "w", encoding="utf-8") as f:
    f.write(code)

import backend.music_quantizer_debug as mqd
quantized = mqd.quantize_notes_music21(notes, beats_data["beats"], beats_data["bpm"], time_signature=beats_data["time_signature"], beats_per_bar=3, rhythm_subdivision="triplet")

for q in quantized[:15]:
    if q.get("pitch") == 40:
        print(f"pitch=40 dur={q.get('duration_divs')} start={q.get('start_time')}")

