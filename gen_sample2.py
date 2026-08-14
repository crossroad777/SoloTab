import json
import backend.music_quantizer as mq

data = json.load(open(r"uploads/20260814-040430/notes_assigned_original.json", encoding="utf-8"))
notes = data.get("notes", data) if isinstance(data, dict) else data
beats_data = json.load(open(r"uploads/20260814-040430/beats.json", encoding="utf-8"))

entries = mq.quantize_notes_music21(notes, beats_data["beats"], beats_data["bpm"], time_signature=beats_data["time_signature"], beats_per_bar=3, rhythm_subdivision="triplet")
entries = mq._fix_onset_collisions(entries, 3, 4)
mq._cap_durations_by_string(entries, 3)

lines = []
for n in entries[:12]:
    p = n.get("pitch")
    s = n.get("string")
    f = n.get("fret")
    pos = n.get("beat_pos_absolute")
    dur = n.get("duration_divs")
    tie_start = n.get("_tie_start")
    tie_stop = n.get("_tie_stop")
    lines.append(f"- `pos={pos:2d}`: pitch={p} (Str {s}, Fret {f}) | dur={dur} | tie_start={tie_start} tie_stop={tie_stop}")

with open("scratch/sample_notes.md", "w") as f:
    f.write("\n".join(lines))
