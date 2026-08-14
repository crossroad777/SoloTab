import json
data = json.load(open(r"uploads/20260814-040430/notes_assigned.json", encoding="utf-8"))
notes = data.get("notes", data) if isinstance(data, dict) else data

lines = []
for n in notes[:12]:
    p = n.get("pitch")
    s = n.get("string")
    f = n.get("fret")
    pos = n.get("beat_pos_absolute")
    dur = n.get("duration_divs")
    tie_start = n.get("_tie_start")
    tie_stop = n.get("_tie_stop")
    lines.append(f"- `pos={pos}`: pitch={p} (Str {s}, Fret {f}) | dur={dur} | tie_start={tie_start} tie_stop={tie_stop}")

with open("scratch/sample_notes.md", "w") as f:
    f.write("\n".join(lines))
