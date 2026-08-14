import json
data = json.load(open(r"uploads/20260814-040009/notes_assigned.json", encoding="utf-8"))
notes = data.get("notes", data) if isinstance(data, dict) else data
for n in notes:
    if 36 <= n.get("beat_pos_absolute") <= 48:
        pos = n.get("beat_pos_absolute")
        dur = n.get("duration_divs")
        pitch = n.get("pitch")
        s = n.get("string")
        print(f"pos={pos} dur={dur} pitch={pitch} str={s}")
