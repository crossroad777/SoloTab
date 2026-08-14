import json

from music21 import stream, note

data = json.load(open(r"uploads/20260814-040009/notes_assigned_original.json", encoding="utf-8"))
notes = data.get("notes", data) if isinstance(data, dict) else data

for n in notes[:10]:
    if int(n.get("pitch")) == 40:
        print(n)

