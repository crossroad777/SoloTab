import json
data = json.load(open(r"uploads/20260814-040430/notes_assigned.json", encoding="utf-8"))
notes = data.get("notes", data) if isinstance(data, dict) else data

tied_notes = [n for n in notes if n.get("_tie_start") or n.get("_tie_stop")]
print(f"Tied notes generated: {len(tied_notes)}")

basses = [n for n in notes if int(n.get("pitch", 60)) <= 55]
long_basses = [n for n in basses if n.get("duration_divs", 0) >= 24]
print(f"Total bass notes: {len(basses)}")
print(f"Long bass notes (>= 2 beats): {len(long_basses)}")
