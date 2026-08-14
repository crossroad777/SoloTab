import json
import backend.music_quantizer as mq

data = json.load(open(r"uploads/20260814-040430/notes_assigned_original.json", encoding="utf-8"))
notes = data.get("notes", data) if isinstance(data, dict) else data
beats_data = json.load(open(r"uploads/20260814-040430/beats.json", encoding="utf-8"))

entries = mq.quantize_notes_music21(notes, beats_data["beats"], beats_data["bpm"], time_signature=beats_data["time_signature"], beats_per_bar=3, rhythm_subdivision="triplet")

e_40 = next(e for e in entries if e["pitch"] == 40 and e["start_time"] == 1.904)
print(f"After quantize: dur={e_40['duration_divs']} pos={e_40['beat_pos_absolute']}")

entries = mq._fix_onset_collisions(entries, 3, 4)
e_40 = next(e for e in entries if e["pitch"] == 40 and e["start_time"] == 1.904)
print(f"After fix cols: dur={e_40['duration_divs']} pos={e_40['beat_pos_absolute']}")

mq._cap_durations_by_string(entries, 3)
e_40 = next(e for e in entries if e["pitch"] == 40 and e["start_time"] == 1.904)
print(f"After cap durs: dur={e_40['duration_divs']} pos={e_40['beat_pos_absolute']}")
