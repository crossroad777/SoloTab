import backend.music_quantizer as mq
with open("backend/music_quantizer.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
skip = False
for line in lines:
    if "meta.get(\"_is_in_arpeggio\"" in line:
        skip = True
        continue
    if skip and "dur_divs = max(dur_divs" in line:
        skip = False
        continue
    if "meta.get(\"_is_chord\"" in line:
        skip = True
        continue
    if "dur_divs = max(dur_divs" in line and skip:
        skip = False
        continue
    new_lines.append(line)

with open("backend/music_quantizer.py", "w", encoding="utf-8") as f:
    f.writelines(new_lines)
