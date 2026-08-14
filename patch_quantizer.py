import backend.music_quantizer as mq
from typing import List

# Read the file
with open("backend/music_quantizer.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    new_lines.append(line)
    if "note_metadata.append({" in line:
        new_lines.append('            "_is_in_arpeggio": is_in_arpeggio,\n')
    
    if "dur_divs = max(3, dur_divs)" in line:
        new_lines.append('        if int(orig.get("pitch", 60)) <= 55 and meta.get("_is_in_arpeggio", False):\n')
        new_lines.append('            dur_divs = max(dur_divs, DIVISIONS * 2)\n')

with open("backend/music_quantizer.py", "w", encoding="utf-8") as f:
    f.writelines(new_lines)
