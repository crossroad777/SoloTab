import sys
import os
import json

project_root = r"D:\Music\nextchord-solotab"
sys.path.insert(0, os.path.join(project_root, "music-transcription", "python"))

from data_processing.preparation import extract_annotations_from_jams
import config

jams_path = r"D:\Music\all_jams_midi_V2_60000_tracks\outall\1 God - Grace - gp4__1 - Acoustic Nylon Guitar__midi\1 - Acoustic Nylon Guitar.jams"

print("Reading", jams_path)
notes = extract_annotations_from_jams(jams_path)
print("EXTRACTED NOTES:", len(notes))
if len(notes) > 0:
    print("FIRST NOTE:", notes[0])
