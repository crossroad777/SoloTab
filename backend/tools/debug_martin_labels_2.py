import sys
import os
import json

project_root = r"D:\Music\nextchord-solotab"
sys.path.insert(0, os.path.join(project_root, "backend"))
sys.path.insert(0, os.path.join(project_root, "music-transcription", "python"))

from resume_martin_training import MartinSynthDataset # type: ignore

ds = MartinSynthDataset(r"D:\Music\datasets\martin_finger", augment=False)
if len(ds) > 0:
    jams_path = ds.tracks[0]['jams']
    with open(jams_path, 'r', encoding='utf-8') as f:
        jdata = json.load(f)
        
    for i, a in enumerate(jdata.get("annotations", [])):
        if a.get("namespace") == "note_tab":
            print(f"--- note_tab [{i}] ---")
            print({k: v for k, v in a.items() if k != 'data'})
            print(f"Data length: {len(a.get('data', []))}")
            if len(a.get('data', [])) > 0:
                print("First data item:", a['data'][0])
