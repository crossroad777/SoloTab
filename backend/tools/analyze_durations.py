import json
import numpy as np

gt_path = r'D:\Music\datasets\GuitarSet\annotation\00_Funk1-97-C_solo.jams'
with open(gt_path, 'r') as f:
    jams_data = json.load(f)

for ann in jams_data['annotations']:
    if ann['namespace'] == 'note_midi':
        durations = [n['duration'] for n in ann['data']]
        if len(durations) > 0:
            print(f"String {ann['annotation_metadata']['data_source']}: Avg duration {np.mean(durations):.3f}s, Min {np.min(durations):.3f}s")
