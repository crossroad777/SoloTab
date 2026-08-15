import sys
import os
import json
sys.path.insert(0, '.')
from e2e_pipeline_benchmark import load_jams_notes_with_string, ANNOTATIONS_DIR

gt = load_jams_notes_with_string(os.path.join(ANNOTATIONS_DIR, '02_Funk2-119-G_comp.jams'))
print(f"Total GT notes: {len(gt)}")
for n in gt[:20]:
    p = n['pitch']
    s = n['string']
    # Standard tuning: [40, 45, 50, 55, 59, 64]
    tuning = [40, 45, 50, 55, 59, 64]
    open_p = tuning[6 - s]
    f = p - open_p
    print(f"Time: {n['start']:.2f}s | Pitch: {p} | String: {s} | Fret: {f}")
