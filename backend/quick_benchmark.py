"""Quick GuitarSet benchmark to verify PIMA impact"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
from optimize_string_assignment import load_guitarset_groundtruth, evaluate_assignment

ANNOTATION_DIRS = [
    r"D:\Music\Datasets\GuitarSet\annotation",
    r"D:\Music\nextchord-solotab\datasets\GuitarSet\annotation",
]

ann_dir = None
for d in ANNOTATION_DIRS:
    if os.path.exists(d):
        ann_dir = d
        break

if ann_dir is None:
    print("GuitarSet annotation dir not found")
    sys.exit(1)

print("Loading GuitarSet...")
songs = load_guitarset_groundtruth(ann_dir, max_songs=10)
total = sum(len(s['notes']) for s in songs)
print(f"Songs: {len(songs)}, Notes: {total}")

print("Evaluating...")
t0 = time.time()
result = evaluate_assignment(songs)
elapsed = time.time() - t0

acc = result['string_accuracy']
correct = result['string_correct']
total_n = result['total_notes']
print(f"String accuracy: {acc:.4f} ({correct}/{total_n}) [{elapsed:.1f}s]")
