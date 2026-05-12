"""Run Optuna optimization with PIMA weights (50 trials)"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
from optimize_string_assignment import (
    load_guitarset_groundtruth, evaluate_assignment, run_optimization
)

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

print("Baseline evaluation...")
t0 = time.time()
baseline = evaluate_assignment(songs)
bl_acc = baseline['string_accuracy']
bl_correct = baseline['string_correct']
bl_total = baseline['total_notes']
print(f"Baseline: {bl_acc:.4f} ({bl_correct}/{bl_total}) [{time.time()-t0:.1f}s]")

print(f"\nOptuna optimization (50 trials)...")
t0 = time.time()
study = run_optimization(songs, n_trials=50)

best = study.best_trial
improvement = (best.value - bl_acc) * 100
print(f"\n{'='*60}")
print(f"Optimization done ({time.time()-t0:.0f}s)")
print(f"Baseline: {bl_acc:.4f}")
print(f"Best:     {best.value:.4f}")
print(f"Improvement: {improvement:+.2f}%")
print(f"\nBest params:")
for k, v in sorted(best.params.items()):
    print(f"  {k}: {v:.4f}")
print(f"{'='*60}")

# Save results
import json
result = {
    'baseline_accuracy': bl_acc,
    'best_accuracy': best.value,
    'best_params': best.params,
    'n_songs': len(songs),
    'n_notes': bl_total,
}
out_path = os.path.join(os.path.dirname(__file__), 'optimized_weights_pima.json')
with open(out_path, 'w') as f:
    json.dump(result, f, indent=2)
print(f"Saved: {out_path}")
