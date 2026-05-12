"""
Optuna optimization - expanded scale
Stage 1: 20 songs x 200 trials
Stage 2: Full GuitarSet verification with best params
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__))

# Suppress verbose output during optimization
import logging
logging.disable(logging.WARNING)

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

# ============================================================
# Stage 1: 20 songs x 200 trials
# ============================================================
print("=" * 60)
print("  Stage 1: 20 songs x 200 trials")
print("=" * 60)

songs_20 = load_guitarset_groundtruth(ann_dir, max_songs=20)
total_20 = sum(len(s['notes']) for s in songs_20)
print(f"  Songs: {len(songs_20)}, Notes: {total_20}")

print("  Baseline evaluation...")
t0 = time.time()
baseline = evaluate_assignment(songs_20)
bl_time = time.time() - t0
bl_acc = baseline['string_accuracy']
print(f"  Baseline: {bl_acc:.4f} ({baseline['string_correct']}/{baseline['total_notes']}) [{bl_time:.1f}s]")
print(f"  Estimated time: {bl_time * 200 / 60:.0f} min for 200 trials")

print(f"\n  Optimizing (200 trials)...")
t0 = time.time()
study = run_optimization(songs_20, n_trials=200)
opt_time = time.time() - t0

best = study.best_trial
print(f"\n  Stage 1 Results ({opt_time:.0f}s):")
print(f"  Baseline: {bl_acc:.4f}")
print(f"  Best:     {best.value:.4f}")
print(f"  Improvement: {(best.value - bl_acc)*100:+.2f}%")

best_params = best.params

# ============================================================
# Stage 2: Verify on full GuitarSet
# ============================================================
print(f"\n{'='*60}")
print("  Stage 2: Full GuitarSet verification")
print("=" * 60)

all_songs = load_guitarset_groundtruth(ann_dir)
total_all = sum(len(s['notes']) for s in all_songs)
print(f"  Songs: {len(all_songs)}, Notes: {total_all}")

print("  Evaluating with current weights...")
t0 = time.time()
current_result = evaluate_assignment(all_songs)
print(f"  Current: {current_result['string_accuracy']:.4f} [{time.time()-t0:.1f}s]")

print("  Evaluating with optimized weights...")
t0 = time.time()
optimized_result = evaluate_assignment(all_songs, best_params)
print(f"  Optimized: {optimized_result['string_accuracy']:.4f} [{time.time()-t0:.1f}s]")

improvement = (optimized_result['string_accuracy'] - current_result['string_accuracy']) * 100

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print("  FINAL RESULTS")
print(f"{'='*60}")
print(f"  20-song optimization: {bl_acc:.4f} -> {best.value:.4f} (+{(best.value-bl_acc)*100:.2f}%)")
print(f"  Full GuitarSet:       {current_result['string_accuracy']:.4f} -> {optimized_result['string_accuracy']:.4f} ({improvement:+.2f}%)")
print(f"\n  Best params:")
for k, v in sorted(best_params.items()):
    print(f"    {k}: {v:.4f}")

# Save
result = {
    'stage1_baseline': bl_acc,
    'stage1_best': best.value,
    'stage1_songs': len(songs_20),
    'stage1_notes': total_20,
    'stage1_trials': 200,
    'full_current': current_result['string_accuracy'],
    'full_optimized': optimized_result['string_accuracy'],
    'full_songs': len(all_songs),
    'full_notes': total_all,
    'best_params': best_params,
}
out_path = os.path.join(os.path.dirname(__file__), 'optimized_weights_v2.json')
with open(out_path, 'w') as f:
    json.dump(result, f, indent=2)
print(f"\n  Saved: {out_path}")
