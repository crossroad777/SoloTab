import os, sys, json, numpy as np

mt = r"D:\Music\nextchord-solotab\music-transcription\python"
PROCESSED_DIR = os.path.join(mt, "_processed_guitarset_data")

with open(os.path.join(PROCESSED_DIR, "test_ids.txt")) as f:
    test_ids = set(l.strip() for l in f if l.strip())
with open(os.path.join(PROCESSED_DIR, "validation_ids.txt")) as f:
    val_ids = set(l.strip() for l in f if l.strip())

results_path = r"D:\Music\nextchord-solotab\backend\benchmark\detailed_benchmark_results.json"
with open(results_path) as f:
    data = json.load(f)

per_track = data["per_track"]
genre_map = {"BN": "Bossa Nova", "Funk": "Funk", "Jazz": "Jazz", "Rock": "Rock", "SS": "Singer-Songwriter"}

for split_name, split_ids in [("TEST", test_ids), ("VALIDATION", val_ids)]:
    split_results = [r for r in per_track if r["name"] in split_ids]
    if split_results:
        f1s = [r["f1"] for r in split_results]
        ps = [r["precision"] for r in split_results]
        rs = [r["recall"] for r in split_results]
        print(f"{split_name} ({len(split_results)} tracks): F1={np.mean(f1s):.4f}  P={np.mean(ps):.4f}  R={np.mean(rs):.4f}")
        for g in ["BN", "Funk", "Jazz", "Rock", "SS"]:
            gr = [r for r in split_results if r["genre"] == g]
            if gr:
                gf1 = np.mean([r["f1"] for r in gr])
                print(f"  {genre_map[g]:<22} N={len(gr):>2}  F1={gf1:.4f}")
        # worst tracks
        sorted_r = sorted(split_results, key=lambda x: x["f1"])
        print(f"  Worst: {sorted_r[0]['name']} F1={sorted_r[0]['f1']:.4f}")
        print(f"  Best:  {sorted_r[-1]['name']} F1={sorted_r[-1]['f1']:.4f}")
        print()

train_r = [r for r in per_track if r["name"] not in test_ids and r["name"] not in val_ids]
f1s = [r["f1"] for r in train_r]
ps = [r["precision"] for r in train_r]
rs = [r["recall"] for r in train_r]
print(f"TRAIN ({len(train_r)} tracks): F1={np.mean(f1s):.4f}  P={np.mean(ps):.4f}  R={np.mean(rs):.4f}")
