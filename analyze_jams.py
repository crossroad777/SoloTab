import json
with open(r"D:\Music\nextchord-solotab\datasets\GuitarSet\00_BN1-129-Eb_solo.jams", "r") as f:
    jams = json.load(f)

print("=== JAMS Structure ===")
print("Top-level keys:", list(jams.keys()))
print("Duration:", jams["file_metadata"]["duration"], "s")
print("Annotations:", len(jams["annotations"]))
for i, ann in enumerate(jams["annotations"]):
    ns = ann["namespace"]
    n_obs = len(ann["data"])
    print(f"  [{i}] namespace={ns}, observations={n_obs}")

# Show note_tab sample
for ann in jams["annotations"]:
    if "note" in ann["namespace"]:
        print(f"\n=== {ann['namespace']} (first 5) ===")
        for obs in ann["data"][:5]:
            print(f"  time={obs['time']:.4f}, dur={obs['duration']:.4f}, value={obs['value']}")
        break
