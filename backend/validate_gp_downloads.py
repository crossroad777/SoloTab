"""Validate downloaded GP5 files from GProTab scraper."""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(__file__))

DL_DIR = r"D:\Music\nextchord-solotab\datasets\gprotab_downloads"
META_PATH = os.path.join(DL_DIR, "metadata.json")

with open(META_PATH, 'r', encoding='utf-8') as f:
    meta = json.load(f)

print(f"Total downloaded: {meta['total']}")
print(f"Failed URLs: {len(meta['failed'])}")

# Find all GP files
gp_files = []
for ext in ['*.gp', '*.gp3', '*.gp4', '*.gp5', '*.gpx']:
    gp_files.extend(glob.glob(os.path.join(DL_DIR, '**', ext), recursive=True))

print(f"GP files on disk: {len(gp_files)}")

# Try parsing with guitarpro
import guitarpro as gp

valid = 0
invalid = 0
guitar_tracks = 0
total_notes = 0
errors = []

for i, fp in enumerate(gp_files):
    try:
        song = gp.parse(fp)
        valid += 1
        # Count guitar tracks and notes
        for track in song.tracks:
            if len(track.strings) == 6:  # Guitar track
                guitar_tracks += 1
                for measure in track.measures:
                    for voice in measure.voices:
                        for beat in voice.beats:
                            total_notes += len(beat.notes)
    except Exception as e:
        invalid += 1
        if len(errors) < 10:
            errors.append(f"{os.path.basename(fp)}: {str(e)[:80]}")
    
    if (i + 1) % 100 == 0:
        print(f"  Checked {i+1}/{len(gp_files)}... valid={valid} invalid={invalid}")

print(f"\n{'='*60}")
print(f"  VALIDATION RESULTS")
print(f"{'='*60}")
print(f"  Files checked:  {len(gp_files)}")
print(f"  Valid:          {valid} ({valid/len(gp_files)*100:.1f}%)")
print(f"  Invalid:        {invalid}")
print(f"  Guitar tracks:  {guitar_tracks}")
print(f"  Total notes:    {total_notes:,}")

if errors:
    print(f"\n  Sample errors:")
    for e in errors[:5]:
        print(f"    {e}")

# Save validation result
result = {
    'files_checked': len(gp_files),
    'valid': valid,
    'invalid': invalid,
    'guitar_tracks': guitar_tracks,
    'total_notes': total_notes,
    'valid_rate': valid / len(gp_files) if gp_files else 0,
}
out = os.path.join(DL_DIR, "validation_result.json")
with open(out, 'w') as f:
    json.dump(result, f, indent=2)
print(f"\n  Saved: {out}")
