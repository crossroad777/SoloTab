"""
Step 2: Build Chord Form Dictionary
=====================================
Analyze chords_dataset.jsonl → build frequency-based chord voicing dictionary
Output: chord_dictionary.json
"""
import json, sys
from pathlib import Path
from collections import Counter, defaultdict

DATA_DIR = Path(r"D:\Music\nextchord-solotab\backend\gp5_training\data")
CHORDS_FILE = DATA_DIR / "chords_dataset.jsonl"
OUTPUT_FILE = DATA_DIR / "chord_dictionary.json"


def chord_key(notes):
    """Create a canonical key for a chord based on intervals from bass"""
    pitches = sorted(n["pitch"] for n in notes)
    if not pitches:
        return None
    bass = pitches[0]
    intervals = tuple(p - bass for p in pitches)
    root_class = bass % 12  # pitch class of bass
    return (root_class, intervals)


def voicing_key(notes):
    """Create a string-fret voicing signature"""
    pairs = tuple(sorted((n["string"], n["fret"]) for n in notes))
    return pairs


def main():
    print("=" * 60)
    print("  Step 2: Chord Form Dictionary Builder")
    print("=" * 60)

    if not CHORDS_FILE.exists():
        print(f"  ERROR: {CHORDS_FILE} not found. Run Step 1 first.")
        return

    print("\n[1/3] Loading chords...")
    chords = []
    with open(CHORDS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            chords.append(json.loads(line))
    print(f"  Loaded {len(chords):,} chords")

    # Standard tuning filter (most common)
    STANDARD = [64, 59, 55, 50, 45, 40]
    std_chords = [c for c in chords if c.get("tuning", [])[:6] == STANDARD]
    print(f"  Standard tuning chords: {len(std_chords):,}")

    print("\n[2/3] Building chord voicing dictionary...")
    
    # Group by chord type (interval set from bass)
    chord_groups = defaultdict(list)
    for chord in std_chords:
        key = chord_key(chord["notes"])
        if key is None:
            continue
        chord_groups[key].append(chord)

    # For each chord type, count voicings
    dictionary = {}
    pitch_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    for (root_class, intervals), chord_list in chord_groups.items():
        if len(chord_list) < 3:  # minimum frequency
            continue
        
        voicing_counts = Counter()
        for chord in chord_list:
            vk = voicing_key(chord["notes"])
            voicing_counts[vk] += 1
        
        # Top voicings
        top_voicings = voicing_counts.most_common(10)
        
        root_name = pitch_names[root_class]
        interval_str = ",".join(str(i) for i in intervals)
        dict_key = f"{root_name}:{interval_str}"
        
        dictionary[dict_key] = {
            "root": root_name,
            "root_class": root_class,
            "intervals": list(intervals),
            "total_occurrences": len(chord_list),
            "voicings": [
                {
                    "string_fret_pairs": list(vk),
                    "count": count,
                    "frequency": round(count / len(chord_list), 3),
                }
                for vk, count in top_voicings
            ],
        }

    # Also build a pitch-to-preferred-string map
    print("\n[3/3] Building pitch-to-string preference map...")
    
    notes_file = DATA_DIR / "notes_dataset.jsonl"
    pitch_string_counts = defaultdict(lambda: Counter())
    
    if notes_file.exists():
        with open(notes_file, 'r', encoding='utf-8') as f:
            for line in f:
                note = json.loads(line)
                tuning = note.get("tuning", [])[:6]
                if tuning == STANDARD:
                    pitch_string_counts[note["pitch"]][note["string"]] += 1
    
    pitch_preferences = {}
    for pitch, string_counts in sorted(pitch_string_counts.items()):
        total = sum(string_counts.values())
        if total < 10:
            continue
        prefs = {}
        for s, c in string_counts.most_common():
            prefs[str(s)] = round(c / total, 3)
        pitch_preferences[str(pitch)] = {
            "total": total,
            "preferred_string": string_counts.most_common(1)[0][0],
            "distribution": prefs,
        }

    # Save
    output = {
        "chord_dictionary": dictionary,
        "pitch_string_preferences": pitch_preferences,
        "stats": {
            "total_chord_types": len(dictionary),
            "total_chords_analyzed": len(std_chords),
            "total_pitches_mapped": len(pitch_preferences),
        },
    }
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n  Saved: {OUTPUT_FILE}")
    print(f"  Chord types: {len(dictionary):,}")
    print(f"  Pitch preferences: {len(pitch_preferences)} pitches mapped")
    
    # Show top 10 most common chord types
    top_chords = sorted(dictionary.items(), key=lambda x: -x[1]["total_occurrences"])[:10]
    print(f"\n  Top 10 chord types:")
    for key, info in top_chords:
        top_v = info["voicings"][0] if info["voicings"] else {}
        print(f"    {key}: {info['total_occurrences']}x, top voicing: {top_v.get('string_fret_pairs', '?')}")

    print(f"\n{'=' * 60}")
    print(f"  COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
