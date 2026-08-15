import sys
import os

# Add the backend directory to sys.path to import music_theory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from music_theory import apply_physical_constraints

def test_same_string_removal():
    print("Test 1: Removing simultaneous notes on the same string")
    
    # Mock notes
    notes = [
        {"start": 1.0, "pitch": 60, "string": 3, "fret": 5, "velocity": 0.8},
        {"start": 1.01, "pitch": 62, "string": 3, "fret": 7, "velocity": 0.5}, # Same string, lower velocity
        {"start": 1.0, "pitch": 64, "string": 2, "fret": 5, "velocity": 0.9}, # Different string, should be kept
    ]
    
    filtered = apply_physical_constraints(notes, max_span=6, time_threshold=0.03)
    
    # We expect 2 notes (the one on string 3 with 0.8 vel, and the one on string 2)
    assert len(filtered) == 2, f"Expected 2 notes, got {len(filtered)}"
    
    pitches = [n["pitch"] for n in filtered]
    assert 60 in pitches
    assert 64 in pitches
    assert 62 not in pitches
    
    print("✓ Test 1 Passed")


def test_impossible_stretch_removal():
    print("Test 2: Removing impossible stretches (>6 frets)")
    
    # Mock notes for a chord at start=2.0
    notes = [
        {"start": 2.0, "pitch": 60, "string": 4, "fret": 2, "velocity": 0.9}, # Fret 2
        {"start": 2.0, "pitch": 64, "string": 3, "fret": 9, "velocity": 0.4}, # Fret 9 -> 9 - 2 = 7 > 6 span! This should be dropped (lowest velocity)
        {"start": 2.0, "pitch": 67, "string": 2, "fret": 3, "velocity": 0.8}, # Fret 3
    ]
    
    filtered = apply_physical_constraints(notes, max_span=6, time_threshold=0.03)
    
    # We expect 2 notes (the impossible stretch fret 9 should be removed because it has the lowest velocity)
    assert len(filtered) == 2, f"Expected 2 notes, got {len(filtered)}"
    
    pitches = [n["pitch"] for n in filtered]
    assert 60 in pitches
    assert 67 in pitches
    assert 64 not in pitches
    
    print("✓ Test 2 Passed")


def test_open_string_stretch():
    print("Test 3: Open strings should not count towards stretch")
    
    # Mock notes for a chord at start=3.0
    notes = [
        {"start": 3.0, "pitch": 40, "string": 6, "fret": 0, "velocity": 0.9}, # Fret 0 (Open string)
        {"start": 3.0, "pitch": 64, "string": 3, "fret": 9, "velocity": 0.8}, # Fret 9
        {"start": 3.0, "pitch": 67, "string": 2, "fret": 8, "velocity": 0.8}, # Fret 8
    ]
    
    filtered = apply_physical_constraints(notes, max_span=6, time_threshold=0.03)
    
    # Since the fretted notes are fret 8 and 9 (span = 1), this is perfectly playable. Open string is ignored for stretch.
    # We expect all 3 notes to be kept.
    assert len(filtered) == 3, f"Expected 3 notes, got {len(filtered)}"
    
    print("✓ Test 3 Passed")

if __name__ == "__main__":
    print("Running music_theory safety tests (no GPU required)...")
    test_same_string_removal()
    test_impossible_stretch_removal()
    test_open_string_stretch()
    print("All tests completed successfully!")
