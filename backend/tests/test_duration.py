"""Test: straight mode no longer snaps to triplet durations"""
import sys; sys.path.insert(0, r"D:\Music\nextchord-solotab\backend")
from music_theory import snap_duration, VALID_DURATIONS_STRAIGHT, VALID_DURATIONS_TRIPLET

print("=== Straight mode snap test ===")
for raw in range(1, 50):
    snapped, dotted, is_trip = snap_duration(raw, is_triplet=False)
    marker = " <<< TRIPLET!" if is_trip else ""
    if marker or raw <= 15:
        print(f"  raw={raw:3d} -> snapped={snapped:3d} dotted={dotted} trip={is_trip}{marker}")

print("\n=== Triplet mode snap test ===")
for raw in [3, 4, 5, 6, 7, 8, 12, 15, 24]:
    snapped, dotted, is_trip = snap_duration(raw, is_triplet=True)
    print(f"  raw={raw:3d} -> snapped={snapped:3d} dotted={dotted} trip={is_trip}")

print(f"\nSTRAIGHT valid: {VALID_DURATIONS_STRAIGHT}")
print(f"TRIPLET valid:  {VALID_DURATIONS_TRIPLET}")
