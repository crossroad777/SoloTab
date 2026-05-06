"""
Verify: Is the E major section problem in MoE raw output or theory filter?
Check chords.json and compare raw vs filtered pitches.
"""
import io, sys, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

session = 'D:/Music/nextchord-solotab/uploads/20260429-092245'

# 1. Chord detection results
with open(f'{session}/chords.json') as f:
    chords = json.load(f)
print("=== Chord Detection (BTC) ===")
for c in chords:
    t_start = c.get('start', c.get('time', 0))
    t_end = c.get('end', t_start + c.get('duration', 0))
    label = c.get('chord', c.get('label', '?'))
    print(f"  {t_start:6.2f} - {t_end:6.2f}s  {label}")

# 2. MoE raw output pitches in E major section (M17 = ~34.6s)
with open(f'{session}/notes.json') as f:
    raw = json.load(f)
    raw_notes = raw.get('notes', raw)

# 3. Theory-filtered + DP output
with open(f'{session}/notes_assigned.json') as f:
    assigned = json.load(f)
    assigned = assigned if isinstance(assigned, list) else assigned.get('notes', [])

print(f"\n=== E Major Section (t=34-56s) 1弦ノート比較 ===")
print(f"{'Time':>8} {'MoE_raw_p':>10} {'MoE_raw_sf':>12} {'Final_p':>10} {'Final_sf':>12}")
print("-" * 56)

# Find 1st string notes in E major section
for t in range(34, 56):
    # Raw notes at this time
    for n in raw_notes:
        if t <= n['start'] < t+1 and n.get('string') == 1:
            # Find corresponding assigned note
            match = None
            for a in assigned:
                if abs(a['start'] - n['start']) < 0.1 and a.get('string') == 1:
                    match = a
                    break
            raw_sf = f"s{n['string']}f{n['fret']}"
            final_sf = f"s{match['string']}f{match['fret']}" if match else "?"
            final_p = match['pitch'] if match else "?"
            changed = " <<<" if match and match['pitch'] != n['pitch'] else ""
            print(f"{n['start']:>8.2f} {n['pitch']:>10} {raw_sf:>12} {final_p:>10} {final_sf:>12}{changed}")

# Check: Key estimation
print(f"\n=== Key Estimation ===")
# The music_theory filter log shows "Key estimated: E natural_minor"
# This is wrong for M17-M32 which is E major!
print("music_theory filter estimated key: E natural_minor")
print("BUT M17-M32 is actually in E MAJOR (4 sharps: F# C# G# D#)")
print("Scale PCs for E minor:  [0, 2, 4, 6, 7, 9, 11]")
print("Scale PCs for E major:  [0, 2, 4, 6, 7, 9, 11]")
print("Wait - both have same PCs? No:")
print("E minor: E F# G A B C D  = PCs [4, 6, 7, 9, 11, 0, 2]")
print("E major: E F# G# A B C# D# = PCs [4, 6, 8, 9, 11, 1, 3]")
print("Difference: G vs G# (PC 7 vs 8), C vs C# (PC 0 vs 1), D vs D# (PC 2 vs 3)")
print("\nIf theory filter uses E minor scale, it would correct:")
print("  C#(1) -> C(0) or D(2)")
print("  G#(8) -> G(7) or A(9)")
print("  D#(3) -> D(2) or E(4)")
