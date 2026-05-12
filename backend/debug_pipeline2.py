"""
推論パイプライン各段階の正確な検証。
notes.json (CRNN raw) vs notes_assigned.json (Viterbi DP) vs reference
"""
import json, sys
sys.path.append(r"D:\Music\nextchord-solotab\backend")

session_dir = r"D:\Music\nextchord-solotab\uploads\20260512-073742"

with open(f"{session_dir}/notes.json") as f:
    raw_data = json.load(f)
raw_notes = raw_data["notes"]

with open(f"{session_dir}/notes_assigned.json") as f:
    assigned_notes = json.load(f)

with open(f"{session_dir}/beats.json") as f:
    bd = json.load(f)

beats = bd["beats"]
bpm = bd["bpm"]

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
def mn(m):
    m = int(m)
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"

print(f"Raw CRNN: {len(raw_notes)} notes, method: {raw_data['method']}")
print(f"Assigned: {len(assigned_notes)} notes")
print(f"Beats: {len(beats)}, BPM: {bpm}")

# Reference pitches (sorted per bar, melody only)
ref = {
    0: [59, 59, 59, 71, 71, 71, 40],   # B3 x3, B4 x3, bass E2
    1: [59, 59, 59, 67, 69, 71, 40],   # B3x3, G4, A4, B4, bass E2
    2: [59, 59, 59, 64, 66, 67, 40],   # B3x3, E4, F#4, G4, bass E2
    3: [59, 59, 59, 64, 67, 71, 40],   # B3x3, E4, G4, B4, bass E2
    4: [59, 59, 59, 76, 76, 76, 40],   # B3x3, B4(12f) x3, bass E2
    5: [59, 59, 59, 72, 74, 76, 40],   # B3x3, C5, D5, B4(12f), bass E2
    6: [60, 60, 60, 69, 71, 72, 45],   # C4x3, A4, B4, C5, bass A2
    7: [60, 60, 60, 69, 71, 72, 45],   # C4x3, A4, B4, C5, bass A2
}

for bar_num in range(8):
    si = bar_num * 3
    ei = si + 3
    if si >= len(beats):
        break
    t0 = beats[si]
    t1 = beats[ei] if ei < len(beats) else t0 + 3*(60/bpm)
    
    # Raw CRNN
    bar_raw = sorted([int(n["pitch"]) for n in raw_notes 
                      if float(n["start"]) >= t0 and float(n["start"]) < t1])
    
    # Assigned (after Viterbi)
    bar_asgn = sorted([int(n["pitch"]) for n in assigned_notes
                       if float(n["start"]) >= t0 and float(n["start"]) < t1])
    
    # Reference
    bar_ref = sorted(ref.get(bar_num, []))
    
    # Compute pitch set match
    raw_set = set(bar_raw)
    asgn_set = set(bar_asgn)
    ref_set = set(bar_ref)
    
    # Multiset intersection for accuracy
    def multiset_match(gen, reference):
        ref_copy = list(reference)
        matched = 0
        for p in gen:
            if p in ref_copy:
                matched += 1
                ref_copy.remove(p)
        return matched
    
    raw_match = multiset_match(bar_raw, bar_ref)
    asgn_match = multiset_match(bar_asgn, bar_ref)
    
    raw_names = " ".join([f"{mn(p)}" for p in bar_raw])
    asgn_names = " ".join([f"{mn(p)}" for p in bar_asgn])
    ref_names = " ".join([f"{mn(p)}" for p in bar_ref])
    
    print(f"\n--- Bar {bar_num} (t={t0:.2f}-{t1:.2f}s) ---")
    print(f"  Reference({len(bar_ref)}): {ref_names}")
    print(f"  Raw CRNN ({len(bar_raw)}): {raw_names}  match={raw_match}/{len(bar_ref)}")
    print(f"  Assigned ({len(bar_asgn)}): {asgn_names}  match={asgn_match}/{len(bar_ref)}")
    
    # Show what changed between raw and assigned
    if bar_raw != bar_asgn:
        raw_only = sorted(set(bar_raw) - set(bar_asgn))
        asgn_only = sorted(set(bar_asgn) - set(bar_asgn))
        if len(bar_raw) != len(bar_asgn):
            print(f"  ** Note count changed: raw={len(bar_raw)} -> assigned={len(bar_asgn)}")
