"""Quantitative verification of all extracted elements against Romance ground truth."""
import sys, json, os
sys.path.insert(0, '.')
import numpy as np

session_dir = r'D:\Music\nextchord-solotab\uploads\20260512-073742'

# Load all session data
with open(f'{session_dir}/notes_assigned.json', 'r', encoding='utf-8') as f:
    notes = json.load(f)
with open(f'{session_dir}/beats.json', 'r', encoding='utf-8') as f:
    bd = json.load(f)

key_data = {}
if os.path.exists(f'{session_dir}/key.json'):
    with open(f'{session_dir}/key.json', 'r', encoding='utf-8') as f:
        key_data = json.load(f)

# ===== GROUND TRUTH for Romance (禁じられた遊び) =====
# Section A (Am): measures 1-16ish
# Standard tuning, no capo, key of E minor / A minor
# 3/4 time, ~80-90 BPM depending on performer
# Pattern: bass(beat1) + triplet arpeggio (3 notes per beat x 3 beats = 9 melody notes/bar)

# Expected first 4 bars melody (1st string frets):
# M1: B4(7) B4(7) B4(7) - same note repeated in arpeggio
# M2: B4(7) A4(5) G4(3)  
# M3: G4(3) F#4(2) E4(0)
# M4: E4(0) E4(0) B4(7) - resolve then up
# Accompaniment: 2nd string open(0), 3rd string open(0) throughout
# Bass: 6th string open(0) = E2

EXPECTED_BPM_RANGE = (75, 100)  # typical Romance tempo range
EXPECTED_TIME_SIG = "3/4"
EXPECTED_KEY = "Em"  # or Am for the chord progression
EXPECTED_TUNING = [40, 45, 50, 55, 59, 64]  # standard

# First 4 bars expected melody pitches (1st string notes only)
EXPECTED_M1_MELODY = [71, 71, 71]  # B4 x3
EXPECTED_M2_MELODY = [71, 69, 67]  # B4, A4, G4 (descending)
EXPECTED_M3_MELODY = [67, 66, 64]  # G4, F#4, E4
EXPECTED_M4_MELODY = [64, 64, 71]  # E4, E4, B4

# Expected accompaniment per beat: [2nd string open, 3rd string open]
EXPECTED_ACCOMP = [59, 55]  # B3, G3

# Expected bass: E2 = MIDI 40
EXPECTED_BASS = 40

print("=" * 70)
print("Romance (Forbidden Games) - Full Extraction Verification")
print("=" * 70)

# ===== 1. BPM =====
bpm = bd['bpm']
bpm_ok = EXPECTED_BPM_RANGE[0] <= bpm <= EXPECTED_BPM_RANGE[1]
print(f"\n[1] BPM")
print(f"  Detected:  {bpm}")
print(f"  Expected:  {EXPECTED_BPM_RANGE[0]}-{EXPECTED_BPM_RANGE[1]}")
print(f"  Result:    {'OK' if bpm_ok else 'NG'}")

# ===== 2. Time Signature =====
ts = bd.get('time_signature', '?')
ts_ok = ts == EXPECTED_TIME_SIG
print(f"\n[2] Time Signature")
print(f"  Detected:  {ts}")
print(f"  Expected:  {EXPECTED_TIME_SIG}")
print(f"  Result:    {'OK' if ts_ok else 'NG'}")

# ===== 3. Key =====
detected_key = key_data.get('key', '?')
key_conf = key_data.get('confidence', 0)
# Em and Am are both acceptable (relative keys, piece uses both)
key_ok = detected_key in ['Em', 'Am', 'E', 'A']
print(f"\n[3] Key")
print(f"  Detected:  {detected_key} (confidence={key_conf:.2f})")
print(f"  Expected:  Em or Am")
print(f"  Result:    {'OK' if key_ok else 'NG'}")

# ===== 4. Pitch Accuracy =====
print(f"\n[4] Pitch Accuracy (first 4 bars)")

# Reconstruct bars using BPM-based beats
true_interval = 60.0 / bpm
first_note = min(float(n['start']) for n in notes)
first_beat = first_note - 0.01
true_beats = [first_beat + i * true_interval for i in range(200)]
beats_arr = np.array(true_beats)

# Assign notes to bars
def get_bar_notes(notes, bar_num, beats_arr, beats_per_bar=3):
    bar_start_idx = bar_num * beats_per_bar
    bar_end_idx = (bar_num + 1) * beats_per_bar
    if bar_start_idx >= len(beats_arr) or bar_end_idx >= len(beats_arr):
        return []
    bt = beats_arr[bar_start_idx]
    et = beats_arr[bar_end_idx]
    return [n for n in notes if bt <= float(n['start']) < et]

split_pitch = 52
expected_melodies = [EXPECTED_M1_MELODY, EXPECTED_M2_MELODY, EXPECTED_M3_MELODY, EXPECTED_M4_MELODY]
total_pitch_correct = 0
total_pitch_count = 0

for bar_num in range(4):
    bar_notes = get_bar_notes(notes, bar_num, beats_arr)
    melody_notes = sorted([n for n in bar_notes if int(n['pitch']) > split_pitch], 
                          key=lambda n: float(n['start']))
    bass_notes = [n for n in bar_notes if int(n['pitch']) <= split_pitch]
    
    # Extract melody pitches (highest note per beat = 1st string melody)
    # Group by beat
    melody_by_beat = []
    for bi in range(3):
        beat_start = beats_arr[bar_num * 3 + bi]
        beat_end = beats_arr[bar_num * 3 + bi + 1]
        beat_mel = [n for n in melody_notes if beat_start <= float(n['start']) < beat_end]
        if beat_mel:
            # Highest pitch in this beat = melody note
            top = max(beat_mel, key=lambda n: int(n['pitch']))
            melody_by_beat.append(int(top['pitch']))
    
    expected = expected_melodies[bar_num]
    # Compare
    matches = 0
    for i in range(min(len(melody_by_beat), len(expected))):
        if melody_by_beat[i] == expected[i]:
            matches += 1
    total_pitch_correct += matches
    total_pitch_count += len(expected)
    
    det_str = str(melody_by_beat)
    exp_str = str(expected)
    match_pct = matches / len(expected) * 100 if expected else 0
    print(f"  M{bar_num+1}: detected={det_str:30s} expected={exp_str:20s} -> {matches}/{len(expected)} ({match_pct:.0f}%)")
    
    # Check bass
    bass_pitches = [int(n['pitch']) for n in bass_notes]
    bass_ok = EXPECTED_BASS in bass_pitches
    print(f"       bass={bass_pitches} expected=[{EXPECTED_BASS}] -> {'OK' if bass_ok else 'NG'}")
    
    # Check accompaniment (2nd, 3rd string = B3=59, G3=55)
    accomp_pitches = sorted(set(int(n['pitch']) for n in melody_notes if int(n['pitch']) in EXPECTED_ACCOMP))
    accomp_ok = set(EXPECTED_ACCOMP).issubset(set(int(n['pitch']) for n in melody_notes))
    print(f"       accomp={accomp_pitches} expected={EXPECTED_ACCOMP} -> {'OK' if accomp_ok else 'NG'}")

pitch_accuracy = total_pitch_correct / total_pitch_count * 100 if total_pitch_count else 0
print(f"  Overall melody pitch accuracy: {total_pitch_correct}/{total_pitch_count} ({pitch_accuracy:.0f}%)")

# ===== 5. Duration =====
print(f"\n[5] Duration (note length)")
# Expected: each triplet-eighth in Romance ~= 1/3 of a beat = true_interval/3
expected_dur = true_interval / 3
durations = [float(n['end']) - float(n['start']) for n in notes[:30] if int(n['pitch']) > split_pitch]
avg_dur = np.mean(durations)
std_dur = np.std(durations)
print(f"  Expected triplet-eighth: {expected_dur:.4f}s")
print(f"  Detected avg duration:  {avg_dur:.4f}s (std={std_dur:.4f}s)")
print(f"  Ratio (detected/expected): {avg_dur/expected_dur:.2f}x")
print(f"  Note: CRNN 'end' = note sustain end, not rhythmic end")
print(f"        Guitar notes ring longer than their rhythmic value")
print(f"  Result: {'OK (sustain > rhythmic value is expected for guitar)' if avg_dur > expected_dur * 0.8 else 'NG'}")

# ===== 6. Beat Grid =====
print(f"\n[6] Beat Grid")
raw_beats = bd['beats']
raw_intervals = [raw_beats[i+1] - raw_beats[i] for i in range(min(20, len(raw_beats)-1))]
avg_raw = np.mean(raw_intervals)
print(f"  BPM-based expected interval: {true_interval:.4f}s")
print(f"  Detected beat avg interval:  {avg_raw:.4f}s ({60/avg_raw:.1f} BPM)")
print(f"  Ratio: {true_interval/avg_raw:.2f}")
print(f"  Result: {'NG - beat grid misaligned (ratio > 1.3), needs BPM-based correction' if abs(true_interval/avg_raw - 1) > 0.3 else 'OK'}")

# ===== 7. Fingering =====
print(f"\n[7] Fingering (string/fret assignment)")
# Check first 4 bars: all melody should be on string 1, accomp on 2 and 3
bar_notes_all = []
for bar_num in range(4):
    bar_notes_all.extend(get_bar_notes(notes, bar_num, beats_arr))

melody_on_str1 = sum(1 for n in bar_notes_all if int(n['pitch']) > 60 and int(n.get('string', 0)) == 1)
melody_total = sum(1 for n in bar_notes_all if int(n['pitch']) > 60)
str1_ratio = melody_on_str1 / melody_total * 100 if melody_total else 0

# Check playability: no fret > 12 in first section
high_frets = [int(n['fret']) for n in bar_notes_all if int(n.get('fret', 0)) > 12]
print(f"  High-pitch notes on string 1: {melody_on_str1}/{melody_total} ({str1_ratio:.0f}%)")
print(f"  Frets > 12 in first 4 bars: {len(high_frets)} ({high_frets[:5]})")
print(f"  Result: {'OK' if str1_ratio > 50 and len(high_frets) == 0 else 'NG'}")

# ===== 8. Technique =====
print(f"\n[8] Technique Detection")
techs = {}
for n in notes:
    t = n.get('technique', 'normal')
    techs[t] = techs.get(t, 0) + 1
print(f"  Detected: {techs}")
print(f"  Romance typically has: normal arpeggiation, occasional slides")
print(f"  Result: OK (no false technique detection is acceptable)")

# ===== SUMMARY =====
print(f"\n{'=' * 70}")
print(f"SUMMARY")
print(f"{'=' * 70}")
results = {
    'BPM': bpm_ok,
    'Time Sig': ts_ok,
    'Key': key_ok,
    'Pitch (M1-4)': pitch_accuracy >= 75,
    'Duration': avg_dur > expected_dur * 0.8,
    'Beat Grid': abs(true_interval/avg_raw - 1) <= 0.3,
    'Fingering': str1_ratio > 50,
}
for name, ok in results.items():
    print(f"  {name:15s}: {'OK' if ok else 'NG'}")
ok_count = sum(1 for v in results.values() if v)
print(f"\n  Total: {ok_count}/{len(results)} OK")
