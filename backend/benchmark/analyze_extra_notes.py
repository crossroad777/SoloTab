import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import json
import numpy as np
import collections

def load_ground_truth(path):
    with open(path, 'r', encoding='utf-8') as f: data = json.load(f)
    notes = []
    if 'measures_detailed' in data:
        for m in data['measures_detailed']:
            for r in m.get('notes', []):
                notes.append({'pitch': int(r['pitch']), 'start': float(r.get('beat', 0))})
    return notes

def load_output(path):
    with open(path, 'r', encoding='utf-8') as f: data = json.load(f)
    if isinstance(data, dict) and 'notes' in data: data = data['notes']
    notes = []
    for r in data:
        notes.append({'pitch': int(r['pitch']), 'start': float(r['start_time']), 'vel': float(r.get('velocity', 0)), 'dur_divs': int(r.get('duration_divs', 0)), 'string': int(r.get('string', 0))})
    return notes

def align_sequences(gt_pitches, pred_pitches):
    n, m = len(gt_pitches), len(pred_pitches)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(1, n + 1): dp[i][0] = i
    for j in range(1, m + 1): dp[0][j] = j
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if gt_pitches[i-1] == pred_pitches[j-1] else 2
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
                           
    i, j = n, m
    matches = []
    while i > 0 and j > 0:
        cost = 0 if gt_pitches[i-1] == pred_pitches[j-1] else 2
        if dp[i][j] == dp[i-1][j-1] + cost:
            if cost == 0: matches.append((i-1, j-1))
            i -= 1; j -= 1
        elif dp[i][j] == dp[i-1][j] + 1: i -= 1
        else: j -= 1
    return matches[::-1]

gt_notes = load_ground_truth('backend/ground_truth/romance_forbidden_games.json')
pred_notes = load_output('uploads/20260814-013518-2a384c/notes_assigned.json')

gt_pitches = [n['pitch'] for n in gt_notes]
pred_pitches = [n['pitch'] for n in pred_notes]

matches = align_sequences(gt_pitches, pred_pitches)
matched_pred = {j for i, j in matches}
matched_gt_dict = {j: i for i, j in matches}

extra_indices = [j for j in range(len(pred_notes)) if j not in matched_pred]

print(f"Total Pred: {len(pred_notes)}, Matched: {len(matched_pred)}, Extra: {len(extra_indices)}")

# Analysis
pitch_rel = collections.Counter()
time_loc = []
durations = []
vels = []
strings = collections.Counter()

for j in extra_indices:
    p = pred_notes[j]
    
    # Find nearest GT note by time or by index?
    # Since GT time is beat, we can't easily map beat to seconds without assuming tempo and offset.
    # But wait, we can just find the nearest MATCHED pred note!
    nearest_j = min(matched_pred, key=lambda xj: abs(pred_notes[xj]['start'] - p['start']))
    closest_matched = pred_notes[nearest_j]
    
    pitch_diff = p['pitch'] - closest_matched['pitch']
    pitch_rel[pitch_diff] += 1
    
    time_diff = p['start'] - closest_matched['start']
    time_loc.append(time_diff)
    
    bpm = 88
    # dur_divs=12 -> quarter note
    dur_sec = (60.0 / bpm) * (p['dur_divs'] / 12.0)
    durations.append(dur_sec)
    
    vels.append(p['vel'])
    strings[p['string']] += 1

print("\n--- Pitch Rel (diff from nearest matched note) ---")
for diff, count in pitch_rel.most_common(10): print(f"  {diff:+d}: {count}")

print("\n--- Time Location (from nearest matched note) ---")
# Count within 50ms (0.05s) after
ghosts = sum(1 for dt in time_loc if 0 < dt <= 0.05)
print(f"  Within +50ms (Ghost): {ghosts} ({100*ghosts/len(extra_indices):.1f}%)")
within_100ms = sum(1 for dt in time_loc if 0 < dt <= 0.1)
print(f"  Within +100ms: {within_100ms} ({100*within_100ms/len(extra_indices):.1f}%)")

print("\n--- Duration ---")
short_notes = sum(1 for d in durations if d < 0.1)
print(f"  < 0.1s: {short_notes} ({100*short_notes/len(extra_indices):.1f}%)")
print(f"  Avg: {np.mean(durations):.3f}s, Median: {np.median(durations):.3f}s")

print("\n--- Velocity / Confidence ---")
low_conf = sum(1 for v in vels if v < 0.5)
print(f"  < 0.5: {low_conf} ({100*low_conf/len(extra_indices):.1f}%)")
print(f"  < 0.3: {sum(1 for v in vels if v < 0.3)} ({100*sum(1 for v in vels if v < 0.3)/len(extra_indices):.1f}%)")
print(f"  Avg: {np.mean(vels):.3f}, Median: {np.median(vels):.3f}")

print("\n--- Assigned Strings ---")
for s, c in strings.most_common(): print(f"  String {s}: {c}")
