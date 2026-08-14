import json, collections
import numpy as np

# Test A: Extra 233 notes pitch histogram
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

gt = json.load(open(r'backend/ground_truth/romance_forbidden_games.json', encoding='utf-8'))
pred = json.load(open(r'uploads/20260814-013518-2a384c/notes_assigned.json', encoding='utf-8'))
if isinstance(pred, dict) and 'notes' in pred: pred = pred['notes']

gt_notes = []
if 'measures_detailed' in gt:
    for m in gt['measures_detailed']:
        for r in m.get('notes', []):
            try:
                p = int(r.get('pitch', r.get('midi', -1)))
                if p >= 0: gt_notes.append({'pitch': p, 'raw': r})
            except: pass

gt_p = [n['pitch'] for n in gt_notes]
pred_p = [int(n['pitch']) for n in pred]
matches = align_sequences(gt_p, pred_p)
matched_pred = {j for i,j in matches}

extra_pitches = collections.Counter([pred_p[j] for j in range(len(pred_p)) if j not in matched_pred])

print("--- Test A: Extra 233 notes pitch histogram ---")
for p, c in extra_pitches.most_common(5):
    print(f"  Pitch {p}: {c} counts")

# Test B: GT duration distribution
# Does GT have duration? Let's check raw note fields
print("\n--- Test B: GT duration distribution ---")
durs = collections.Counter()
for n in gt_notes:
    r = n['raw']
    # Sometimes it's duration or type
    if 'type' in r: durs[r['type']] += 1
    if 'duration' in r: durs[r['duration']] += 1
for k, c in durs.most_common():
    print(f"  {k}: {c}")
if not durs:
    # If no duration, let's print the first few notes
    print("  No explicit duration field found. Raw notes:")
    for n in gt_notes[:5]: print(f"    {n['raw']}")

# Test C: Triplets in GT
# Are there multiple notes with the same beat but fractional beats?
print("\n--- Test C: Simultaneous Triplets in GT ---")
triplet_count = 0
for m in gt['measures_detailed']:
    beats = [r.get('beat', 0) for r in m.get('notes', [])]
    # In romance, a triplet means beats are 1.0, 1.33, 1.67
    fracs = [b - int(b) for b in beats]
    triplets = sum(1 for f in fracs if 0.1 < f < 0.9)
    if triplets > 0: triplet_count += triplets
print(f"  Fractional beats (e.g. 1.33, 1.67) found: {triplet_count}")
